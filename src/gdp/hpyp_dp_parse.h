#ifndef _GDP_HPYP_DP_PARSE_H_
#define _GDP_HPYP_DP_PARSE_H_

#include <cstdlib>

#include "transition_parser.h"
#include "eisner_parser.h"
#include "hpyplm/hpyplm.h"
#include "utils/random.h"
#include "pyp/crp.h"

namespace std {

template<typename T, typename... Args>
inline unique_ptr<T> make_unique(Args&&... args) {
  return unique_ptr<T>{new T{args...}};
} 

}

namespace oxlm {

//TODO try to sort out if we can employ polymorphism
//typedef std::unique_ptr<ArcEagerParser> AeParserPtr;
typedef std::vector<std::unique_ptr<ArcEagerParser>> AeParserList;
//typedef std::unique_ptr<ArcStandardParser> AsParserPtr;
typedef std::vector<std::unique_ptr<ArcStandardParser>> AsParserList;

//TODO extend for arc eager
inline void resampleParticles(AsParserList* beam_stack, unsigned num_particles, MT19937& eng) {
  std::vector<double> importance_w(beam_stack->size(), 0); //importance weights
  for (unsigned i = 0; i < importance_w.size(); ++i) {
    //if (beam_stack->at(i)->num_particles()==0) importance_w[i] = 100000; 
    //assume particles have count > 0
    importance_w[i] = beam_stack->at(i)->weighted_importance_weight();
  }

  //resample according to importance weight
  multinomial_distribution_log part_mult(importance_w); 
  std::vector<int> sample_counts(beam_stack->size(), 0);
  for (unsigned i = 0; i < num_particles;) {
    unsigned pi = part_mult(eng);
    //disallow resampling to 0 count particles
    if (beam_stack->at(pi)->num_particles() > 0) {
      ++sample_counts[pi];
      ++i;
    } else {
      std::cerr << "bad sample\n";
    }
  }

  for (unsigned i = 0; i < beam_stack->size(); ++i) {
    beam_stack->at(i)->set_num_particles(sample_counts[i]);
    beam_stack->at(i)->reset_importance_weight();
  }
}

inline void resampleParticles(AeParserList* beam_stack, unsigned num_particles, MT19937& eng) {
  std::vector<double> importance_w(beam_stack->size(), 0); //importance weights
  for (unsigned i = 0; i < importance_w.size(); ++i) {
    //if (beam_stack->at(i)->num_particles()==0) importance_w[i] = 100000; 
    //assume particles have count > 0
    importance_w[i] = beam_stack->at(i)->weighted_importance_weight();
  }

  //resample according to importance weight
  multinomial_distribution_log part_mult(importance_w); 
  std::vector<int> sample_counts(beam_stack->size(), 0);
  for (unsigned i = 0; i < num_particles;) {
    unsigned pi = part_mult(eng);
    //disallow resampling to 0 count particles
    if (beam_stack->at(pi)->num_particles() > 0) {
      ++sample_counts[pi];
      ++i;
    } else {
      std::cerr << "bad sample\n";
    }
  }

  for (unsigned i = 0; i < beam_stack->size(); ++i) {
    beam_stack->at(i)->set_num_particles(sample_counts[i]);
    beam_stack->at(i)->reset_importance_weight();
  }
}

inline double harmonicProbability(WordIndex i, WordIndex j, int n) {
  //head i, dependent j, sent length n (indices 1 to n)
  double Z = 0;
  for (int k = 1; k <= n; ++k) {
    if (!(k==i))
      Z += 1.0/(std::abs(i-k));
  }

  double s = 1.0/(std::abs(i-j));
  return s/Z;
}


//sample a derivation for the gold parse, given the current model
//4-way decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcEagerParser particleEagerGoldParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned num_particles, bool resample, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
    
  //std::cout << "gold arcs: ";
  //gold_dep.print_arcs();

  AeParserList beam_stack; 
  beam_stack.push_back(std::make_unique<ArcEagerParser>(sent, tags, static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift
      Words r_ctx = beam_stack[j]->reduce_context();

      int num_samples = beam_stack[j]->num_particles();

      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double leftarcreducep = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
      double rightarcshiftp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
      double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx); 
      double noarcp = reducep + shiftp;
      //double reducep = reduceleftarcp + reducerightarcp; 
      
      kAction oracle_next = beam_stack[j]->oracleNext(gold_dep);
      //only ambiguity is if oracle_next==re - sh also allowed

      if (oracle_next==kAction::la) {
        beam_stack.push_back(std::make_unique<ArcEagerParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(leftarcreducep); 
        beam_stack.back()->add_importance_weight(leftarcreducep); 
        beam_stack[j]->set_num_particles(0);
      } else if (oracle_next==kAction::ra) {
        Words t_ctx = beam_stack[j]->tag_context(kAction::ra);
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->rightArc();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(rightarcshiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
      } else {
        if (oracle_next==kAction::re) {
          //enforce at least 1 particle to reduce
          std::vector<int> sample_counts = {0, 1}; //shift, reduce
          std::vector<double> distr = {shiftp, reducep};
          multinomial_distribution<double> mult(distr); 
          for (int k = 1; k < num_samples; k++) {
            WordId act = mult(eng);
            ++sample_counts[act];
          }

          if (sample_counts[1] > 0) {
            beam_stack.push_back(std::make_unique<ArcEagerParser>(*beam_stack[j]));
            beam_stack.back()->reduce();
            beam_stack.back()->add_particle_weight(reducep); 
            beam_stack.back()->add_importance_weight(reducep/noarcp);
            beam_stack.back()->set_num_particles(sample_counts[1]); 
            beam_stack[j]->set_num_particles(sample_counts[0]);
            beam_stack[j]->add_importance_weight(shiftp/noarcp);
          }  
          
        }

        //shift allowed
        Words t_ctx = beam_stack[j]->tag_context(kAction::sh);
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp);
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
      }
    }
       
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_ae); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      if (beam_stack.size() > 0)
        resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
  
  //std::cout << "Beam size: " << beam_stack.size();
  int active_particle_count = 0;
  for (int j = 0; j < beam_stack.size(); ++j)
    if (beam_stack[j]->num_particles() > 0)
      ++active_particle_count;
  //std::cout << " -> " << active_particle_count << " without null \n";

  ///completion
  bool has_more_states = true;

  while (has_more_states) {
    has_more_states = false;
    //unsigned cur_beam_size = beam_stack.size();
    //std::cout << cur_beam_size << ": ";

    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->is_terminal_configuration()) {
        //add paths for reduce actions
        has_more_states = true; 
        Words r_ctx = beam_stack[j]->reduce_context();
        //change for with stop word
        //double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
       
        kAction oracle_next = beam_stack[j]->oracleNext(gold_dep);
        if (oracle_next==kAction::re) {
          beam_stack[j]->reduce();
          //beam_stack[j]->add_particle_weight(reducep); 
          //beam_stack[j]->add_importance_weight(reducep); 
        } else {
          beam_stack[j]->set_num_particles(0);
        }
      }
    }
    //std::cerr << std::endl;
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_ae); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cerr << beam_stack.size() << " ";

  //just take 1 sample
  if (beam_stack.size() > 0)
    resampleParticles(&beam_stack, 1, eng);
  for (unsigned i = 0; i < beam_stack.size(); ++i) {
    if (beam_stack[i]->num_particles() == 1) {
      //beam_stack[i]->print_arcs();
      //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
      //std::cout << "  Dir Accuracy: " << dir_acc;
      std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;

      return ArcEagerParser(*beam_stack[i]); 
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcEagerParser(sent, tags);  
}

//sample a derivation for the gold parse, given the current model
//three-way decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcStandardParser particleGoldParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned num_particles, bool resample, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
    
  //std::cout << "gold arcs: ";
  //gold_dep.print_arcs();
  
  AsParserList beam_stack; 
  beam_stack.push_back(std::make_unique<ArcStandardParser>(sent, tags, static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift
      Words r_ctx = beam_stack[j]->reduce_context();

      int num_samples = beam_stack[j]->num_particles();
      std::vector<int> sample_counts = {0, 0, 0}; //shift, reduceleftarc, reducerightarc

      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double reduceleftarcp = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
      double reducerightarcp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
      double reducep = reduceleftarcp + reducerightarcp; 
      
      kAction oracle_next = beam_stack[j]->oracleNext(gold_dep);

      if (oracle_next==kAction::sh) {
        //only shift is allowed
        sample_counts[0] += num_samples;
        if (beam_stack[j]->stack_depth() >= 2)
          beam_stack[j]->add_importance_weight(shiftp);  
      } else {
        //enforce at least one particle to reduce
        std::vector<double> distr; //= {shiftp, reduceleftarcp, reducerightarcp};
        if (oracle_next==kAction::la) {
          distr = {shiftp, reducep, 0};
          sample_counts[1] =  1;            
        }
        if (oracle_next==kAction::ra) {
          distr = {shiftp, 0, reducep};
          sample_counts[2] =  1;            
        }

        multinomial_distribution<double> mult(distr); 
        for (int k = 1; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }
      }
      
     if (sample_counts[2] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->rightArc();
        beam_stack.back()->add_particle_weight(reducerightarcp); 
        beam_stack.back()->add_importance_weight(reducerightarcp/reducep); 
        beam_stack.back()->set_num_particles(sample_counts[2]); 
      } else if (sample_counts[1] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp); 
        beam_stack.back()->add_importance_weight(reduceleftarcp/reducep); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      //perform shift if > 0 samples
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Words t_ctx = beam_stack[j]->tag_context();
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp);
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
     
  ///completion
  bool has_more_states = true;

  while (has_more_states) {
    has_more_states = false;
    //unsigned cur_beam_size = beam_stack.size();
    //std::cerr << cur_beam_size << ": ";

    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->is_terminal_configuration()) {
        //add paths for reduce actions
        has_more_states = true; 
        Words r_ctx = beam_stack[j]->reduce_context();

        double reduceleftarcp = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
        double reducerightarcp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
        
        kAction oracle_next = beam_stack[j]->oracleNext(gold_dep);
        //std::cerr << " (" << beam_stack[j]->num_particles() << ") " << static_cast<WordId>(oracle_next);
        if (oracle_next==kAction::re) {
          //invalid, so let particles die (else all particles are moved on)
          beam_stack[j]->set_num_particles(0);
        } else if (oracle_next == kAction::ra) {
          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->add_importance_weight(reducerightarcp); 
        } else if (oracle_next == kAction::la) {
          beam_stack[j]->leftArc();
          beam_stack[j]->add_particle_weight(reduceleftarcp); 
          beam_stack[j]->add_importance_weight(reduceleftarcp); 
        }
      }
    }
    //std::cerr << std::endl;

    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //    ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cerr << beam_stack.size() << " ";

  //just take 1 sample
  if (beam_stack.size() > 0)
    resampleParticles(&beam_stack, 1, eng);
  for (unsigned i = 0; i < beam_stack.size(); ++i) {
    if (beam_stack[i]->num_particles() == 1) {
      //beam_stack[i]->print_arcs();
      //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
      //std::cout << "  Dir Accuracy: " << dir_acc;
      //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;

      return ArcStandardParser(*beam_stack[i]); 
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcStandardParser(sent, tags);  
}

//sample a derivation for the gold parse, given the current model
//binary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser particleGoldParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned num_particles, bool resample, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
    
  // std::cout << "gold arcs: ";
  //gold_dep.print_arcs();

  AsParserList beam_stack; 
  beam_stack.push_back(std::make_unique<ArcStandardParser>(sent, tags, static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift
      Words r_ctx = beam_stack[j]->reduce_context();
      Words a_ctx = beam_stack[j]->arc_context();

      int num_samples = beam_stack[j]->num_particles();
      std::vector<int> sample_counts = {0, 0, 0}; //shift, reduceleftarc, reducerightarc

      double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double leftarcp = arc_lm.prob(static_cast<WordId>(kAction::la), a_ctx);
      double rightarcp = arc_lm.prob(static_cast<WordId>(kAction::ra), a_ctx);
      
      kAction oracle_next = beam_stack[j]->oracleNext(gold_dep);

      if (oracle_next==kAction::sh) {
        //only shift is allowed
        sample_counts[0] += num_samples;
        if (beam_stack[j]->stack_depth() >= 2)
          beam_stack[j]->add_importance_weight(shiftp);  
      } else {
        //enforce at least one particle to reduce
        std::vector<double> distr; //= {shiftp, reduceleftarcp, reducerightarcp};
        if (oracle_next==kAction::la) {
          distr = {shiftp, reducep, 0};
          sample_counts[1] =  1;            
        }
        if (oracle_next==kAction::ra) {
          distr = {shiftp, 0, reducep};
          sample_counts[2] =  1;            
        }

        multinomial_distribution<double> mult(distr); 
        for (int k = 1; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }
      }
      
     if (sample_counts[2] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->rightArc();
        beam_stack.back()->add_particle_weight(reducep*rightarcp); 
        beam_stack.back()->add_importance_weight(rightarcp); 
        beam_stack.back()->set_num_particles(sample_counts[2]); 
      } else if (sample_counts[1] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reducep*leftarcp); 
        beam_stack.back()->add_importance_weight(leftarcp); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      //perform shift if > 0 samples
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Words t_ctx = beam_stack[j]->tag_context();
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp);
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
     
  ///completion
  AsParserList final_beam; 
  bool has_more_states = true;

  while (has_more_states) {
    has_more_states = false;
    //unsigned cur_beam_size = beam_stack.size();
    //std::cerr << cur_beam_size << ": ";

    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->is_terminal_configuration()) {
        //add paths for reduce actions
        has_more_states = true; 
        Words r_ctx = beam_stack[j]->reduce_context();
        Words a_ctx = beam_stack[j]->arc_context();
        double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
        double reduceleftarcp = reducep*arc_lm.prob(static_cast<WordId>(kAction::la), a_ctx);
        double reducerightarcp = reducep*arc_lm.prob(static_cast<WordId>(kAction::ra), a_ctx);
        
        kAction oracle_next = beam_stack[j]->oracleNext(gold_dep);
        //std::cerr << " (" << beam_stack[j]->num_particles() << ") " << static_cast<WordId>(oracle_next);
        if (oracle_next==kAction::re) {
          //invalid, so let particles die (else all particles are moved on)
          beam_stack[j]->set_num_particles(0);
        } else if (oracle_next == kAction::ra) {
          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->add_importance_weight(reducerightarcp); 
        } else if (oracle_next == kAction::la) {
          beam_stack[j]->leftArc();
          beam_stack[j]->add_particle_weight(reduceleftarcp); 
          beam_stack[j]->add_importance_weight(reduceleftarcp); 
        }
      }
    }
    //std::cerr << std::endl;

    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //    ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cerr << beam_stack.size() << " ";

  //just take 1 sample
  if (beam_stack.size() > 0)
    resampleParticles(&beam_stack, 1, eng);
  for (unsigned i = 0; i < beam_stack.size(); ++i) {
    if (beam_stack[i]->num_particles() == 1) {
      //beam_stack[i]->print_arcs();
      //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
      //std::cout << "  Dir Accuracy: " << dir_acc;
      //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;

      return ArcStandardParser(*beam_stack[i]); 
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcStandardParser(sent, tags);  
}

//four-way decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcEagerParser particleEagerParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned num_particles, bool resample, bool take_max, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
    
  //std::cout << "gold arcs: ";
  //gold_dep.print_arcs();

  AeParserList beam_stack; 
  beam_stack.push_back(std::make_unique<ArcEagerParser>(sent, tags, static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    unsigned shift_beam_size = beam_stack.size();
    //std::cout << i << ": " << shift_beam_size << " ";
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if ((beam_stack[j]->num_particles()==0) || 
          ((j >= shift_beam_size) && (beam_stack[j]->last_action() == kAction::ra)))
        continue;
       
      //sample a sequence of possible actions leading up to the next shift or ra
      Words r_ctx = beam_stack[j]->reduce_context();

      int num_samples = beam_stack[j]->num_particles();

      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double leftarcreducep = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
      double rightarcshiftp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
      double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx); 
      //double totalreducep = reducep + leftarcreducep;

      std::vector<int> sample_counts = {0, 0, 0, 0}; //shift, la, ra, re
      
      if (!beam_stack[j]->left_arc_valid())
        leftarcreducep = 0;
      if (!beam_stack[j]->reduce_valid())
        reducep = 0;

      std::vector<double> distr = {shiftp, leftarcreducep, rightarcshiftp, reducep};
      multinomial_distribution<double> mult(distr); 
      for (int k = 0; k < num_samples; k++) {
        WordId act = mult(eng);
        ++sample_counts[act];
      }
      
      //for reduce actions: 
      if (sample_counts[1] > 0) {
        beam_stack.push_back(std::make_unique<ArcEagerParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(leftarcreducep); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      if (sample_counts[3] > 0) {
        beam_stack.push_back(std::make_unique<ArcEagerParser>(*beam_stack[j]));
        beam_stack.back()->reduce();
        beam_stack.back()->add_particle_weight(reducep); 
        beam_stack.back()->set_num_particles(sample_counts[3]); 
      }

      //for shift actions
      if (sample_counts[2] > 0) {
        beam_stack.push_back(std::make_unique<ArcEagerParser>(*beam_stack[j]));

        Words t_ctx = beam_stack[j]->tag_context(kAction::ra);
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack.back()->rightArc();
        beam_stack.back()->add_importance_weight(wordp); 
        beam_stack.back()->add_importance_weight(tagp); 
        beam_stack.back()->add_particle_weight(rightarcshiftp); 
        beam_stack.back()->add_particle_weight(wordp); 
        beam_stack.back()->add_particle_weight(tagp); 
        beam_stack.back()->set_num_particles(sample_counts[2]);
      } 
      
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Words t_ctx = beam_stack[j]->tag_context(kAction::sh);
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_ae); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      if (beam_stack.size() > 0)
        resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
     
  ///completion
  bool has_more_states = true;
  //std::cout << " compl " << std::endl;

  while (has_more_states) {
    has_more_states = false;
    unsigned cur_beam_size = beam_stack.size();
    //std::cout << cur_beam_size << ": ";

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->is_terminal_configuration()) {
        //add paths for reduce actions
        //std::cout << beam_stack[j]->stack_depth() << "," << beam_stack[j]->num_particles() << " ";
        has_more_states = true; 
        Words r_ctx = beam_stack[j]->reduce_context();
        double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
        
        if (beam_stack[j]->reduce_valid()) {
          beam_stack[j]->reduce();
          beam_stack[j]->add_particle_weight(reducep); 
          beam_stack[j]->add_importance_weight(reducep); 
        } else {
          beam_stack[j]->set_num_particles(0);
        }
      }
    }
    //std::cout << std::endl;

    //no point in resampling here
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_ae); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cout << "Final beam size: " << beam_stack.size() << std::endl;


  if ((beam_stack.size() > 0) && take_max) {
    //resampleParticles(&beam_stack, num_particles, eng);
    std::sort(beam_stack.begin(), beam_stack.end(), cmp_particle_ptr_weights_ae); 
    return ArcEagerParser(*beam_stack[0]);
  } else if (beam_stack.size() > 0) {
    //just take 1 sample
    resampleParticles(&beam_stack, 1, eng);
    for (unsigned i = 0; i < beam_stack.size(); ++i) {
      if (beam_stack[i]->num_particles() == 1) {
        //beam_stack[i]->print_arcs();
        //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
        //std::cout << "  Dir Accuracy: " << dir_acc;
        //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;
        return ArcEagerParser(*beam_stack[i]); 
      }
    }
  }

  //std::cout << "no parse found" << std::endl;
  return ArcEagerParser(sent, tags);  
}

//four-way decisions
template<unsigned kShiftOrder, unsigned kTagOrder>
ArcEagerParser particleInitEagerParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned num_particles, bool resample, bool take_max, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kTagOrder>& tag_lm) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
    
  //std::cout << "gold arcs: ";
  //gold_dep.print_arcs();
  AeParserList beam_stack; 
  beam_stack.push_back(std::make_unique<ArcEagerParser>(sent, tags, static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    unsigned shift_beam_size = beam_stack.size();
    //std::cout << i << ": " << shift_beam_size << " ";
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if ((beam_stack[j]->num_particles()==0) || 
          ((j >= shift_beam_size) && (beam_stack[j]->last_action() == kAction::ra)))
        continue;
       
      //sample a sequence of possible actions leading up to the next shift or ra
      int num_samples = beam_stack[j]->num_particles();

      double reducep = 0.4;
      double rightarcshiftp = 0.3;
      double leftarcreducep = 0.3;
      double shiftp = 0.0;

      WordIndex s_first = beam_stack[j]->stack_top();
      WordIndex b_first = beam_stack[j]->buffer_next();

      leftarcreducep = harmonicProbability(b_first, s_first, sent.size()-1)/2; ///2
      rightarcshiftp = harmonicProbability(s_first, b_first, sent.size()-1)/2; ///2
      shiftp = (1.0 - leftarcreducep - rightarcshiftp)/2;
      reducep = (1.0 - leftarcreducep - rightarcshiftp)/2;

      std::vector<int> sample_counts = {0, 0, 0, 0}; //shift, la, ra, re
      
      if (!beam_stack[j]->left_arc_valid())
        leftarcreducep = 0;
      if (!beam_stack[j]->reduce_valid())
        reducep = 0;
      if (beam_stack[j]->stack_depth() == 1) {
        if (beam_stack[j]->root_has_child()) {
          rightarcshiftp = 0;
          shiftp = 1;
        } else {
          rightarcshiftp = 1.0/(static_cast<int>(sent.size())-1);
          shiftp = 1 - rightarcshiftp;
        }
      }

      std::vector<double> distr = {shiftp, leftarcreducep, rightarcshiftp, reducep};
      multinomial_distribution<double> mult(distr); 
      for (int k = 0; k < num_samples; k++) {
        WordId act = mult(eng);
        ++sample_counts[act];
      }
      
      //for reduce actions: 
      if (sample_counts[1] > 0) {
        beam_stack.push_back(std::make_unique<ArcEagerParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(leftarcreducep); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      if (sample_counts[3] > 0) {
        beam_stack.push_back(std::make_unique<ArcEagerParser>(*beam_stack[j]));
        beam_stack.back()->reduce();
        beam_stack.back()->add_particle_weight(reducep); 
        beam_stack.back()->set_num_particles(sample_counts[3]); 
      }

      //for shift actions
      if (sample_counts[2] > 0) {
        beam_stack.push_back(std::make_unique<ArcEagerParser>(*beam_stack[j]));

        Words t_ctx = beam_stack[j]->tag_context(kAction::ra);
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack.back()->rightArc();
        beam_stack.back()->add_importance_weight(wordp); 
        beam_stack.back()->add_importance_weight(tagp); 
        beam_stack.back()->add_particle_weight(rightarcshiftp); 
        beam_stack.back()->add_particle_weight(wordp); 
        beam_stack.back()->add_particle_weight(tagp); 
        beam_stack.back()->set_num_particles(sample_counts[2]);
      } 
      
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Words t_ctx = beam_stack[j]->tag_context(kAction::sh);
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_ae); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      if (beam_stack.size() > 0)
        resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
     
  ///completion
  bool has_more_states = true;
  //std::cout << " compl " << std::endl;

  while (has_more_states) {
    has_more_states = false;
    unsigned cur_beam_size = beam_stack.size();
    //std::cout << cur_beam_size << ": ";

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->is_terminal_configuration()) {
        //add paths for reduce actions
        //std::cout << beam_stack[j]->stack_depth() << "," << beam_stack[j]->num_particles() << " ";
        has_more_states = true; 
        double reducep = 1.0;
        
        if (beam_stack[j]->reduce_valid()) {
          beam_stack[j]->reduce();
          beam_stack[j]->add_particle_weight(reducep); 
          beam_stack[j]->add_importance_weight(reducep); 
        } else {
          beam_stack[j]->set_num_particles(0);
        }
      }
    }
    //std::cout << std::endl;

    //no point in resampling here
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_ae); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cout << "Final beam size: " << beam_stack.size() << std::endl;


  if ((beam_stack.size() > 0) && take_max) {
    //resampleParticles(&beam_stack, num_particles, eng);
    std::sort(beam_stack.begin(), beam_stack.end(), cmp_particle_ptr_weights_ae); 
    return ArcEagerParser(*beam_stack[0]);
  } else if (beam_stack.size() > 0) {
    //just take 1 sample
    resampleParticles(&beam_stack, 1, eng);
    for (unsigned i = 0; i < beam_stack.size(); ++i) {
      if (beam_stack[i]->num_particles() == 1) {
        //beam_stack[i]->print_arcs();
        //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
        //std::cout << "  Dir Accuracy: " << dir_acc;
        //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;
        return ArcEagerParser(*beam_stack[i]); 
      }
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcEagerParser(sent, tags);  
}

//ternary decisions
//use for unsup initialization
template<unsigned kShiftOrder, unsigned kTagOrder>
ArcStandardParser particleInitParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned num_particles, bool resample, bool take_max, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kTagOrder>& tag_lm) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
    
  //std::cout << "gold arcs: ";
  //gold_dep.print_arcs();
  
  double rightarcp = 0.5;

  AsParserList beam_stack; 
  beam_stack.push_back(std::make_unique<ArcStandardParser>(sent, tags, static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift
      int num_samples = beam_stack[j]->num_particles();

      double reducerightarcp = rightarcp;
      double reduceleftarcp = 1 - rightarcp;
      double shiftp = 0.0;

      std::vector<int> sample_counts = {0, 0, 0}; //shift, reduceleftarc, reducerightarc

      if (beam_stack[j]->stack_depth() < 2) {
        //only shift is allowed
        sample_counts[0] += num_samples;
      } else {
        //WordIndex s_first = beam_stack[j]->stack_top();
        //WordIndex s_second = beam_stack[j]->stack_top_second();

        //double reduceleftarcp = harmonicProbability(s_first, s_second, sent.size()-1)/2; ///2
        //double reducerightarcp = harmonicProbability(s_second, s_first, sent.size()-1)/2; ///2
        //double shiftp = 1.0 - reduceleftarcp - reducerightarcp;

        if (beam_stack[j]->stack_depth() == 2) {
          //left arc disallowed
          reduceleftarcp = 0;
          if (beam_stack[j]->root_has_child()) {
            reducerightarcp = 0;
          } else {
            reducerightarcp = 1.0/(static_cast<int>(sent.size())-1);
          }
          
          shiftp = 1 - reducerightarcp;
        }

        std::vector<double> distr = {shiftp, reduceleftarcp, reducerightarcp};
        multinomial_distribution<double> mult(distr); 
        for (int k = 0; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }
      }        
     
      if ((sample_counts[1] > 0) && (sample_counts[2] > 0)) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));

        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp);
        beam_stack.back()->set_num_particles(sample_counts[1]); 

        beam_stack.rbegin()[1]->rightArc();
        beam_stack.rbegin()[1]->add_particle_weight(reducerightarcp); 
        beam_stack.rbegin()[1]->set_num_particles(sample_counts[2]); 

      } else if (sample_counts[2] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->rightArc();
        beam_stack.back()->add_particle_weight(reducerightarcp); 
        beam_stack.back()->set_num_particles(sample_counts[2]); 
      } else if (sample_counts[1] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      //perform shift if > 0 samples
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Words t_ctx = beam_stack[j]->tag_context();
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
     
  ///completion
  AsParserList final_beam; 
  bool has_more_states = true;

  while (has_more_states) {
    has_more_states = false;
    unsigned cur_beam_size = beam_stack.size();

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->is_terminal_configuration()) {
        //add paths for reduce actions
        has_more_states = true; 
  
        WordIndex s_first = beam_stack[j]->stack_top();
        WordIndex s_second = beam_stack[j]->stack_top_second();

        //double reduceleftarcp = 1 - rightarcp;
        //double reducerightarcp = rightarcp;
        double reduceleftarcp = harmonicProbability(s_first, s_second, sent.size()-1)/2; ///2
        double reducerightarcp = harmonicProbability(s_second, s_first, sent.size()-1)/2; ///2
        double reducep = reduceleftarcp + reducerightarcp;
        
        int num_samples = beam_stack[j]->num_particles();
        std::vector<int> sample_counts = {0, 0}; //reduceleftarc, reducerightarc

        if (beam_stack[j]->stack_depth() == 2) {
          //only allow right arc
          sample_counts[1] = num_samples;
        } else {
          std::vector<double> distr = {reduceleftarcp, reducerightarcp};
          multinomial_distribution<double> mult(distr); 
          for (int k = 0; k < num_samples; k++) {
            WordId act = mult(eng);
            ++sample_counts[act];
          }
        }

        if ((sample_counts[0]>0) && (sample_counts[1]>0)) {
          beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));

          beam_stack.back()->leftArc();
          beam_stack.back()->add_particle_weight(reduceleftarcp);
          beam_stack.back()->set_num_particles(sample_counts[0]); 
          beam_stack.back()->add_importance_weight(reducep); 

          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->set_num_particles(sample_counts[1]); 
          beam_stack[j]->add_importance_weight(reducep); 
          
        } else if (sample_counts[1] > 0) {
          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->set_num_particles(sample_counts[1]); 
          beam_stack[j]->add_importance_weight(reducep); 

        } else if (sample_counts[0] > 0) {
          beam_stack[j]->leftArc();
          beam_stack[j]->add_particle_weight(reduceleftarcp); 
          beam_stack[j]->set_num_particles(sample_counts[0]); 
          beam_stack[j]->add_importance_weight(reducep); 
        }
      }
    }

    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //    ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cout << "Final beam size: " << beam_stack.size();

  if ((beam_stack.size() > 0) && take_max) {
    //resampleParticles(&beam_stack, num_particles, eng);
    std::sort(beam_stack.begin(), beam_stack.end(), cmp_particle_ptr_weights_as); 
    return ArcStandardParser(*beam_stack[0]);
  } else if (beam_stack.size() > 0) {
    //just take 1 sample
    resampleParticles(&beam_stack, 1, eng);
    for (unsigned i = 0; i < beam_stack.size(); ++i) {
      if (beam_stack[i]->num_particles() == 1) {
        //beam_stack[i]->print_arcs();
        //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
        //std::cout << "  Dir Accuracy: " << dir_acc << std::endl;
        //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;
        return ArcStandardParser(*beam_stack[i]); 
      }
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcStandardParser(sent, tags);  
}

//ternary decisions
//correspond to dynamic oracle
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcStandardParser particleDynamicGoldParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned num_particles, bool resample, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
    
  //std::cout << "gold arcs: ";
  //gold_dep.print_arcs();

  AsParserList beam_stack; 
  beam_stack.push_back(std::make_unique<ArcStandardParser>(sent, tags, static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift
      Words r_ctx = beam_stack[j]->reduce_context();

      int num_samples = beam_stack[j]->num_particles();

      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double reduceleftarcp = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
      double reducerightarcp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
      //double reducep = reduceleftarcp + reducerightarcp; 

      std::vector<int> sample_counts = {0, 0, 0}; //shift, reduceleftarc, reducerightarc

      if (beam_stack[j]->stack_depth() < 2) {
        //only shift is allowed
        sample_counts[0] += num_samples;
      } else {
        kAction oracle_next = beam_stack[j]->oracleNext(gold_dep);

        //fix oracle as first sample
        if (oracle_next==kAction::sh) {
          sample_counts[0] = 1;
        } else if (oracle_next==kAction::la) {
          sample_counts[1] = 1;            
        }
        else if (oracle_next==kAction::ra) {
          sample_counts[2] = 1;            
        }

        if (beam_stack[j]->stack_depth() == 2) {
          //left arc disallowed
          reduceleftarcp = 0;
          //reducerightarcp = reducep;
        }

        std::vector<double> distr = {shiftp, reduceleftarcp, reducerightarcp};
        multinomial_distribution<double> mult(distr); 
        for (int k = 1; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }
      }        
     
      if ((sample_counts[1] > 0) && (sample_counts[2] > 0)) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));

        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp);
        beam_stack.back()->set_num_particles(sample_counts[1]); 

        beam_stack.rbegin()[1]->rightArc();
        beam_stack.rbegin()[1]->add_particle_weight(reducerightarcp); 
        beam_stack.rbegin()[1]->set_num_particles(sample_counts[2]); 

      } else if (sample_counts[2] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->rightArc();
        beam_stack.back()->add_particle_weight(reducerightarcp); 
        beam_stack.back()->set_num_particles(sample_counts[2]); 
      } else if (sample_counts[1] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      //perform shift if > 0 samples
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Words t_ctx = beam_stack[j]->tag_context();
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
     
  ///completion
  AsParserList final_beam; 
  bool has_more_states = true;

  while (has_more_states) {
    has_more_states = false;
    unsigned cur_beam_size = beam_stack.size();

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->is_terminal_configuration()) {
        //add paths for reduce actions
        has_more_states = true; 
        Words r_ctx = beam_stack[j]->reduce_context();
  
        double reduceleftarcp = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
        double reducerightarcp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
        double reducep = reduceleftarcp + reducerightarcp; 
        
        int num_samples = beam_stack[j]->num_particles();
        std::vector<int> sample_counts = {0, 0}; //reduceleftarc, reducerightarc

        if (beam_stack[j]->stack_depth() == 2) {
          //only allow right arc
          sample_counts[1] = num_samples;
        } else {
          //ensure oracle arc has at least 1 particle
          kAction oracle_next = beam_stack[j]->oracleNext(gold_dep);
          if (oracle_next==kAction::re) {
            ++num_samples; //no oracle arc
          } else if (oracle_next == kAction::ra) {
            sample_counts[1] = 1;
          } else if (oracle_next == kAction::la) {
            sample_counts[1] = 1;
          }

          std::vector<double> distr = {reduceleftarcp, reducerightarcp};
          multinomial_distribution<double> mult(distr); 
          for (int k = 1; k < num_samples; k++) {
            WordId act = mult(eng);
            ++sample_counts[act];
          }
        }

        if ((sample_counts[0]>0) && (sample_counts[1]>0)) {
          beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));

          beam_stack.back()->leftArc();
          beam_stack.back()->add_particle_weight(reduceleftarcp);
          beam_stack.back()->set_num_particles(sample_counts[0]); 
          beam_stack.back()->add_importance_weight(reducep); 

          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->set_num_particles(sample_counts[1]); 
          beam_stack[j]->add_importance_weight(reducep); 
          
        } else if (sample_counts[1] > 0) {
          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->set_num_particles(sample_counts[1]); 
          beam_stack[j]->add_importance_weight(reducep); 

        } else if (sample_counts[0] > 0) {
          beam_stack[j]->leftArc();
          beam_stack[j]->add_particle_weight(reduceleftarcp); 
          beam_stack[j]->set_num_particles(sample_counts[0]); 
          beam_stack[j]->add_importance_weight(reducep); 
        }
      }
    }

    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //    ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cout << "Final beam size: " << beam_stack.size();

  if (beam_stack.size() > 0) {
    //resampleParticles(&beam_stack, num_particles, eng);
    std::sort(beam_stack.begin(), beam_stack.end(), cmp_particle_ptr_weights_as); 
    return ArcStandardParser(*beam_stack[0]);
  } else if (beam_stack.size() > 0) {
    //just take 1 sample
    resampleParticles(&beam_stack, 1, eng);
    for (unsigned i = 0; i < beam_stack.size(); ++i) {
      if (beam_stack[i]->num_particles() == 1) {
        //beam_stack[i]->print_arcs();
        //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
        //std::cout << "  Dir Accuracy: " << dir_acc;
        //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;
        return ArcStandardParser(*beam_stack[i]); 
      }
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcStandardParser(sent, tags);  
}



//ternary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcStandardParser particleParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned num_particles, bool resample, bool take_max, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
    
  //std::cout << "gold arcs: ";
  //gold_dep.print_arcs();

  AsParserList beam_stack; 
  beam_stack.push_back(std::make_unique<ArcStandardParser>(sent, tags, static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift
      Words r_ctx = beam_stack[j]->reduce_context();

      int num_samples = beam_stack[j]->num_particles();

      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double reduceleftarcp = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
      double reducerightarcp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
      double reducep = reduceleftarcp + reducerightarcp; 

      std::vector<int> sample_counts = {0, 0, 0}; //shift, reduceleftarc, reducerightarc

      if (beam_stack[j]->stack_depth() < 2) {
        //only shift is allowed
        sample_counts[0] += num_samples;
      } else {
        if (beam_stack[j]->stack_depth() == 2) {
          //left arc disallowed
          reduceleftarcp = 0;
          reducerightarcp = reducep;
        }

        std::vector<double> distr = {shiftp, reduceleftarcp, reducerightarcp};
        multinomial_distribution<double> mult(distr); 
        for (int k = 0; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }
      }        
     
      if ((sample_counts[1] > 0) && (sample_counts[2] > 0)) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));

        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp);
        beam_stack.back()->set_num_particles(sample_counts[1]); 

        beam_stack.rbegin()[1]->rightArc();
        beam_stack.rbegin()[1]->add_particle_weight(reducerightarcp); 
        beam_stack.rbegin()[1]->set_num_particles(sample_counts[2]); 

      } else if (sample_counts[2] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->rightArc();
        beam_stack.back()->add_particle_weight(reducerightarcp); 
        beam_stack.back()->set_num_particles(sample_counts[2]); 
      } else if (sample_counts[1] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      //perform shift if > 0 samples
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Words t_ctx = beam_stack[j]->tag_context();
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
     
  ///completion
  AsParserList final_beam; 
  bool has_more_states = true;

  while (has_more_states) {
    has_more_states = false;
    unsigned cur_beam_size = beam_stack.size();

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->is_terminal_configuration()) {
        //add paths for reduce actions
        has_more_states = true; 
        Words r_ctx = beam_stack[j]->reduce_context();
  
        double reduceleftarcp = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
        double reducerightarcp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
        double reducep = reduceleftarcp + reducerightarcp; 
        
        int num_samples = beam_stack[j]->num_particles();
        std::vector<int> sample_counts = {0, 0}; //reduceleftarc, reducerightarc

        if (beam_stack[j]->stack_depth() == 2) {
          //only allow right arc
          sample_counts[1] = num_samples;
        } else {
          std::vector<double> distr = {reduceleftarcp, reducerightarcp};
          multinomial_distribution<double> mult(distr); 
          for (int k = 0; k < num_samples; k++) {
            WordId act = mult(eng);
            ++sample_counts[act];
          }
        }

        if ((sample_counts[0]>0) && (sample_counts[1]>0)) {
          beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));

          beam_stack.back()->leftArc();
          beam_stack.back()->add_particle_weight(reduceleftarcp);
          beam_stack.back()->set_num_particles(sample_counts[0]); 
          beam_stack.back()->add_importance_weight(reducep); 

          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->set_num_particles(sample_counts[1]); 
          beam_stack[j]->add_importance_weight(reducep); 
          
        } else if (sample_counts[1] > 0) {
          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->set_num_particles(sample_counts[1]); 
          beam_stack[j]->add_importance_weight(reducep); 

        } else if (sample_counts[0] > 0) {
          beam_stack[j]->leftArc();
          beam_stack[j]->add_particle_weight(reduceleftarcp); 
          beam_stack[j]->set_num_particles(sample_counts[0]); 
          beam_stack[j]->add_importance_weight(reducep); 
        }
      }
    }

    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //    ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cout << "Final beam size: " << beam_stack.size();

  if ((beam_stack.size() > 0) && take_max) {
    //resampleParticles(&beam_stack, num_particles, eng);
    std::sort(beam_stack.begin(), beam_stack.end(), cmp_particle_ptr_weights_as); 
    return ArcStandardParser(*beam_stack[0]);
  } else if (beam_stack.size() > 0) {
    //just take 1 sample
    resampleParticles(&beam_stack, 1, eng);
    for (unsigned i = 0; i < beam_stack.size(); ++i) {
      if (beam_stack[i]->num_particles() == 1) {
        //beam_stack[i]->print_arcs();
        //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
        //std::cout << "  Dir Accuracy: " << dir_acc;
        //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;
        return ArcStandardParser(*beam_stack[i]); 
      }
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcStandardParser(sent, tags);  
}


template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser particleParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned num_particles, bool resample, bool take_max, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
    
  //std::cout << "gold arcs: ";
  //gold_dep.print_arcs();

  AsParserList beam_stack; 
  beam_stack.push_back(std::make_unique<ArcStandardParser>(sent, tags, static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift
      Words r_ctx = beam_stack[j]->reduce_context();
      Words a_ctx = beam_stack[j]->arc_context();

      int num_samples = beam_stack[j]->num_particles();

      double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double reduceleftarcp = reducep*arc_lm.prob(static_cast<WordId>(kAction::la), a_ctx);
      double reducerightarcp = reducep*arc_lm.prob(static_cast<WordId>(kAction::ra), a_ctx);

      std::vector<int> sample_counts = {0, 0, 0}; //shift, reduceleftarc, reducerightarc

      if (beam_stack[j]->stack_depth() < 2) {
        //only shift is allowed
        sample_counts[0] += num_samples;
      } else {
        if (beam_stack[j]->stack_depth() == 2) {
          //left arc disallowed
          reduceleftarcp = 0;
          reducerightarcp = reducep;
        }

        std::vector<double> distr = {shiftp, reduceleftarcp, reducerightarcp};
        multinomial_distribution<double> mult(distr); 
        for (int k = 0; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }
      }        
     
      if ((sample_counts[1] > 0) && (sample_counts[2] > 0)) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));

        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp);
        beam_stack.back()->set_num_particles(sample_counts[1]); 

        beam_stack.rbegin()[1]->rightArc();
        beam_stack.rbegin()[1]->add_particle_weight(reducerightarcp); 
        beam_stack.rbegin()[1]->set_num_particles(sample_counts[2]); 

      } else if (sample_counts[2] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->rightArc();
        beam_stack.back()->add_particle_weight(reducerightarcp); 
        beam_stack.back()->set_num_particles(sample_counts[2]); 
      } else if (sample_counts[1] > 0) {
        beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      //perform shift if > 0 samples
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Words t_ctx = beam_stack[j]->tag_context();
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
     
  ///completion
  AsParserList final_beam; 
  bool has_more_states = true;

  while (has_more_states) {
    has_more_states = false;
    unsigned cur_beam_size = beam_stack.size();

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->is_terminal_configuration()) {
        //add paths for reduce actions
        has_more_states = true; 
        Words r_ctx = beam_stack[j]->reduce_context();
        Words a_ctx = beam_stack[j]->arc_context();
        double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
        double reduceleftarcp = reducep*arc_lm.prob(static_cast<WordId>(kAction::la), a_ctx);
        double reducerightarcp = reducep*arc_lm.prob(static_cast<WordId>(kAction::ra), a_ctx);
        
        int num_samples = beam_stack[j]->num_particles();
        std::vector<int> sample_counts = {0, 0}; //reduceleftarc, reducerightarc

        if (beam_stack[j]->stack_depth() == 2) {
          //only allow right arc
          sample_counts[1] = num_samples;
        } else {
          std::vector<double> distr = {reduceleftarcp, reducerightarcp};
          multinomial_distribution<double> mult(distr); 
          for (int k = 0; k < num_samples; k++) {
            WordId act = mult(eng);
            ++sample_counts[act];
          }
        }

        if ((sample_counts[0]>0) && (sample_counts[1]>0)) {
          beam_stack.push_back(std::make_unique<ArcStandardParser>(*beam_stack[j]));

          beam_stack.back()->leftArc();
          beam_stack.back()->add_particle_weight(reduceleftarcp);
          beam_stack.back()->set_num_particles(sample_counts[0]); 
          beam_stack.back()->add_importance_weight(reducep); 

          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->set_num_particles(sample_counts[1]); 
          beam_stack[j]->add_importance_weight(reducep); 
          
        } else if (sample_counts[1] > 0) {
          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->set_num_particles(sample_counts[1]); 
          beam_stack[j]->add_importance_weight(reducep); 

        } else if (sample_counts[0] > 0) {
          beam_stack[j]->leftArc();
          beam_stack[j]->add_particle_weight(reduceleftarcp); 
          beam_stack[j]->set_num_particles(sample_counts[0]); 
          beam_stack[j]->add_importance_weight(reducep); 
        }
      }
    }

    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, num_particles, eng);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //    ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), cmp_weighted_importance_ptr_weights_as); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cout << "Final beam size: " << beam_stack.size();

  if ((beam_stack.size() > 0) && take_max) {
    //resampleParticles(&beam_stack, num_particles, eng);
    std::sort(beam_stack.begin(), beam_stack.end(), cmp_particle_ptr_weights_as); 
    return ArcStandardParser(*beam_stack[0]);
  } else if (beam_stack.size() > 0) {
    //just take 1 sample
    resampleParticles(&beam_stack, 1, eng);
    for (unsigned i = 0; i < beam_stack.size(); ++i) {
      if (beam_stack[i]->num_particles() == 1) {
        //beam_stack[i]->print_arcs();
        //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
        //std::cout << "  Dir Accuracy: " << dir_acc;
        //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;
        return ArcStandardParser(*beam_stack[i]); 
      }
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcStandardParser(sent, tags);  
}

//generate a sentence: 4-way decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcEagerParser generateEagerSentence(Dict& dict, MT19937& eng, unsigned sent_limit, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  ArcEagerParser parser;
  bool terminate_shift = false;
  //bool need_shift = false;
  parser.push_tag(0);
  parser.shift(0);
    
  do {
    kAction a = kAction::sh; //placeholder action
    Words r_ctx = parser.reduce_context();
    if (parser.sentence_length() >= sent_limit) {
        // check to upper bound sentence length
        //if (!terminate_shift)
        //  std::cout << " LENGTH LIMITED ";
        terminate_shift = true;
        a = kAction::re;
    } else {
      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double leftarcreducep = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
      double rightarcshiftp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
      double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx); 

      if (!parser.left_arc_valid())
        leftarcreducep = 0;
      if (!parser.reduce_valid())
        reducep = 0;
      //if (leftarcreducep==0 && reducep==0)
      //  std::cout << "[ra:" << rightarcshiftp << " sh:" << shiftp << "] ";

      //sample an action
      std::vector<double> distr = {shiftp, leftarcreducep, rightarcshiftp, reducep};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
      //std::cout << "(" << parser.stack_depth() << ") ";
      //std::cout << act << " ";
      parser.add_particle_weight(distr[act]);
      
      if (act==0) {
        a = kAction::sh;
      } else if (act==1) {
        a = kAction::la; 
        parser.leftArc();
        //need_shift = true;
      } else if (act==2) {
        a = kAction::ra;
      } else {
        a = kAction::re;
        parser.reduce();
      }
    } 

    if ((a == kAction::sh) || (a == kAction::ra)) {
      //need_shift = false;
      //sample a tag - disallow root tag
      Words t_ctx = parser.tag_context(a);
      std::vector<double> t_distr(dict.tag_size() - 1, 0);
      for (WordId w = 1; w < dict.tag_size(); ++w) 
        t_distr[w-1] = tag_lm.prob(w, t_ctx); 
      multinomial_distribution<double> t_mult(t_distr);
      WordId tag = t_mult(eng) + 1;

      double tagp = tag_lm.prob(tag, t_ctx); 
      parser.push_tag(tag);
      parser.add_particle_weight(tagp);

      //sample a word 
      Words w_ctx = parser.shift_context();
      std::vector<double> w_distr(dict.size(), 0);

      w_distr[0] = shift_lm.prob(-1, w_ctx); //unk probability
      for (WordId w = 1; w < dict.size(); ++w) 
        w_distr[w] = shift_lm.prob(w, w_ctx); 
      multinomial_distribution<double> w_mult(w_distr);
      WordId word = w_mult(eng);
      if (word==0)
        word = -1;

      double wordp = shift_lm.prob(word, w_ctx); 
      parser.add_particle_weight(wordp);
      
      //perform action
      if (a == kAction::ra) {
        parser.rightArc(word);
      } else {
        parser.shift(word);
      }
    } //extra condition that left-arc needs a shift following
    //if (parser.is_terminal_configuration())
    //  std::cout << ":" << parser.stack_depth() << " ";

  //} while ((!parser.is_terminal_configuration() && !terminate_shift) || need_shift);
  } while (!parser.is_terminal_configuration() && !terminate_shift);
  //std::cout << std::endl;

  return parser;
}

//generate a sentence: ternary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcStandardParser generateSentence(Dict& dict, MT19937& eng, unsigned sent_limit, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  ArcStandardParser parser;
  bool terminate_shift = false;
  parser.push_tag(0);
  parser.shift(0);
    
  do {
    kAction a = kAction::sh; //placeholder action
    //std::cerr << "arcs: " << parser.arcs().size() << std::endl;
    Words r_ctx = parser.reduce_context();
    
    if (parser.stack_depth() < 2) {
      a = kAction::sh;
    } else if (parser.sentence_length() >= sent_limit) {
        // check to upper bound sentence length
        //if (!terminate_shift)
        //  std::cout << " LENGTH LIMITED ";
        terminate_shift = true;
        a = kAction::re;
    } else {
      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double leftarcreducep = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
      double rightarcreducep = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);

      if (parser.stack_depth() == 2)
        leftarcreducep = 0;

      //sample an action
      std::vector<double> distr = {shiftp, leftarcreducep, rightarcreducep};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
      //std::cout << "(" << parser.stack_depth() << ") ";
      //std::cout << act << " ";
      parser.add_particle_weight(distr[act]);
      
      if (act==0) {
        a = kAction::sh;
      } else if (act==1) {
        a = kAction::la; 
        parser.leftArc();
      } else {
        a = kAction::ra;
        parser.rightArc();
      }
    } 

    if (a == kAction::sh) {
      //sample a tag - disallow root tag
      Words t_ctx = parser.tag_context();
      std::vector<double> t_distr(dict.tag_size() - 1, 0);
      for (WordId w = 1; w < dict.tag_size(); ++w) 
        t_distr[w-1] = tag_lm.prob(w, t_ctx); 
      multinomial_distribution<double> t_mult(t_distr);
      WordId tag = t_mult(eng) + 1;

      double tagp = tag_lm.prob(tag, t_ctx); 
      parser.push_tag(tag);
      parser.add_particle_weight(tagp);

      //sample a word 
      Words w_ctx = parser.shift_context();
      std::vector<double> w_distr(dict.size(), 0);

      w_distr[0] = shift_lm.prob(-1, w_ctx); //unk probability
      for (WordId w = 1; w < dict.size(); ++w) 
        w_distr[w] = shift_lm.prob(w, w_ctx); 
      multinomial_distribution<double> w_mult(w_distr);
      WordId word = w_mult(eng);
      if (word==0)
        word = -1;

      double wordp = shift_lm.prob(word, w_ctx); 
      parser.shift(word);
      parser.add_particle_weight(wordp);
    }
  } while (!parser.is_terminal_configuration() && !terminate_shift);
  //std::cout << std::endl;

  //std::cout << std::endl;
  return parser;
}

//generate a sentence - binary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser generateSentence(Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  ArcStandardParser parser;
  bool terminate_shift = false;
  parser.push_tag(0);
  parser.shift(0);
    
  do {
    kAction a = kAction::sh; //placeholder action
    Words r_ctx = parser.reduce_context();
    Words a_ctx = parser.arc_context();
    
    if (parser.stack_depth() < 2) {
      a = kAction::sh;
    } else if (parser.sentence_length() > 100) {
        // check to upper bound sentence length
        if (!terminate_shift)
          std::cout << " LENGTH LIMITED ";
        terminate_shift = true;
        a = kAction::re;
    }  
    else {
      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);

      //sample an action
      std::vector<double> distr = {shiftp, reducep};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
      parser.add_particle_weight(distr[act]);
      
      if (act==0) {
        a = kAction::sh;
      } else {
        a = kAction::re; 
      }
    } 

    if (a == kAction::sh) {
      //sample a word 
      Words t_ctx = parser.tag_context();
      std::vector<double> t_distr(dict.tag_size() - 1, 0);
      for (WordId w = 1; w < dict.tag_size(); ++w) 
        t_distr[w-1] = tag_lm.prob(w, t_ctx); 
      multinomial_distribution<double> t_mult(t_distr);
      WordId tag = t_mult(eng) + 1;

      double tagp = tag_lm.prob(tag, t_ctx); 
      parser.push_tag(tag);
      parser.add_particle_weight(tagp);

      Words w_ctx = parser.shift_context();
      std::vector<double> w_distr(dict.size(), 0);

      w_distr[0] = shift_lm.prob(-1, w_ctx); //unk probability
      for (WordId w = 1; w < dict.size(); ++w) 
        w_distr[w] = shift_lm.prob(w, w_ctx); 
      multinomial_distribution<double> w_mult(w_distr);
      WordId word = w_mult(eng);
      if (word==0)
        word = -1;

      double wordp = shift_lm.prob(word, w_ctx); 
      parser.shift(word);
      parser.add_particle_weight(wordp);
    } else if (a == kAction::re) {
      double leftarcp = arc_lm.prob(static_cast<WordId>(kAction::la), a_ctx);
      double rightarcp = arc_lm.prob(static_cast<WordId>(kAction::ra), a_ctx);

      if (parser.stack_depth() == 2)
        leftarcp = 0;

      //sample arc direction
      std::vector<double> distr = {leftarcp, rightarcp};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
      parser.add_particle_weight(distr[act]);
      
      if (act==0) {
        parser.leftArc();
      } else {
        parser.rightArc();
      }
    }
  } while (!parser.is_terminal_configuration() && !terminate_shift);

  return parser;
}

}

#endif
