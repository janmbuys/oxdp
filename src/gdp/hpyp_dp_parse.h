#ifndef _GDP_HPYP_DP_PARSE_H_
#define _GDP_HPYP_DP_PARSE_H_

#include <cstdlib>
#include <memory>

#include "transition_parser.h"
#include "eisner_parser.h"
#include "hpyplm/hpyplm.h"
#include "utils/random.h"
#include "pyp/crp.h"

namespace std {

namespace oxlm {

//TODO move this initialization to a weights class

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

//TODO rather add more options to the particle gold parser later, or implement a full dynamic oracle

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





}

#endif
