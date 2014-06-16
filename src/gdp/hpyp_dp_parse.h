#ifndef _HPYP_DP_PARSE_H_
#define _HPYP_DP_PARSE_H_

#include <cstdlib>

#include "transition_parser.h"
#include "hpyplm/hpyplm.h"
#include "pyp/random.h"
#include "pyp/crp.h"

namespace std {

template<typename T, typename... Args>
inline unique_ptr<T> make_unique(Args&&... args) {
  return unique_ptr<T>{new T{args...}};
} 

}

namespace oxlm {

typedef std::unique_ptr<ArcStandardParser> ParserPtr;
typedef std::vector<std::unique_ptr<ArcStandardParser>> ParserList;

void resampleParticles(ParserList* particles, unsigned num_particles, MT19937& eng) {
//TODO 
/*  std::vector<double> importance_w(num_particles, 0); //importance weights
  for (unsigned i = 0; i < num_particles; ++i)
    importance_w[i] = particles[i]->importance_weight();

  //resample according to importance weight
  multinomial_distribution_log part_mult(importance_w); 
  std::vector<unsigned> sample_indx;
  for (unsigned i = 0; i < num_particles; ++i) {
    unsigned pi = part_mult(eng);
    sample_indx.push_back(pi);
  }
  std::sort(sample_indx.begin(), sample_indx.end());

  std::vector<bool> occur_indx; //-1 if not in sample, else its position 
  unsigned s_ind = 0; //index in sample_indx
  for (unsigned i = 0; i < num_particles; ++i) { //over original particles
    //std::cout << s_ind << " ";
    if ((s_ind >= num_particles) || (i < sample_indx[s_ind]))
      occur_indx.push_back(false);
    else if (i == sample_indx[s_ind]) {
      occur_indx.push_back(true); 
      while ((s_ind < num_particles - 1) && (sample_indx[s_ind]==sample_indx[s_ind+1])) ++s_ind;
      ++s_ind;
    } else {
      cerr << "contradiction (" << i << ") " << std::endl;     
    }
  }

  std::vector<int> new_indx(num_particles, -1); //just for the record
  for (unsigned i = 0; i < num_particles; ++i) {
    new_indx[i] = i;
  }
  
  unsigned p_pos = 0; //over the particles that may be replaced
  for (unsigned i = 1; i < num_particles; ++i) { //over sorted indexes
    if (sample_indx[i]==sample_indx[i-1]) {
      while ((p_pos < num_particles) && (occur_indx[p_pos])) ++p_pos; //not supposed to go out of index 
      //actual (copy) reassignment
      if (p_pos < num_particles) {
        particles[p_pos] = particles[sample_indx[i]]; //key line: copy particle      
        new_indx[p_pos] = sample_indx[i];
        ++p_pos;
      } else
        std::cout << "invalid index ";
    } 
  }  */
}


//more sophisticated beam parser
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser beamParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned beam_size, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  //index in beam_chart is depth-of-stack - 1
  std::vector<ParserList> beam_chart; 
  beam_chart.push_back(ParserList());
  beam_chart[0].push_back(std::make_unique<ArcStandardParser>(sent, tags)); 
  //beam_chart[0][0]->print_sentence(dict);
  //beam_chart[0][0]->print_tags(dict);
  
  //std::cout << "gold arcs: ";
  for (auto d: gold_dep.arcs())
    std::cout << d << " ";
  std::cout << std::endl;   

  //shift ROOT symbol (probability 1)
  beam_chart[0][0]->shift(); 

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 1; k <= sent.size(); ++k) {
    //there are k beam lists. perform reduces down to list 1

    for (unsigned i = k - 1; i > 0; --i) { 
      //prune if size exceeds beam_size
      if (beam_chart[i].size() > beam_size) {
        std::sort(beam_chart[i].begin(), beam_chart[i].end(), cmp_particle_ptr_weights); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_chart[i].size(); j > beam_size; --j)
          beam_chart[i].pop_back();
      }

      //for every item in the list, add valid reduce actions to list i - 1 
      for (unsigned j = 0; (j < beam_chart[i].size()); ++j) {
        Words r_ctx = beam_chart[i][j]->reduce_context();
        Words a_ctx = beam_chart[i][j]->arc_context();
        double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(kAction::la), a_ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(kAction::ra), a_ctx);
       
        //TODO have option to make la/ra choice deterministic
        beam_chart[i-1].push_back(std::make_unique<ArcStandardParser>(*beam_chart[i][j]));
        if (i > 1) { //left arc only invalid when stack size is 2 **
          beam_chart[i-1].push_back(std::make_unique<ArcStandardParser>(*beam_chart[i][j]));

          beam_chart[i-1].back()->leftArc();
          beam_chart[i-1].back()->add_particle_weight(reducep*leftarcp);
          beam_chart[i-1].rbegin()[1]->rightArc();
          beam_chart[i-1].rbegin()[1]->add_particle_weight(reducep*rightarcp); 

          if (k == sent.size()) {  
            beam_chart[i-1].back()->add_importance_weight(reducep); 
            beam_chart[i-1].rbegin()[1]->add_importance_weight(reducep); 
          }
        } else {
          beam_chart[i-1].back()->rightArc();
          beam_chart[i-1].back()->add_particle_weight(reducep*rightarcp); 
          
          if (k == sent.size()) 
            beam_chart[i-1].back()->add_importance_weight(rightarcp); 
        }
      }
    }

    if ((beam_chart[0].size() > beam_size) || (k == sent.size())) {
        std::sort(beam_chart[0].begin(), beam_chart[0].end(), cmp_particle_ptr_weights); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_chart[0].size(); j > beam_size; --j)
          beam_chart[0].pop_back();
    }

    //perform shifts
    if (k < sent.size()) {
      for (unsigned i = 0; (i < k); ++i) { 
        for (unsigned j = 0; j < beam_chart[i].size(); ++j) {
          Words w_ctx = beam_chart[i][j]->shift_context();
          Words r_ctx = beam_chart[i][j]->reduce_context();
          Words t_ctx = beam_chart[i][j]->tag_context();
          
          WordId w = beam_chart[i][j]->next_word();
          WordId tag = beam_chart[i][j]->next_tag();
          double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
          double wordp = shift_lm.prob(w, w_ctx); 
          double tagp = tag_lm.prob(tag, t_ctx);

          beam_chart[i][j]->shift();
          beam_chart[i][j]->add_importance_weight(wordp); 
          beam_chart[i][j]->add_importance_weight(tagp); 
          beam_chart[i][j]->add_particle_weight(shiftp); 
          beam_chart[i][j]->add_particle_weight(wordp); 
          beam_chart[i][j]->add_particle_weight(tagp); 
        }
      }
      //insert new beam_chart[0] to increment indexes
      beam_chart.insert(beam_chart.begin(), ParserList());
    }   
  }
  
  //print parses
  unsigned n = 0; 
  for (unsigned i = 0; (i < 5) && (i < beam_chart[n].size()); ++i) {
    beam_chart[n][i]->print_arcs();

    float dir_acc = (beam_chart[n][i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
    std::cout << "  Dir Accuracy: " << dir_acc;
    std::cout << "  Sample weight: " << (beam_chart[n][i]->particle_weight()) << std::endl;
  }  

  if (beam_chart[n].size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardParser(sent, tags);  
  } else
    return ArcStandardParser(*beam_chart[n][0]); 
}  

template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser particleParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned num_particles, bool resample, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
    
  std::cout << "gold arcs: ";
  for (auto d: gold_dep.arcs())
    std::cout << d << " ";
  std::cout << std::endl; 

  ParserList beam_stack; 
  beam_stack.push_back(std::make_unique<ArcStandardParser>(sent, tags)); 
  beam_stack[0]->set_num_particles(num_particles);

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    unsigned sh_bound = beam_stack.size(); 
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]==nullptr)
        continue;
        
      //sample a sequence of possible actions leading up to the next shift
      Words r_ctx = beam_stack[j]->reduce_context();
      Words a_ctx = beam_stack[j]->arc_context();
      
      double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      double reduceleftarcp = reducep*arc_lm.prob(static_cast<WordId>(kAction::la), a_ctx);
      double reducerightarcp = reducep*arc_lm.prob(static_cast<WordId>(kAction::ra), a_ctx);

      int num_samples = beam_stack[j]->num_particles();
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

        /* if (k == sent.size()) {  
            beam_stack[i-1].back()->add_importance_weight(reducep); 
            beam_stack[i-1].rbegin()[1]->add_importance_weight(reducep); 
          } */
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
      if (sample_counts[0] == 0) {
        beam_stack[j] = nullptr;
      } else {
        Words t_ctx = beam_stack[j]->tag_context();
        Words w_ctx = beam_stack[j]->shift_context();
        double wordp = shift_lm.prob(beam_stack[j]->next_word(), w_ctx); 
        double tagp = tag_lm.prob(beam_stack[j]->next_tag(), t_ctx);

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }

      if (j==sh_bound) {
        sh_bound = beam_stack.size();
      }
    }
   
    if (resample)
      resampleParticles(&beam_stack, num_particles, eng);
  }
     
  ///completion
  ParserList final_beam; 
  bool has_more_states = true;

  while (has_more_states) {
    bool has_more_states = false;
    unsigned cur_beam_size = beam_stack.size();

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      //std::cerr << beam_stack[j].stack_depth() << " " << "(" << beam_stack[j].particle_weight() << ")";
      if (beam_stack[j]==nullptr) {
        continue;
      } else if (beam_stack[j]->is_terminal_configuration()) {
        //try to move to final beam
        final_beam.push_back(nullptr);
        final_beam.back() = beam_stack[j]; 
      } else {
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

    if (resample)
      resampleParticles(&beam_stack, num_particles, eng);
  }

  //resample final beam ?
  //if (resample)
  //  resampleParticles(&final_beam, num_particles, eng);

  //alternatively, sort according to particle weight 
  std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers

  //print parses
  for (unsigned i = 0; (i < 5) && (i < final_beam.size()); ++i) {
    final_beam[i]->print_arcs();

    float dir_acc = (final_beam[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
    std::cout << "  Dir Accuracy: " << dir_acc;
    std::cout << "  Sample weight: " << (final_beam[i]->particle_weight()) << std::endl;
  }  

  if (final_beam.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardParser(sent, tags);  
  } else
    return ArcStandardParser(*final_beam[0]); 
}



//generate a sentence
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser generateSentence(Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  ArcStandardParser parser;
  bool terminate_shift = false;
  parser.buffer_tag(0);
  parser.shift(0);
    
  do {
    kAction a = kAction::sh; //placeholder action
    Words t_ctx = parser.tag_context();
    Words r_ctx = parser.tag_context();
    Words a_ctx = parser.tag_context();
    
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
      std::vector<double> t_distr(dict.tag_size() - 1, 0);
      for (WordId w = 1; w < dict.tag_size(); ++w) 
        t_distr[w-1] = tag_lm.prob(w, t_ctx); 
      multinomial_distribution<double> t_mult(t_distr);
      WordId tag = t_mult(eng) + 1;

      double tagp = tag_lm.prob(tag, t_ctx); 
      parser.buffer_tag(tag);
      parser.add_particle_weight(tagp);

      Words w_ctx = parser.word_tag_next_context();
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

      //sample arc direction
      std::vector<double> distr = {leftarcp, rightarcp};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
      parser.add_particle_weight(distr[act]);
      
      //may need to enforce the la constraint here
      if (act==0) {
        parser.leftArc();
      } else {
        parser.rightArc();
      }
    }
  } while (!parser.is_terminal_configuration());

  return parser;
}

/*
//gold parser
//TODO update
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser goldParseSentence(Words sent, Words tag, ArcList gold_dep, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  //compute and score oracle parse sequence
  ArcStandardParser gold_p(s);

  while (!gold_p.is_terminal_configuration()) {
    Words ctx = gold_p.word_context();
    double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), ctx);
    double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), ctx);
    
    kAction nexta = gold_p.oracleNext(goldd);
    if (nexta==kAction::sh) {
      WordId w = gold_p.next_word();
      double wordp = shift_lm.prob(w, ctx); 
      gold_p.shift();
      gold_p.add_importance_weight(wordp); 
      gold_p.add_particle_weight(shiftp*wordp); 
    } else {
      double leftarcp = arc_lm.prob(static_cast<WordId>(kAction::la), ctx);
      double rightarcp = arc_lm.prob(static_cast<WordId>(kAction::ra), ctx);

      if (gold_p.is_buffer_empty())
        gold_p.add_importance_weight(reducep);
      if (nexta==kAction::la) {
        gold_p.leftArc();
        gold_p.add_particle_weight(reducep*leftarcp); 
      } else if (nexta==kAction::ra) {
        gold_p.rightArc();
        gold_p.add_particle_weight(reducep*rightarcp); 
      } else {
        cerr << "Invalid gold parse." << std::endl;
      }
    }
  }

  std::cout << "Gold weight: " << gold_p.particle_weight() << std::endl;
  return gold_p;
}   

//old resampler
void resampleParticles(ParserList* particles, unsigned num_particles, MT19937& eng) {
 std::vector<double> importance_w(num_particles, 0); //importance weights
  for (unsigned i = 0; i < num_particles; ++i)
    importance_w[i] = particles[i]->importance_weight();

  //resample according to importance weight
  multinomial_distribution_log part_mult(importance_w); 
  std::vector<unsigned> sample_indx;
  for (unsigned i = 0; i < num_particles; ++i) {
    unsigned pi = part_mult(eng);
    sample_indx.push_back(pi);
  }
  std::sort(sample_indx.begin(), sample_indx.end());

  std::vector<bool> occur_indx; //-1 if not in sample, else its position 
  unsigned s_ind = 0; //index in sample_indx
  for (unsigned i = 0; i < num_particles; ++i) { //over original particles
    //std::cout << s_ind << " ";
    if ((s_ind >= num_particles) || (i < sample_indx[s_ind]))
      occur_indx.push_back(false);
    else if (i == sample_indx[s_ind]) {
      occur_indx.push_back(true); 
      while ((s_ind < num_particles - 1) && (sample_indx[s_ind]==sample_indx[s_ind+1])) ++s_ind;
      ++s_ind;
    } else {
      cerr << "contradiction (" << i << ") " << std::endl;     
    }
  }

  std::vector<int> new_indx(num_particles, -1); //just for the record
  for (unsigned i = 0; i < num_particles; ++i) {
    new_indx[i] = i;
  }
  
  unsigned p_pos = 0; //over the particles that may be replaced
  for (unsigned i = 1; i < num_particles; ++i) { //over sorted indexes
    if (sample_indx[i]==sample_indx[i-1]) {
      while ((p_pos < num_particles) && (occur_indx[p_pos])) ++p_pos; //not supposed to go out of index 
      //actual (copy) reassignment
      if (p_pos < num_particles) {
        particles[p_pos] = particles[sample_indx[i]]; //key line: copy particle      
        new_indx[p_pos] = sample_indx[i];
        ++p_pos;
      } else
        std::cout << "invalid index ";
    } 
  }  
}

//old beam parser (per word shifted)
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser beamShiftParseSentence(Words sent, Words tag, ArcList gold_dep, unsigned beam_size, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  std::vector<ArcStandardParser> beam(1, ArcStandardParser(s, ps));  
  beam[0].print_sentence(dict);
  beam[0].print_postags(dict);
  
  std::cout << "gold arcs: ";
  for (auto d: goldd)
    std::cout << d << " ";
  std::cout << std::endl; 
 
  for (unsigned i = 0; i < s.size(); ++i) {
    unsigned sh_bound = beam.size(); 
    for (unsigned j = 0; j < beam.size(); ++j) { 
      //sample a sequence of possible actions leading up to the next shift
      Words w_ctx = beam[j].word_tag_next_context();
      Words p_ctx = beam[j].word_tag_context();
      Words r_ctx = beam[j].word_tag_context();
      Words t_ctx = beam[j].pos_context();
      //if (j%10 == 0)
      //  cerr << j << " " << beam.size() << std::endl;

      if (beam[j].stack_depth() >= 2) {
        //add paths for reduce actions
        double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(kAction::la), p_ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(kAction::ra), p_ctx);
       
        //add to the beam 
        beam.push_back(beam[j]);
        if (beam[j].left_arc_valid()) {
          beam.push_back(beam[j]);
          ArcStandardParser& right_p = beam.rbegin()[1]; 
          ArcStandardParser& left_p = beam.back(); 

          left_p.leftArc();
          left_p.add_particle_weight(reducep*leftarcp);
          left_p.set_importance_weight(reducep); 
          right_p.rightArc();
          right_p.add_particle_weight(reducep*rightarcp); 
          right_p.set_importance_weight(reducep); 

        } else {
          ArcStandardParser& right_p = beam.back(); 

          right_p.right_arc();
          right_p.add_particle_weight(reducep*rightarcp); 
          right_p.set_importance_weight(reducep*rightarcp); 
        }
      }

      //perform shift
      ArcStandardParser& shift_p = beam[j];
      double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
      WordId w = shift_p.next_word();
      double wordp = shift_lm.prob(w, w_ctx); 
      WordId tag = shift_p.next_tag();
      double tagp = tag_lm.prob(tag, t_ctx);

      shift_p.shift();
      shift_p.set_importance_weight(wordp); 
      shift_p.add_importance_weight(tagp); 
      shift_p.add_particle_weight(shiftp*wordp); 
      shift_p.add_particle_weight(tagp); 
      //cerr << "p: " <<  parser.stack_depth() << " " << parser.particle_weight() << " " << parser.buffer_length() << " " << parser.sentence_length() << std::endl;

      if (j==sh_bound) {
        if (beam.size() > (beam_size + sh_bound)) {
          //cap unprocessed items for each local number of operations at beam size
          std::sort(beam.begin() + (beam_size + sh_bound), beam.end(), cmp_particle_weights);
          for (unsigned j = beam.size()- 1; j >= (beam_size + sh_bound); --j)
            beam.pop_back();
        }

        sh_bound = beam.size();
      }
    }
   
    //otherwise compare normalized weights 
    std::sort(beam.begin(), beam.end(), cmp_particle_weights);
    //remove items with worst scores
    for (unsigned j = beam.size()- 1; j >= beam_size; --j)
      beam.pop_back();
  }

  //cerr << "start completion" << std::endl;
  std::vector<ArcStandardParser> final_beam;  

  while (beam.size() > 0) {
    unsigned cur_beam_size = beam.size();

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      //cerr << beam[j].stack_depth() << " " << "(" << beam[j].particle_weight() << ")";
      if (beam[j].is_terminal_configuration()) {
        //has a complete parse
        final_beam.push_back(beam[j]);
        beam[j].set_log_particle_weight(1000); 
      } else {
        //add paths for reduce actions
        //Words w_ctx = beam[j].word_tag_next_context();
        Words p_ctx = beam[j].word_tag_context();
        Words r_ctx = beam[j].word_tag_context();
        double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(kAction::la), p_ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(kAction::ra), p_ctx);

        if (beam[j].left_arc_valid()) {
          beam.push_back(beam[j]);
          ArcStandardParser& right_p = beam[j];
          ArcStandardParser& left_p = beam.back(); 

          left_p.leftArc();
          left_p.add_particle_weight(reducep*leftarcp);
          left_p.set_importance_weight(reducep); 
          right_p.rightArc();
          right_p.add_particle_weight(reducep*rightarcp); 
          right_p.set_importance_weight(reducep); 
        } else {
          ArcStandardParser& right_p = beam[j];

          right_p.rightArc();
          right_p.add_particle_weight(reducep*rightarcp); 
          right_p.set_importance_weight(reducep*rightarcp); 
        }
      }  
    }
    //cerr << std::endl;

    //can also compare normalized weights 
    std::sort(beam.begin(), beam.end(), cmp_particle_weights);
    //remove items with worst scores
    for (int j = beam.size()- 1; ((j >= static_cast<int>(beam_size)) || ((j >= 0) && (beam[j].particle_weight() >= 1000))); --j) {
      beam.pop_back();
    }

    if ((final_beam.size() > 2*beam_size) || (beam.size()==0)) {
      std::sort(final_beam.begin(), final_beam.end(), cmp_particle_weights);
      //remove items with worst scores
      for (int j = final_beam.size()- 1; (j >= static_cast<int>(beam_size)); --j)
        final_beam.pop_back();
    }
  }

  
  //print parses
  for (unsigned i = 0; (i < 5) && (i < final_beam.size()); ++i) {
    auto& parser = final_beam[i];
      parser.print_arcs();
    //std::cout << parser.actions_str() << std::endl;

    float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/(s.size()-1);
    float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/(s.size()-1);

    std::cout << "  Dir Accuracy: " << dir_acc;
    std::cout << "  UnDir Accuracy: " << undir_acc;
    std::cout << "  Sample weight: " << (parser.particle_weight()) << std::endl; // /log(10) 
  }  

  if (final_beam.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardParser(s);  
  } else
    return final_beam[0];
}

//bad beam parser (per action)
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser beam_action_parse_sentence(Words sent, Words tag, ArcList gold_dep, unsigned beam_size, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  std::vector<ArcStandardParser> beam(1, ArcStandardParser(s, ps));  
  beam[0].print_sentence(dict);
  beam[0].print_tags(dict);
  
  std::cout << "gold arcs: ";
  for (auto d: goldd.arcs())
    std::cout << d << " ";
  std::cout << std::endl;  
 
  //all beam items will terminate after same number of steps
  for (unsigned i = 0; i < 2*(s.size()-1); ++i) {
    unsigned b_size = beam.size(); 
    for (unsigned j = 0; j < b_size; ++j) { 
    //std::cout << i << " " << beam[j].stack_depth() << " " << beam[j].buffer_length() << std::endl;
      //sample a sequence of possible actions leading up to the next shift
      Words p_ctx = beam[j].word_tag_context();
      Words r_ctx = beam[j].word_tag_context();
      Words t_ctx = beam[j].tag_context();

      if (beam[j].stack_depth() < 2) {
        //have to shift
        ArcStandardParser& shift_p = beam[j];
        Words w_ctx = beam[j].word_tag_next_context();
        WordId w = shift_p.next_word();
        double wordp = shift_lm.prob(w, w_ctx); 
        WordId tag = shift_p.next_tag();
        double tagp = tag_lm.prob(tag, t_ctx);

        shift_p.shift();
        shift_p.add_importance_weight(wordp); 
        shift_p.add_importance_weight(tagp); 
        //shift_p.add_particle_weight(wordp); 
        //shift_p.add_particle_weight(tagp); 
      } else {
        //add paths for reduce actions
        double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(kAction::la), p_ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(kAction::ra), p_ctx);
       
        //add to the beam: replace current position with right arc
        if (beam[j].is_buffer_empty()) {
          if (beam[j].left_arc_valid()) {
            beam.push_back(beam[j]);
            ArcStandardParser& right_p = beam[j]; 
            ArcStandardParser& left_p = beam.back(); 

            left_p.leftArc();
            left_p.add_importance_weight(reducep); 
            //left_p.add_particle_weight(reducep);
            left_p.add_particle_weight(leftarcp);
            right_p.rightArc();
            right_p.add_importance_weight(reducep); 
            //right_p.add_particle_weight(reducep); 
            right_p.add_particle_weight(rightarcp); 
          } else {
            ArcStandardParser& right_p = beam[j]; 
            right_p.rightArc();
            right_p.add_importance_weight(reducep*rightarcp); 
            //right_p.add_particle_weight(reducep*rightarcp); 
          }
        } else {
          beam.push_back(beam[j]);
          if (beam[j].left_arc_valid()) {
            beam.push_back(beam[j]);
            ArcStandardParser& right_p = beam.rbegin()[1]; 
            ArcStandardParser& left_p = beam.back(); 

            left_p.leftArc();
            left_p.set_importance_weight(reducep); 
            //left_p.add_particle_weight(reducep);
            left_p.add_particle_weight(leftarcp);
            right_p.rightArc();
            right_p.set_importance_weight(reducep); 
            //right_p.add_particle_weight(reducep); 
            right_p.add_particle_weight(rightarcp); 
          } else {
            ArcStandardParser& right_p = beam.back(); 
            right_p.rightArc();
            right_p.set_importance_weight(reducep*rightarcp); 
            //right_p.add_particle_weight(reducep*rightarcp); 
          }
          //std::cout << " + " << beam.size();

          ArcStandardParser& shift_p = beam[j];
          double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
          WordId w = shift_p.next_word();
          Words w_ctx = beam[j].word_tag_next_context();
          double wordp = shift_lm.prob(w, w_ctx); 
          WordId tag = shift_p.next_tag();
          double tagp = tag_lm.prob(tag, t_ctx);

          shift_p.shift();
          shift_p.set_importance_weight(wordp); 
          shift_p.add_importance_weight(tagp); 
          //shift_p.add_particle_weight(wordp); 
          //shift_p.add_particle_weight(tagp); 
          shift_p.add_particle_weight(shiftp); 
        }
      }
    }
    //std::cout << std::endl;

    //otherwise compare normalized weights 
    std::sort(beam.begin(), beam.end(), cmp_particle_weights);
    //remove items with worst scores
    for (unsigned j = beam.size()- 1; j >= beam_size; --j)
      beam.pop_back();
  } 
  
  //now sort according to total probability
  std::sort(beam.begin(), beam.end(), cmp_particle_importance_weights);

  //print parses
  for (unsigned i = 0; (i < 5) && (i < beam.size()); ++i) {
    auto& parser = beam[i];
    parser.print_arcs();
    //std::cout << parser.arcs_str();
    //std::cout << parser.actions_str() << std::endl;

    float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/(s.size()-1);
    float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/(s.size()-1);

    std::cout << "  Dir Accuracy: " << dir_acc;
    //std::cout << "  UnDir Accuracy: " << undir_acc;
    std::cout << "  Sample weight: " << (parser.particle_weight()) << std::endl; // /log(10) 
  }  

  if (beam.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardParser(s);  
  } else
    return beam[0];
}

//for greedy binary action decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser greedyParseSentence(Words sent, Words tag, ArcList gold_dep, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  //read in dev sentence; sample actions: for each action, execute and compute next word probability for shift
  std::cout << "gold arcs: ";
  for (auto d: goldd.arcs())
    std::cout << d << " ";
  std::cout << std::endl; 

  ArcStandardParser parser(s); 

  do {
    kAction a = kAction::re; //placeholder action
    Words ctx = parser.word_context();
    double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), ctx);
    double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), ctx);
    std::cout << "(sh " << shiftp << ", re " << reducep << ") ";

    if ((parser.stack_depth()< 2) && !parser.is_buffer_empty()) 
      a = kAction::sh;
    else if (parser.is_buffer_empty()) {
      a = kAction::re;
      parser.add_importance_weight(reducep);
    } else {
      //shift or reduce
      if (shiftp >= reducep) {
        a = kAction::sh;
        parser.add_particle_weight(shiftp);
      } else {
        a = kAction::re; 
        parser.add_particle_weight(reducep);
      }
    }
    
    if (a == kAction::sh) {
      WordId w = parser.next_word();
      double wordp = shift_lm.prob(w, ctx); 
      parser.shift();
      parser.add_particle_weight(wordp);

    } else if (a == kAction::re) {
      double leftarcp = arc_lm.prob(static_cast<WordId>(kAction::la), ctx);
      double rightarcp = arc_lm.prob(static_cast<WordId>(kAction::ra), ctx);
      std::cout << "(la " << leftarcp << ", ra " << rightarcp << ") ";
    
      if ((leftarcp > rightarcp) && parser.leftArc()) {
        parser.add_particle_weight(leftarcp);
      } else {
        if (leftarcp > rightarcp) 
          parser.add_importance_weight(rightarcp);
        parser.rightArc();
        parser.add_particle_weight(rightarcp);
      }
    }
  } while (!parser.is_terminal_configuration());

  std::cout << std::endl;
  parser.print_arcs();
  std::cout << parser.actions_str() << std::endl;

  float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/s.size();
  float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/s.size();

  std::cout << "  Dir Accuracy: " << dir_acc;
  std::cout << "  UnDir Accuracy: " << undir_acc;
  std::cout << "  Importance weight: " << parser.importance_weight();
  std::cout << "  Sample weight: " << parser.particle_weight() << std::endl;

  return parser;
}
   */
}

#endif
