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

//Compute score for Eisner parses
template<unsigned kShiftOrder, unsigned kTagOrder>
void eisnerScoreSentence(EisnerParser* parser, bool with_words, Dict& dict, PYPLM<kShiftOrder>& shift_lm, PYPLM<kTagOrder>& tag_lm) {
  parser->reset_weight();

  //std::cout << " prev children: ";
  for (WordIndex i = 1; (i < static_cast<int>(parser->sentence_length())); ++i) {
    //dependent i, head j
    WordIndex j = parser->head_at(i);
    if (j == -1)
      continue;
    WordIndex prev_child = 0;
    if (i < j)
      prev_child = parser->prev_left_child(i, j);
    else if (j < i)
      prev_child = parser->prev_right_child(i, j);
    //std::cout << prev_child << " ";

    double weight = -std::log(tag_lm.prob(parser->tag_at(i), parser->tag_context(i, j, prev_child)));
    if (with_words) 
      weight -= std::log(shift_lm.prob(parser->sentence_at(i), parser->shift_context(i, j, prev_child)));
    parser->add_weight(weight);
    //std::cout << "(" << dict.lookupTag(i) << ", " << dict.lookupTag(j) << ", " << weight << ") ";
  }  

  for (WordIndex j = 0; (j < static_cast<int>(parser->sentence_length())); ++j) {
    WordIndex prev_child = parser->leftmost_child(j);
    double weight;
    //root only generates right children
    if (j > 0) {
      weight = -std::log(tag_lm.prob(1, parser->tag_context(-1, j, prev_child)));
      if (with_words) 
        weight -= std::log(shift_lm.prob(1, parser->shift_context(-1, j, prev_child)));
      parser->add_weight(weight);
    }

    prev_child = parser->rightmost_child(j);
    weight = -std::log(tag_lm.prob(1, parser->tag_context(parser->sentence_length(), j, prev_child)));
    if (with_words) 
      weight -= std::log(shift_lm.prob(1, parser->shift_context(parser->sentence_length(), j, prev_child)));
    parser->add_weight(weight);
  }
  //std::cout << std::endl;
  //std::cout << parser->weight() << std::endl;
}

//Eisner parsing algorithm
template<unsigned kShiftOrder, unsigned kTagOrder>
EisnerParser eisnerParseSentence(Words sent, Words tags, ArcList gold_dep, bool with_words, Dict& dict, PYPLM<kShiftOrder>& shift_lm, PYPLM<kTagOrder>& tag_lm) {
  EisnerParser parser(sent, tags); 
  //std::cout << sent.size() << ": ";

  for (WordIndex k = 1; k < sent.size(); ++k) {
    //std::cout << k << " ";
    for (WordIndex s = 0; ((s + k) < sent.size()); ++s) {
      WordIndex t = s + k;

      //assign weights and splits to the four things in chart item (s, t)

      double left_incomp_w = L_MAX;
      WordIndex split = -1;
      for (WordIndex r = s; r < t; ++r) {
        WordIndex prev_child = parser.left_complete_split(r+1, t); // + 1;
        double left_weight = -std::log(tag_lm.prob(parser.tag_at(s), parser.tag_context(s, t, prev_child)));
        if (with_words) 
          left_weight -= std::log(shift_lm.prob(parser.sentence_at(s), parser.shift_context(s, t, prev_child)));
        //s has no more right children
        prev_child = parser.right_complete_split(s, r);
        double stop_weight = -std::log(tag_lm.prob(1, parser.tag_context(sent.size(), s, prev_child)));
        if (with_words) 
          stop_weight -= std::log(shift_lm.prob(1, parser.shift_context(sent.size(), s, prev_child)));
        
        double w = parser.right_complete_weight(s, r)
                   + parser.left_complete_weight(r+1, t)
                   + left_weight
                   + stop_weight;
        if (w < left_incomp_w) {
          left_incomp_w = w;
          split = r;
        }
      }
      parser.set_left_incomplete_weight(s, t, left_incomp_w);
      parser.set_left_incomplete_split(s, t, split);
      
      double right_incomp_w = L_MAX;
      split = -1;
      for (WordIndex r = s; r < t; ++r) {
        WordIndex prev_child = parser.right_complete_split(s, r);
        double right_weight = -std::log(tag_lm.prob(parser.tag_at(t), parser.tag_context(t, s, prev_child)));
        if (with_words) 
          right_weight -= std::log(shift_lm.prob(parser.sentence_at(t), parser.shift_context(t, s, prev_child)));
        
        //t has no more left children
        prev_child = parser.left_complete_split(r+1, t);
        double stop_weight = -std::log(tag_lm.prob(1, parser.tag_context(-1, t, prev_child)));
        if (with_words) 
          stop_weight -= std::log(shift_lm.prob(1, parser.shift_context(-1, t, prev_child)));

        double w = parser.right_complete_weight(s, r)
                   + parser.left_complete_weight(r+1, t)
                   + right_weight
                   + stop_weight;
        if (w < right_incomp_w) {
          right_incomp_w = w;
          split = r;
        }
      }
      
      parser.set_right_incomplete_weight(s, t, right_incomp_w);
      parser.set_right_incomplete_split(s, t, split);

      double left_comp_w = L_MAX;
      split = -1;
      for (WordIndex r = s; r < t; ++r) {
        //r has no more left children
        WordIndex prev_child = parser.left_complete_split(s, r);
        double stop_weight = -std::log(tag_lm.prob(1, parser.tag_context(-1, r, prev_child)));
        if (with_words)
          stop_weight -= std::log(shift_lm.prob(1, parser.shift_context(-1, r, prev_child)));      
        double w = parser.left_complete_weight(s, r)
                   + parser.left_incomplete_weight(r, t)
                   + stop_weight;
        if (w < left_comp_w) {
          left_comp_w = w;
          split = r;
        }
      }
      parser.set_left_complete_weight(s, t, left_comp_w);
      parser.set_left_complete_split(s, t, split);

      double right_comp_w = L_MAX;
      split = -1;
      for (WordIndex r = s + 1; r <= t; ++r) {
        //r has no more right children
        WordIndex prev_child = parser.right_complete_split(r, t);
        double stop_weight = -std::log(tag_lm.prob(1, parser.tag_context(sent.size(), r, prev_child)));
        if (with_words)
          stop_weight -= std::log(shift_lm.prob(1, parser.shift_context(sent.size(), r, prev_child)));

        //for sentence completion, root has no more right children
        if (s==0 && t==(sent.size()-1)) {
          stop_weight -= std::log(tag_lm.prob(1, parser.tag_context(sent.size(), s, r)));
          if (with_words)
            stop_weight -= std::log(shift_lm.prob(1, parser.shift_context(sent.size(), s, r)));
        }

        double w = parser.right_incomplete_weight(s, r)
             + parser.right_complete_weight(r, t)
             + stop_weight;
        if (w < right_comp_w) {
          right_comp_w = w;
          split = r;
        }
      }
      parser.set_right_complete_weight(s, t, right_comp_w);
      parser.set_right_complete_split(s, t, split);
    }
  } 
 
  double best_weight = parser.right_complete_weight(0, sent.size()-1);
  std::cout << best_weight << " ";

  //recover and assign best parse
  //must have right arc from node 0 
  //indexes are inclusive
  parser.recover_parse_tree(0, sent.size() - 1, true, false);
  parser.set_arc_children();
  eisnerScoreSentence(&parser, with_words, dict, shift_lm, tag_lm);
  //parser.print_chart();

  std::cout << parser.weight() << " ";
  parser.print_arcs();
  return parser;
}

/* inline void recover_parse_tree(EisnerParser* parser, WordIndex s, WordIndex t, bool complete, bool left_arc) {
  if (!complete) {
    WordIndex r;

    if (left_arc) {
      //left arc
      parser->set_arc(s, t);
      r = parser->left_incomplete_split(s, t);
    } else {
      //right arc
      parser->set_arc(t, s);
      r = parser->right_incomplete_split(s, t);
    }

    recover_parse_tree(parser, s, r, true, false);
    recover_parse_tree(parser, r+1, t, true, true);
  } else {

    if (left_arc) {
      WordIndex r = left_complete_split(s, t);
      recover_parse_tree(parser, s, r, true, true);
      recover_parse_tree(parser, r, t, false, true);
    } else {
      WordIndex r = right_complete_split(s, t);
      recover_parse_tree(parser, s, r, false, false);
      recover_parse_tree(parser, r, t, true, false);
    }
  }
} */


//beam parser for arc eager
//try to follow same logic as much as possible as for arc standard
//4-way decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcEagerParser beamParseSentenceEager(Words sent, Words tags, ArcList gold_dep, unsigned beam_size, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  //index in beam_chart is depth-of-stack - 1
  std::vector<AeParserList> beam_chart; 
  beam_chart.push_back(AeParserList());
  beam_chart[0].push_back(std::make_unique<ArcEagerParser>(sent, tags)); 
  beam_chart[0][0]->print_sentence(dict);
  //beam_chart[0][0]->print_tags(dict);
  
  //std::cout << "gold arcs: ";
  gold_dep.print_arcs();

  //shift ROOT symbol (probability 1)
  beam_chart[0][0]->shift(); 

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 1; k < sent.size(); ++k) {
    //std::cout << "sent position " << k << std::endl;
    //there are k beam lists. perform reduces down to list 1
    for (unsigned i = k - 1; i > 0; --i) { 
      //prune if size exceeds beam_size
      if (beam_chart[i].size() > beam_size) {
        std::sort(beam_chart[i].begin(), beam_chart[i].end(), cmp_particle_ptr_weights_ae); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_chart[i].size(); j > beam_size; --j)
          beam_chart[i].pop_back();
      }

      //std::cout << "reduce list size: " << beam_chart[i].size() << std::endl;
      //consider reduce and left arc actions
      //for every item in the list, add valid reduce actions to list i - 1 
      for (unsigned j = 0; (j < beam_chart[i].size()); ++j) {
        Words r_ctx = beam_chart[i][j]->reduce_context();
        double leftarcreducep = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
        double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
        //std::cout << "(la: " << leftarcreducep << ", re: " << reducep << ") ";
        //double reducetotalp = leftarcreducep + reducep;
       
        //actually also need importance weight if either is invalid
        //left arc invalid also after last shift
        if (beam_chart[i][j]->left_arc_valid()) { 
          beam_chart[i-1].push_back(std::make_unique<ArcEagerParser>(*beam_chart[i][j]));
          beam_chart[i-1].back()->leftArc();
          beam_chart[i-1].back()->add_particle_weight(leftarcreducep);
        } 
        
        if (beam_chart[i][j]->reduce_valid()) {          
          beam_chart[i-1].push_back(std::make_unique<ArcEagerParser>(*beam_chart[i][j]));
          beam_chart[i-1].back()->reduce();
          beam_chart[i-1].back()->add_particle_weight(reducep); 
          
        }
      }
      //std::cout << std::endl;
    }

    if (beam_chart[0].size() > beam_size) {
        std::sort(beam_chart[0].begin(), beam_chart[0].end(), cmp_particle_ptr_weights_ae); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_chart[0].size(); j > beam_size; --j)
          beam_chart[0].pop_back();
    }

    //perform shifts: shift or right arc
    for (unsigned i = 0; (i < k); ++i) { 
      unsigned list_size = beam_chart[i].size();
      //std::cout << "shift list size: " << list_size << std::endl;
      for (unsigned j = 0; j < list_size; ++j) {
        Words w_ctx = beam_chart[i][j]->shift_context();
        Words r_ctx = beam_chart[i][j]->reduce_context();
          
        double wordp = 1;
        double tagp = 1;
        double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
        double rightarcshiftp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
        //double shifttotalp = shiftp + rightarcshiftp;
        //std::cout << "(sh: " << shiftp << ", ra: " << rightarcshiftp << ") ";

        //TODO: ra not valid for stop symbol
        if (k < (sent.size() - 1)) {
        //assume ra is valid
          if (with_words)
            wordp = shift_lm.prob(beam_chart[i][j]->next_word(), w_ctx); 
          Words t_ctx = beam_chart[i][j]->tag_context(kAction::ra);
          tagp = tag_lm.prob(beam_chart[i][j]->next_tag(), t_ctx);
          
          beam_chart[i].push_back(std::make_unique<ArcEagerParser>(*beam_chart[i][j]));
          beam_chart[i].back()->rightArc();
          beam_chart[i].back()->add_particle_weight(rightarcshiftp);
          
          beam_chart[i].back()->add_importance_weight(wordp); 
          beam_chart[i].back()->add_importance_weight(tagp); 
          beam_chart[i].back()->add_particle_weight(wordp); 
          beam_chart[i].back()->add_particle_weight(tagp); 
        }

        //shift is valid
        if (with_words)
          wordp = shift_lm.prob(beam_chart[i][j]->next_word(), w_ctx); 
        Words t_ctx = beam_chart[i][j]->tag_context(kAction::sh);
        tagp = tag_lm.prob(beam_chart[i][j]->next_tag(), t_ctx);
          
        beam_chart[i][j]->shift();
        beam_chart[i][j]->add_particle_weight(shiftp); 
          
        beam_chart[i][j]->add_importance_weight(wordp); 
        beam_chart[i][j]->add_importance_weight(tagp); 
        beam_chart[i][j]->add_particle_weight(wordp); 
        beam_chart[i][j]->add_particle_weight(tagp); 
      }
    }
    //insert new beam_chart[0] to increment indexes
    beam_chart.insert(beam_chart.begin(), AeParserList());
       
    //std::cout << std::endl;
  }
 
  //completion: reduce after last shift
  //std::cout << "completion" << std::endl;
  for (unsigned i = beam_chart.size() - 1; i > 0; --i) {  //sent.size()
    //prune if size exceeds beam_size
    if (beam_chart[i].size() > beam_size) {
      std::sort(beam_chart[i].begin(), beam_chart[i].end(), cmp_reduce_particle_ptr_weights_ae); //handle pointers
      //remove items with worst scores, and those that cannot reduce
      for (unsigned j = beam_chart[i].size() - 1; ((j >= beam_size) || ((j > 0) && !beam_chart[i][j]->reduce_valid())); --j)
        beam_chart[i].pop_back();
    }

    //std::cout << i <<  " reduce list size: " << beam_chart[i].size() << std::endl;
    //consider reduce and left arc actions
    //for every item in the list, add valid reduce actions to list i - 1 
    for (unsigned j = 0; (j < beam_chart[i].size()); ++j) {
      Words r_ctx = beam_chart[i][j]->reduce_context();
      double reducep = reduce_lm.prob(static_cast<WordId>(kAction::re), r_ctx);
                
      if (beam_chart[i][j]->reduce_valid()) {  //modify to handle stop symbol differently       
        beam_chart[i-1].push_back(std::make_unique<ArcEagerParser>(*beam_chart[i][j]));
        beam_chart[i-1].back()->reduce();
        //TODO: don't add probabilities, reduce only use to check for completeness
        //check also if we do add probabilities
        beam_chart[i-1].back()->add_particle_weight(reducep); 
        beam_chart[i-1].back()->add_importance_weight(reducep); 
        //std::cout << j << " re valid ";
      }
    }
    //std::cout << std::endl;
  }

  std::sort(beam_chart[0].begin(), beam_chart[0].end(), cmp_particle_ptr_weights_ae); //handle pointers

  //print parses
  unsigned const n = 0; 
  //std::cout << "Beam size: " << beam_chart[n].size() << std::endl;
  for (unsigned i = 0; (i < beam_chart[n].size()); ++i) 
    beam_chart[n][0]->add_beam_particle_weight(beam_chart[n][i]->particle_weight());

  for (unsigned i = 0; (i < 5) && (i < beam_chart[n].size()); ++i) {
    //beam_chart[n][i]->print_arcs();
    //std::cout << beam_chart[n][i]->actions_str() << "\n";
    //std::cout << "\n";

    //float dir_acc = (beam_chart[n][i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
    //std::cout << "  Dir Accuracy: " << dir_acc;
    //std::cout << "  Sample weight: " << (beam_chart[n][i]->particle_weight()) << std::endl;
  }  

  if (beam_chart[n].size()==0) {
    //std::cout << "no parse found" << std::endl;
    return ArcEagerParser(sent, tags);  
  } else {
    beam_chart[n][0]->print_arcs();
    std::cout << beam_chart[n][0]->actions_str() << "\n";
    return ArcEagerParser(*beam_chart[n][0]); 
  }
}  

//more sophisticated beam parser
//ternary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcStandardParser beamParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned beam_size, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  //index in beam_chart is depth-of-stack - 1
  std::vector<AsParserList> beam_chart; 
  beam_chart.push_back(AsParserList());
  beam_chart[0].push_back(std::make_unique<ArcStandardParser>(sent, tags)); 
  //beam_chart[0][0]->print_sentence(dict);
  //beam_chart[0][0]->print_tags(dict);
  
  //gold_dep.print_arcs();
  //std::cout << "  Gold\n";

  //shift ROOT symbol (probability 1)
  beam_chart[0][0]->shift(); 

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 1; k <= sent.size(); ++k) {
    //there are k beam lists. perform reduces down to list 1

    for (unsigned i = k - 1; i > 0; --i) { 
      //prune if size exceeds beam_size
      if (beam_chart[i].size() > beam_size) {
        std::sort(beam_chart[i].begin(), beam_chart[i].end(), cmp_particle_ptr_weights_as); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_chart[i].size(); j > beam_size; --j)
          beam_chart[i].pop_back();
      }

      //for every item in the list, add valid reduce actions to list i - 1 
      for (unsigned j = 0; (j < beam_chart[i].size()); ++j) {
        Words r_ctx = beam_chart[i][j]->reduce_context();
        double reduceleftarcp = reduce_lm.prob(static_cast<WordId>(kAction::la), r_ctx);
        double reducerightarcp = reduce_lm.prob(static_cast<WordId>(kAction::ra), r_ctx);
        //std::cout << "(la: " << reduceleftarcp << ", ra: " << reducerightarcp << ")" << " ";
        double reducep = reduceleftarcp + reducerightarcp;
       
        //TODO have option to make la/ra choice deterministic
        beam_chart[i-1].push_back(std::make_unique<ArcStandardParser>(*beam_chart[i][j]));
        if (i > 1) { //left arc only invalid when stack size is 2 **
          beam_chart[i-1].push_back(std::make_unique<ArcStandardParser>(*beam_chart[i][j]));

          beam_chart[i-1].back()->leftArc();
          beam_chart[i-1].back()->add_particle_weight(reduceleftarcp);
          beam_chart[i-1].rbegin()[1]->rightArc();
          beam_chart[i-1].rbegin()[1]->add_particle_weight(reducerightarcp); 

          if (k == sent.size()) {  
            beam_chart[i-1].back()->add_importance_weight(reducep); 
            beam_chart[i-1].rbegin()[1]->add_importance_weight(reducep); 
          } 
        } else {
          beam_chart[i-1].back()->rightArc();
          beam_chart[i-1].back()->add_particle_weight(reducerightarcp); 
          
          if (k == sent.size()) 
            beam_chart[i-1].back()->add_importance_weight(reducerightarcp/reducep); 
        }
      }
    }

    if ((beam_chart[0].size() > beam_size) || (k == sent.size())) {
        std::sort(beam_chart[0].begin(), beam_chart[0].end(), cmp_particle_ptr_weights_as); //handle pointers
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
          
          double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
          double wordp = 1;
          if (with_words)
            wordp = shift_lm.prob(beam_chart[i][j]->next_word(), w_ctx); 
          double tagp = tag_lm.prob(beam_chart[i][j]->next_tag(), t_ctx);

          //TODO change back
          beam_chart[i][j]->shift();
          beam_chart[i][j]->add_importance_weight(wordp); 
          beam_chart[i][j]->add_importance_weight(tagp); 
          beam_chart[i][j]->add_particle_weight(shiftp); 
          beam_chart[i][j]->add_particle_weight(wordp); 
          beam_chart[i][j]->add_particle_weight(tagp); 
        }
      }
      //insert new beam_chart[0] to increment indexes
      beam_chart.insert(beam_chart.begin(), AsParserList());
    } 
    //std::cout << std::endl; 
  }
  
  unsigned n = 0; 
  for (unsigned i = 0; (i < beam_chart[n].size()); ++i) 
    beam_chart[n][0]->add_beam_particle_weight(beam_chart[n][i]->particle_weight());

  for (unsigned i = 0; (i < 5) && (i < beam_chart[n].size()); ++i) {
    //print parses
    //beam_chart[n][i]->print_arcs();
    //std::cout << "\n";
    //std::cout << beam_chart[n][i]->actions_str() << "\n";

    //float dir_acc = (beam_chart[n][i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
    //std::cout << "  Dir Accuracy: " << dir_acc;
    //std::cout << "  Sample weight: " << (beam_chart[n][i]->particle_weight()) << std::endl;
  }  

  if (beam_chart[n].size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardParser(sent, tags);  
  } else
    return ArcStandardParser(*beam_chart[n][0]); 
}  


//more sophisticated beam parser
//binary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
ArcStandardParser beamParseSentence(Words sent, Words tags, ArcList gold_dep, unsigned beam_size, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kArcOrder>& arc_lm, PYPLM<kTagOrder>& tag_lm) {
  //index in beam_chart is depth-of-stack - 1
  std::vector<AsParserList> beam_chart; 
  beam_chart.push_back(AsParserList());
  beam_chart[0].push_back(std::make_unique<ArcStandardParser>(sent, tags)); 
  //beam_chart[0][0]->print_sentence(dict);
  //beam_chart[0][0]->print_tags(dict);
  
  //std::cout << "gold arcs: ";
  //gold_dep.print_arcs();
  
  //shift ROOT symbol (probability 1)
  beam_chart[0][0]->shift(); 

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 1; k <= sent.size(); ++k) {
    //there are k beam lists. perform reduces down to list 1

    for (unsigned i = k - 1; i > 0; --i) { 
      //prune if size exceeds beam_size
      if (beam_chart[i].size() > beam_size) {
        std::sort(beam_chart[i].begin(), beam_chart[i].end(), cmp_particle_ptr_weights_as); //handle pointers
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
        std::sort(beam_chart[0].begin(), beam_chart[0].end(), cmp_particle_ptr_weights_as); //handle pointers
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
          
          double shiftp = reduce_lm.prob(static_cast<WordId>(kAction::sh), r_ctx);
          double wordp = 1;
          if (with_words)
            wordp = shift_lm.prob(beam_chart[i][j]->next_word(), w_ctx); 
          double tagp = tag_lm.prob(beam_chart[i][j]->next_tag(), t_ctx);

          beam_chart[i][j]->shift();
          beam_chart[i][j]->add_importance_weight(wordp); 
          beam_chart[i][j]->add_importance_weight(tagp); 
          beam_chart[i][j]->add_particle_weight(shiftp); 
          beam_chart[i][j]->add_particle_weight(wordp); 
          beam_chart[i][j]->add_particle_weight(tagp); 
        }
      }
      //insert new beam_chart[0] to increment indexes
      beam_chart.insert(beam_chart.begin(), AsParserList());
    }   
  }
  
  //print parses
  unsigned n = 0; 
  for (unsigned i = 0; (i < 5) && (i < beam_chart[n].size()); ++i) {
    //beam_chart[n][i]->print_arcs();

    //float dir_acc = (beam_chart[n][i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
    //std::cout << "  Dir Accuracy: " << dir_acc;
    //std::cout << "  Sample weight: " << (beam_chart[n][i]->particle_weight()) << std::endl;
  }  

  if (beam_chart[n].size()==0) {
    //std::cout << "no parse found" << std::endl;
    return ArcStandardParser(sent, tags);  
  } else
    return ArcStandardParser(*beam_chart[n][0]); 
}  

template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcStandardParser staticGoldParseSentence(Words sent, Words tags, ArcList gold_dep, bool with_words, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  ArcStandardParser parser(sent, tags);
  
  kAction a = kAction::sh;
  while (!parser.is_terminal_configuration() && (a != kAction::re)) {
    a = parser.oracleNext(gold_dep);  
    if (a != kAction::re) {
      //update particle weight
      Words r_ctx = parser.reduce_context();
      double actionp = reduce_lm.prob(static_cast<WordId>(a), r_ctx);
      parser.add_particle_weight(actionp);

      if (a == kAction::sh) {
        Words t_ctx = parser.tag_context();
        Words w_ctx = parser.shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(parser.next_word(), w_ctx); 
        double tagp = tag_lm.prob(parser.next_tag(), t_ctx);
        parser.add_particle_weight(wordp);
        parser.add_particle_weight(tagp);

      }

      parser.execute_action(a);
    } 
  }

  return parser;
}
    
inline ArcStandardParser staticGoldParseSentence(Words sent, Words tags, ArcList gold_dep) {
  ArcStandardParser parser(sent, tags);
  
  kAction a = kAction::sh;
  while (!parser.is_terminal_configuration() && (a != kAction::re)) {
    a = parser.oracleNext(gold_dep);  
    if (a != kAction::re) 
      parser.execute_action(a);
  }

  return parser;
}

template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
ArcEagerParser staticEagerGoldParseSentence(Words sent, Words tags, ArcList gold_dep, bool with_words, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  ArcEagerParser parser(sent, tags);

  kAction a = kAction::sh;
  while (!parser.is_terminal_configuration() && !(parser.is_buffer_empty() && (a == kAction::sh))) {
    a = parser.oracleNext(gold_dep);
    if (!(parser.is_buffer_empty() && (a == kAction::sh))) {
      //update particle weight
      Words r_ctx = parser.reduce_context();
      double actionp = reduce_lm.prob(static_cast<WordId>(a), r_ctx);
      parser.add_particle_weight(actionp);

      if (a == kAction::sh || a == kAction::ra) {
        Words t_ctx = parser.tag_context(a);
        Words w_ctx = parser.shift_context();
        double wordp = 1; 
        if (with_words)
          wordp = shift_lm.prob(parser.next_word(), w_ctx); 
        double tagp = tag_lm.prob(parser.next_tag(), t_ctx);
        parser.add_particle_weight(wordp);
        parser.add_particle_weight(tagp);
      }
      
      std::cout << static_cast<WordId>(a) << " "; 
      parser.execute_action(a); 
    } else
      std::cout << static_cast<WordId>(a) << "NOP "; 
  }

  //std::cout << parser.actions_str() << std::endl;
  std::cout << std::endl;

  return parser;
}

inline ArcEagerParser staticEagerGoldParseSentence(Words sent, Words tags, ArcList gold_dep) {
  ArcEagerParser parser(sent, tags);
  /*for (auto w: sent)
    std::cout << w << " ";
  std::cout << std::endl;
  for (auto w: tags)
    std::cout << w << " ";
  std::cout << std::endl;*/
  //std::cout << sent.size() << " " << tags.size() << gold_dep.size() << std::endl;

  kAction a = kAction::sh;
  while (!parser.is_terminal_configuration() && !(parser.is_buffer_empty() && (a == kAction::sh))) {
    //assume that it will reach terminal configuration even if not a complete parse
    a = parser.oracleNext(gold_dep);
    //std::cout << static_cast<WordId>(a) << " (" << parser.buffer_next() << ") ";
    if (!(parser.is_buffer_empty() && (a == kAction::sh))) 
      parser.execute_action(a); 
  }
  //std::cout << std::endl;

  return parser;
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
  parser.buffer_tag(0);
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
      parser.buffer_tag(tag);
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
  parser.buffer_tag(0);
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
      parser.buffer_tag(tag);
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
  parser.buffer_tag(0);
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
      parser.buffer_tag(tag);
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
