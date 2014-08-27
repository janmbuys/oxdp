#ifndef _GDP_HPYP_DP_TRAIN_H_
#define _GDP_HPYP_DP_TRAIN_H_

#include <iostream>
#include <unordered_map>
#include <cstdlib>
#include <chrono>

#include "hpyp_dp_parse.h"
#include "transition_parser.h"
#include "eisner_parser.h"
#include "hpyplm/hpyplm.h"
#include "corpus/dict.h"
#include "utils/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

namespace oxlm {

//given a parse, extract its training examples (add to given vectors)
//for arc-eager parser (4-way decisions)
inline void extractParseTrainExamples(const ArcEagerParser& prop_parser, std::vector<Words>* examples_sh, std::vector<Words>* examples_re, std::vector<Words>* examples_tag) {
  ArcEagerParser parser(prop_parser.sentence(), prop_parser.tags());
 
  for(kAction& a: prop_parser.actions()) {
    Words re_ctx = parser.reduce_context();
    Words re_tuple(1, static_cast<WordId>(kAction::re));
    re_tuple.insert(re_tuple.end(), re_ctx.begin(), re_ctx.end());
    
    if (a == kAction::sh || a == kAction::ra) {
      //TODO later include ra or not as feature for sh
      //word prediction
      if (!(examples_sh == nullptr)) {
        Words sh_ctx = parser.shift_context();
        Words sh_tuple(1, parser.next_word());
        sh_tuple.insert(sh_tuple.end(), sh_ctx.begin(), sh_ctx.end());
        examples_sh->push_back(sh_tuple);
      }  

      //tag prediction
      Words tag_ctx = parser.tag_context(a);
      Words tag_tuple(1, parser.next_tag());
      tag_tuple.insert(tag_tuple.end(), tag_ctx.begin(), tag_ctx.end());
      examples_tag->push_back(tag_tuple);
    }

    //action decision
    re_tuple[0] = static_cast<WordId>(a);
    examples_re->push_back(re_tuple);  
    parser.execute_action(a);
     
  }
}

//for Eisner parser
inline void extractParseTrainExamples(const EisnerParser& parser, std::vector<Words>* examples_sh, std::vector<Words>* examples_tag) {
  //we should actually extract training examples in the same order as generation,
  //but this shouldn't matter too much in practice
  for (WordIndex i = 1; i < static_cast<int>(parser.sentence_length()); ++i) {
    //training example head j to i
    WordIndex j = parser.head_at(i);
    if (j == -1)
      continue;
    WordIndex prev_child = 0;
    if (i < j)
      prev_child = parser.prev_left_child(i, j);
    else if (j < i)
      prev_child = parser.prev_right_child(i, j);

    Words tag_ctx = parser.tag_context(i, j, prev_child);
    Words tag_tuple(1, parser.tag_at(i));
    tag_tuple.insert(tag_tuple.end(), tag_ctx.begin(), tag_ctx.end());
    examples_tag->push_back(tag_tuple);

    if (!(examples_sh == nullptr)) {
      Words sh_ctx = parser.shift_context(i, j, prev_child);
      Words sh_tuple(1, parser.sentence_at(i));
      sh_tuple.insert(sh_tuple.end(), sh_ctx.begin(), sh_ctx.end());
      examples_sh->push_back(sh_tuple);
    }
  } 

  for (WordIndex j = 0; j < static_cast<int>(parser.sentence_length()); ++j) {
    //training examples: no further left children
    WordIndex prev_child = parser.leftmost_child(j);
 
    Words tag_ctx = parser.tag_context(-1, j, prev_child);
    Words tag_tuple(1, 1);
    tag_tuple.insert(tag_tuple.end(), tag_ctx.begin(), tag_ctx.end());
    examples_tag->push_back(tag_tuple);

    if (!(examples_sh == nullptr)) {
      Words sh_ctx = parser.shift_context(-1, j, prev_child);
      Words sh_tuple(1, 1);
      sh_tuple.insert(sh_tuple.end(), sh_ctx.begin(), sh_ctx.end());
      examples_sh->push_back(sh_tuple);
    }

    //training examples: no further right children
    prev_child = parser.rightmost_child(j);
 
    tag_ctx = parser.tag_context(parser.sentence_length(), j, prev_child);
    tag_tuple = {1};
    tag_tuple.insert(tag_tuple.end(), tag_ctx.begin(), tag_ctx.end());
    examples_tag->push_back(tag_tuple);

    if (!(examples_sh == nullptr)) {
      Words sh_ctx = parser.shift_context(parser.sentence_length(), j, prev_child);
      Words sh_tuple(1, 1);
      sh_tuple.insert(sh_tuple.end(), sh_ctx.begin(), sh_ctx.end());
      examples_sh->push_back(sh_tuple);
    }  
  } 
  
}

//given a parse, extract its training examples (add to given vectors)
//for three-way decisions
inline void extractParseTrainExamples(const ArcStandardParser& prop_parser, std::vector<Words>* examples_sh, std::vector<Words>* examples_re, std::vector<Words>* examples_tag) {
  ArcStandardParser parser(prop_parser.sentence(), prop_parser.tags());
 
  for (kAction& a: prop_parser.actions()) {
    Words re_ctx = parser.reduce_context();
    Words re_tuple(1, static_cast<WordId>(kAction::re));
    re_tuple.insert(re_tuple.end(), re_ctx.begin(), re_ctx.end());
    
    if (a == kAction::sh) {
      //word prediction
      if (!(examples_sh == nullptr)) {
        Words sh_ctx = parser.shift_context();
        Words sh_tuple(1, parser.next_word());
        sh_tuple.insert(sh_tuple.end(), sh_ctx.begin(), sh_ctx.end());
        examples_sh->push_back(sh_tuple);
      }  

      //tag prediction
      Words tag_ctx = parser.tag_context();
      Words tag_tuple(1, parser.next_tag());
      tag_tuple.insert(tag_tuple.end(), tag_ctx.begin(), tag_ctx.end());
      examples_tag->push_back(tag_tuple);
    }

    //action decision
    re_tuple[0] = static_cast<WordId>(a);
    examples_re->push_back(re_tuple);  
    //std::cout << static_cast<WordId>(a) << std::endl;
    parser.execute_action(a);
  }
}


//given a parse, extract its training examples (add to given vectors)
//for binary decisions
inline void extractParseTrainExamples(const ArcStandardParser& prop_parser, std::vector<Words>* examples_sh, std::vector<Words>* examples_re, std::vector<Words>* examples_arc, std::vector<Words>* examples_tag) {
  ArcStandardParser parser(prop_parser.sentence(), prop_parser.tags());
 
  for(kAction& a: prop_parser.actions()) {
    Words re_ctx = parser.reduce_context();
    Words re_tuple(1, static_cast<WordId>(kAction::re));
    re_tuple.insert(re_tuple.end(), re_ctx.begin(), re_ctx.end());
    
    if (a == kAction::sh) {
      //word prediction
      if (!(examples_sh == nullptr)) {
        Words sh_ctx = parser.shift_context();
        Words sh_tuple(1, parser.next_word());
        sh_tuple.insert(sh_tuple.end(), sh_ctx.begin(), sh_ctx.end());
        examples_sh->push_back(sh_tuple);
      }  

      //tag prediction
      Words tag_ctx = parser.tag_context();
      Words tag_tuple(1, parser.next_tag());
      tag_tuple.insert(tag_tuple.end(), tag_ctx.begin(), tag_ctx.end());
      examples_tag->push_back(tag_tuple);

      //shift decision
      re_tuple[0] = static_cast<WordId>(kAction::sh);
      examples_re->push_back(re_tuple);  
              
      parser.shift();
    } else if (a == kAction::la) {
      Words arc_ctx = parser.arc_context();
      Words arc_tuple(1, static_cast<WordId>(kAction::la)); 
      arc_tuple.insert(arc_tuple.end(), arc_ctx.begin(), arc_ctx.end());
      examples_arc->push_back(arc_tuple);
      examples_re->push_back(re_tuple);  
       
      parser.leftArc();
    } else if (a == kAction::ra) {
      Words arc_ctx = parser.arc_context();
      Words arc_tuple(1, static_cast<WordId>(kAction::ra)); 
      arc_tuple.insert(arc_tuple.end(), arc_ctx.begin(), arc_ctx.end());
      examples_arc->push_back(arc_tuple);
      examples_re->push_back(re_tuple);  
     
      parser.rightArc();
    } 
  }
}

//update PYP model to insert new training examples or to remove old training examples
template <unsigned kOrder>
void updatePYPModel(bool insert, MT19937& eng, const std::vector<Words>& examples, PYPLM<kOrder>* lm) {
  std::vector<WordId> ctx(kOrder - 1, 0);
  
  for (const auto& s: examples) {
    WordId w = s[0];
    for (unsigned i = 0; ((i < kOrder - 1) && (i < s.size() - 1)); i++)
      ctx[i] = s[i+1];
    if (insert)
      lm->increment(w, ctx, eng);
    else
      lm->decrement(w, ctx, eng);
  }
}

//for Eisner parsing
template<unsigned kShiftOrder, unsigned kTagOrder>
void trainSupervisedEisnerParser(const std::vector<Words>& sents, const std::vector<Words>& tags, const std::vector<WxList>& gold_deps, int num_iterations, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kTagOrder>* tag_lm) {

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_sh(sents.size(), WordsList());
  std::vector<WordsList> examples_list_tag(sents.size(), WordsList());
  
  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {
     //std::cout << "Training iter " << iter << std::endl;

    for (unsigned j = 0; j < sents.size(); ++j) {
      //if (tags[j].size() <= 11) {
      //remove old sample from model
      if (iter > 0) {
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        if (with_words)
          updatePYPModel(false, eng, examples_list_sh[j], shift_lm);
      }

      if (iter == 0) {
        EisnerParser sample_parse(sents[j], tags[j], gold_deps[j]);
        extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_tag[j]);
        //for (auto ex: examples_list_tag[j])
        //  std::cout << dict.lookupTag(ex[0]) << " " << dict.lookupTag(ex[1]) << std::endl;
      } 

      //update with new sample
      if (with_words)
        updatePYPModel(true, eng, examples_list_sh[j], shift_lm);
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
    //}
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words)
        std::cerr << " [Shift LLH=" << shift_lm->log_likelihood() << "]\n"; 
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words) {
        shift_lm->resample_hyperparameters(eng);      
        std::cerr << "  [Shift LLH=" << shift_lm->log_likelihood() << "]\n\n";    
      }
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
    }
  }
}

//for three-way decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
void trainSupervisedParser(const std::vector<Words>& sents, const std::vector<Words>& tags, const std::vector<WxList>& gold_deps, int num_iterations, bool with_words, bool arceager, bool static_oracle, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kReduceOrder>* reduce_lm, PYPLM<kTagOrder>* tag_lm) {

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_sh(sents.size(), WordsList());
  std::vector<WordsList> examples_list_tag(sents.size(), WordsList());
  std::vector<WordsList> examples_list_re(sents.size(), WordsList());
  
  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {
     //std::cout << "Training iter " << iter << std::endl;

    for (unsigned j = 0; j < sents.size(); ++j) {
      //if (tags[j].size() <= 11) {
      //remove old sample from model
      if (iter > 0) {
        if (with_words)
          updatePYPModel(false, eng, examples_list_sh[j], shift_lm);
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
      }

      //get a parse (derivation) given the current parameters
      ArcList gold_dep(gold_deps[j].size());
      gold_dep.set_arcs(gold_deps[j]);  //problematic

      if (iter == 0) {
        if (arceager) {
          //std::cout << "gold " << j << std::endl;
          ArcEagerParser sample_parse = staticEagerGoldParseSentence(sents[j], tags[j], gold_dep);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
          //gold_dep.print_arcs();
          //sample_parse.print_arcs();
        } else { 
          ArcStandardParser sample_parse = staticGoldParseSentence(sents[j], tags[j], gold_dep);
          //std::cout << sample_parse.actions_str() << "\n";
          //sample_parse.print_arcs();
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
        }
      } else if (!static_oracle) {
        unsigned num_particles = 100;
        bool resample = false;
        
        if (arceager) {
          ArcEagerParser sample_parse = particleEagerGoldParseSentence(sents[j], tags[j], gold_dep, num_particles, resample, with_words, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
        } else {  
          ArcStandardParser sample_parse = particleDynamicGoldParseSentence(sents[j], tags[j], gold_dep, num_particles, resample, with_words, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
          //technically need a MH acceptance test
        }
      } 

      //update with new sample
      if (with_words)
        updatePYPModel(true, eng, examples_list_sh[j], shift_lm);
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
      updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
    //}
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words)
        std::cerr << " [Shift LLH=" << shift_lm->log_likelihood() << "]\n"; 
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
      std::cerr << " [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n";
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words) {
        shift_lm->resample_hyperparameters(eng);      
        std::cerr << "  [Shift LLH=" << shift_lm->log_likelihood() << "]\n\n";    
      }
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
      reduce_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n\n";
    }
  
  }
}

//for binary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
void trainSupervisedParser(const std::vector<Words>& sents, const std::vector<Words>& tags, const std::vector<WxList>& gold_deps, int num_iterations, bool with_words, bool static_oracle, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kReduceOrder>* reduce_lm, PYPLM<kArcOrder>* arc_lm, PYPLM<kTagOrder>* tag_lm) {
  unsigned num_particles = 100;
  bool resample = false;

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_sh(sents.size(), WordsList());
  std::vector<WordsList> examples_list_tag(sents.size(), WordsList());
  std::vector<WordsList> examples_list_re(sents.size(), WordsList());
  std::vector<WordsList> examples_list_arc(sents.size(), WordsList());
  
  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {

    for (unsigned j = 0; j < sents.size(); ++j) {
      //remove old sample from model
      if (iter > 0) {
        if (with_words)
          updatePYPModel(false, eng, examples_list_sh[j], shift_lm);
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
        updatePYPModel(false, eng, examples_list_arc[j], arc_lm);
      }

      //get a parse (derivation) given the current parameters
      ArcList gold_dep(sents[j].size());
      gold_dep.set_arcs(gold_deps[j]); 
      ArcStandardParser sample_parse;
     
      if (static_oracle) {
        if (iter == 0) {
          sample_parse = staticGoldParseSentence(sents[j], tags[j], gold_dep);

          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_arc[j], &examples_list_tag[j]);
        }
      } else {
        sample_parse = particleGoldParseSentence(sents[j], tags[j], gold_dep, num_particles, resample, with_words, dict, eng, *shift_lm, *reduce_lm, *arc_lm, *tag_lm);

        extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_arc[j], &examples_list_tag[j]);
        //technically need a MH acceptance test
      }

      //update with new sample
      if (with_words)
        updatePYPModel(true, eng, examples_list_sh[j], shift_lm);
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
      updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
      updatePYPModel(true, eng, examples_list_arc[j], arc_lm);
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words)
        std::cerr << " [Shift LLH=" << shift_lm->log_likelihood() << "]\n"; 
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
      std::cerr << " [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n";
      std::cerr << " [Arc LLH=" << arc_lm->log_likelihood() << "]\n";
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words) {
        shift_lm->resample_hyperparameters(eng);      
        std::cerr << "  [Shift LLH=" << shift_lm->log_likelihood() << "]\n\n";    
      }
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
      reduce_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n\n";
      arc_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Arc LLH=" << arc_lm->log_likelihood() << "]\n\n";
    }
  }
}

//for three-way decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
void trainSemisupervisedParser(const std::vector<Words>& sents, const std::vector<Words>& tags, const std::vector<WxList>& gold_deps, int num_iterations, bool with_words, bool arceager, bool static_oracle, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kReduceOrder>* reduce_lm, PYPLM<kTagOrder>* tag_lm) {

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_sh(sents.size(), WordsList());
  std::vector<WordsList> examples_list_tag(sents.size(), WordsList());
  std::vector<WordsList> examples_list_re(sents.size(), WordsList());
  
  unsigned split_n = 4000;
  unsigned num_particles = 100;
  bool resample = false;
  unsigned max_sent_length = 15;

  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {
     //std::cout << "Training iter " << iter << std::endl;

    for (unsigned j = 0; j < split_n; ++j) {
      //if (tags[j].size() <= (max_sent_length+1)) {
      //remove old sample from model
      if (iter > 0) {
        if (with_words)
          updatePYPModel(false, eng, examples_list_sh[j], shift_lm);
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
      }

      //get a parse (derivation) given the current parameters
      ArcList gold_dep(sents[j].size());
      gold_dep.set_arcs(gold_deps[j]); 

      if (iter == 0) {
        if (arceager) {
          ArcEagerParser sample_parse = staticEagerGoldParseSentence(sents[j], tags[j], gold_dep);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
        } else { 
          ArcStandardParser sample_parse = staticGoldParseSentence(sents[j], tags[j], gold_dep);
          //sample_parse.print_arcs();
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
        }
        //std::cout << "\n" << sample_parse.actions_str() << "\n";
      } else if (!static_oracle) {
        
        if (arceager) {
          ArcEagerParser sample_parse = particleEagerGoldParseSentence(sents[j], tags[j], gold_dep, num_particles, resample, with_words, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
        } else {  
          ArcStandardParser sample_parse = particleDynamicGoldParseSentence(sents[j], tags[j], gold_dep, num_particles, resample, with_words, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
          //technically need a MH acceptance test
        }
      } 

      //update with new sample
      if (with_words)
        updatePYPModel(true, eng, examples_list_sh[j], shift_lm);
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
      updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
    //}
    }

    //Unsupervised part
    num_particles = 1000;

    for (unsigned j = split_n; j < tags.size(); ++j) {
      if (tags[j].size() <= (max_sent_length+1)) {
        
        //remove old sample from model (if there is any)
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
        
        //only for evaluation
        ArcList gold_dep(tags[j].size());
        gold_dep.set_arcs(gold_deps[j]); 
    
        //sample new parse (derivation) 
        //technically need a MH acceptance test
        if (arceager) {
          ArcEagerParser sample_parse = particleEagerParseSentence(Words(tags[j].size(), '_'), tags[j], gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, nullptr, &examples_list_re[j], &examples_list_tag[j]);
        } else {
          ArcStandardParser sample_parse = particleParseSentence(Words(tags[j].size(), '_'), tags[j], gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, nullptr, &examples_list_re[j], &examples_list_tag[j]);
        }

        //update with new sample
        updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
      }
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words)
        std::cerr << " [Shift LLH=" << shift_lm->log_likelihood() << "]\n"; 
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
      std::cerr << " [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n";
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words) {
        shift_lm->resample_hyperparameters(eng);      
        std::cerr << "  [Shift LLH=" << shift_lm->log_likelihood() << "]\n\n";    
      }
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
      reduce_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n\n";
    }
  
  }
}



//for ternary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder,  unsigned kTagOrder>
void trainUnsupervisedParser(const std::vector<Words>& tags, const std::vector<WxList>& gold_deps, int num_iterations, bool init, bool arceager, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kReduceOrder>* reduce_lm, PYPLM<kTagOrder>* tag_lm) {
  unsigned num_particles = 1000;
  bool resample = false;
  unsigned max_sent_length = 15;

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_tag(tags.size(), WordsList());
  std::vector<WordsList> examples_list_re(tags.size(), WordsList());
 
  std::string log_file = "arceager.scores.out";
  std::ofstream outl;
  outl.open(log_file); 

  if (init) {
    //initialize tag_lm with a sequential trigram model
    std::vector<WordId> ctx(kTagOrder - 1, 0);
    for (unsigned j = 0; j < tags.size(); ++j) {
      ctx.resize(kTagOrder - 1);
      for (unsigned i = 0; i < tags[j].size(); ++i) {
        WordId tag = tags[j][i];
        Words tag_tuple(1, tag);
        tag_tuple.insert(tag_tuple.end(), ctx.begin() + (ctx.size() - (kTagOrder - 1)), ctx.end());
        examples_list_tag[j].push_back(tag_tuple);
        ctx.push_back(tag);
      }
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm); 
    }
  } 

  //TODO try different initializations
  
  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {

    //for (unsigned j = 0; j < tags.size(); ++j) {
    for (unsigned j = 4000; j < tags.size(); ++j) {
      if (tags[j].size() <= (max_sent_length+1)) {
      //if (tags[j].size() <= std::min(iter/2 + 3, 11)) {

      //if (((iter < 20) && (tags[j].size() <= 11)) || 
      //    ((iter >= 20) && (iter < 40) && (tags[j].size()) <= 21)) {
        
        //remove old sample from model (if there is any)
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
        
        //if (iter > 0) {
        //} else if (init) {
        //  updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        //}

        //only for evaluation
        ArcList gold_dep(tags[j].size());
        gold_dep.set_arcs(gold_deps[j]); 
    
        //sample new parse (derivation) 
        if (arceager) {
          ArcEagerParser sample_parse;
          if ((iter==0) && init)
            sample_parse = particleInitEagerParseSentence(Words(tags[j].size(), '_'), tags[j], gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *tag_lm);
          else
            sample_parse = particleEagerParseSentence(Words(tags[j].size(), '_'), tags[j], gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          //technically need a MH acceptance test
         
          extractParseTrainExamples(sample_parse, nullptr, &examples_list_re[j], &examples_list_tag[j]);
        } else {
          ArcStandardParser sample_parse;
          if ((iter==0) && init)
            sample_parse = particleInitParseSentence(Words(tags[j].size(), '_'), tags[j], gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *tag_lm);
          else
            sample_parse = particleParseSentence(Words(tags[j].size(), '_'), tags[j], gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          //technically need a MH acceptance test
         
          extractParseTrainExamples(sample_parse, nullptr, &examples_list_re[j], &examples_list_tag[j]);
 
        }

        //update with new sample
        updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
      }
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
      std::cerr << " [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n";
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
      reduce_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n\n";
      
      //evaluate
      std::string test_file;
      double acc;
  
      test_file = "english-wsj-conll07-nowords-nopunc10/english_wsj_dev.conll";
      acc = evaluate(test_file, arceager, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
      outl << acc << " ";

      test_file = "english-wsj-conll07-nowords-nopunc20/english_wsj_dev.conll";
      acc = evaluate(test_file, arceager, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
      outl << acc << " ";

      test_file = "english-wsj-conll07-nowords-nopunc/english_wsj_dev.conll";
      acc = evaluate(test_file, arceager, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
      outl << acc << std::endl;
    }
  }

  outl.close();
}

//for binary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
void trainUnsupervisedParser(const std::vector<Words>& tags, const std::vector<WxList>& gold_deps, int num_iterations, bool init, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kReduceOrder>* reduce_lm, PYPLM<kArcOrder>* arc_lm, PYPLM<kTagOrder>* tag_lm) {
  unsigned num_particles = 1000;
  bool resample = false;

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_tag(tags.size(), WordsList());
  std::vector<WordsList> examples_list_re(tags.size(), WordsList());
  std::vector<WordsList> examples_list_arc(tags.size(), WordsList());
 
  if (init) {
    //initialize tag_lm with a sequential trigram model
    std::vector<WordId> ctx(kTagOrder - 1, 0);
    for (unsigned j = 0; j < tags.size(); ++j) {
      ctx.resize(kTagOrder - 1);
      for (unsigned i = 0; i < tags[j].size(); ++i) {
        WordId tag = tags[j][i];
        Words tag_tuple(1, tag);
        tag_tuple.insert(tag_tuple.end(), ctx.begin() + (ctx.size() - (kTagOrder - 1)), ctx.end());
        examples_list_tag[j].push_back(tag_tuple);
        ctx.push_back(tag);
      }
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm); 
    }
  } 
  
  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {

    for (unsigned j = 0; j < tags.size(); ++j) {
      if (((iter < 20) && (tags[j].size() <= 11)) || 
          ((iter >= 20) && (iter < 40) && (tags[j].size()) <= 21)) {

        //remove old sample from model
        if (iter > 0) {
          updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
          updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
          updatePYPModel(false, eng, examples_list_arc[j], arc_lm);
        } else if (init) {
          updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        }

        //only for evaluation
        ArcList gold_dep(tags[j].size());
        gold_dep.set_arcs(gold_deps[j]); 
    
        //sample new parse (derivation) 
        ArcStandardParser sample_parse = particleParseSentence(Words(tags[j].size(), '_'), tags[j], gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *reduce_lm, *arc_lm, *tag_lm);
        //technically need a MH acceptance test
      
        extractParseTrainExamples(sample_parse, nullptr, &examples_list_re[j], &examples_list_arc[j], &examples_list_tag[j]);

        //update with new sample
        updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
        updatePYPModel(true, eng, examples_list_arc[j], arc_lm);
      }
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
      std::cerr << " [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n";
      std::cerr << " [Arc LLH=" << arc_lm->log_likelihood() << "]\n";
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
      reduce_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n\n";
      arc_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Arc LLH=" << arc_lm->log_likelihood() << "]\n\n";
    }
  }
}

}
#endif
