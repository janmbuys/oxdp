#ifndef _GDP_HPYP_DP_TRAIN_H_
#define _GDP_HPYP_DP_TRAIN_H_

#include <iostream>
#include <unordered_map>
#include <cstdlib>
#include <chrono>

#include "hpyp_dp_parse.h"
#include "transition_parser.h"
#include "hpyplm/hpyplm.h"
#include "corpus/corpus.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

namespace oxlm {

//given a parse, extract its training examples (add to given vectors)
void extractParseTrainExamples(const ArcStandardParser& prop_parser, std::vector<Words>* examples_sh, std::vector<Words>* examples_re, std::vector<Words>* examples_arc, std::vector<Words>* examples_tag);

//construct training examples for the individual models (binary decisions)
//use cannonical derivation (not based on current model probability)
void constructTrainExamples(const std::vector<Words>& corpus_sents, const std::vector<Words>& corpus_tags, const std::vector<WxList>& corpus_deps, std::vector<Words>* examples_sh, std::vector<Words>* examples_re, std::vector<Words>* examples_arc, std::vector<Words>* examples_tag);

//update PYP model given old and new training examples
//?should we only have 1 sample, or more?
template <unsigned kOrder>
void updatePYPModel(int num_samples, MT19937& eng, const std::vector<Words>& old_examples, const std::vector<Words>& new_examples, PYPLM<kOrder>* lm) {
  std::vector<WordId> ctx(kOrder - 1, 0);

  //**order of removing and adding samples?
  
  //firstly remove all old samples
  for (const auto& s: old_examples) {
    WordId w = s[0];
    for (unsigned i = 0; ((i < kOrder - 1) && (i < s.size() - 1)); i++)
      ctx[i] = s[i+1];
    lm->decrement(w, ctx, eng);
  }

  for (int sample=0; sample < num_samples; ++sample) {
    for (const auto& s: new_examples) {
      WordId w = s[0];
      for (unsigned i = 0; ((i < kOrder - 1) && (i < s.size() - 1)); i++)
        ctx[i] = s[i+1];
      if (sample > 0) 
        lm->decrement(w, ctx, eng);
      lm->increment(w, ctx, eng);
    }

    if (sample % 30u == 29) 
      lm->resample_hyperparameters(eng);      
    if (sample % 10 == 9) {
      std::cerr << (sample + 1) << " iterations [LLH=" << lm->log_likelihood() << "]" << std::endl;
    }  
  }
}

//train a single PYP model from given training examples
template <unsigned kOrder>
void trainPYPModel(int num_samples, MT19937& eng, const std::vector<Words>& examples, PYPLM<kOrder>* lm) {
  std::vector<WordId> ctx(kOrder - 1, 0);

  for (int sample=0; sample < num_samples; ++sample) {
    for (const auto& s: examples) {
      WordId w = s[0];
      for (unsigned i = 0; ((i < kOrder - 1) && (i < s.size() - 1)); i++)
        ctx[i] = s[i+1];
      if (sample > 0) 
        lm->decrement(w, ctx, eng);
      lm->increment(w, ctx, eng);
    }

    if (sample % 30u == 29) 
      lm->resample_hyperparameters(eng);      
    if ((sample==0) || (sample % 10 == 9)) {
      std::cerr << (sample + 1) << " iterations [LLH=" << lm->log_likelihood() << "]" << std::endl;
    }  
  }
}  

template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
void trainUnsupervised(const std::vector<Words>& sents, const std::vector<Words>& tags, const std::vector<WxList>& gold_deps, int num_iterations, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kReduceOrder>* reduce_lm, PYPLM<kArcOrder>* arc_lm, PYPLM<kTagOrder>* tag_lm) {
  unsigned num_particles = 100;
  bool resample = false;

  //keep an example list for each sentence
  //std::vector<WordsList> examples_list_sh(sents.size(), WordsList());
  std::vector<WordsList> examples_list_tag(sents.size(), WordsList());
  std::vector<WordsList> examples_list_re(sents.size(), WordsList());
  std::vector<WordsList> examples_list_arc(sents.size(), WordsList());

  //repeat for number of samples: for each sentence, get training examples and update model
  for (int i = 0; i < num_iterations; ++i) {
    //?? how online should this be? update models after each sentence, or only after all sentences?
    //after each sentece makes sens

    for (unsigned j = 0; j < sents.size(); ++j) {
      WordsList new_examples_sh;
      WordsList new_examples_tag;
      WordsList new_examples_re;
      WordsList new_examples_arc;

      //get a parse (derivation) given the current parameters

      //arcs: ArcList(sents[j].size())
      ArcList gold_dep(sents[j].size());
      gold_dep.set_arcs(gold_deps[j]); //TODO just for eval
      ArcStandardParser sample_parse = particleParseSentence(sents[j], tags[j], gold_dep, num_particles, resample, dict, eng, *shift_lm, *reduce_lm, *arc_lm, *tag_lm);

      //extract new training examples
      extractParseTrainExamples(sample_parse, &new_examples_sh, &new_examples_re, &new_examples_arc, &new_examples_tag);
      
      //update with new training examples
      //TODO may have to increase num_samples here
      //updatePYPModel(1, eng, examples_list_sh[j], new_examples_sh, shift_lm);
      //examples_list_sh[j] = new_examples_sh;      
      updatePYPModel(1, eng, examples_list_tag[j], new_examples_tag, tag_lm);
      examples_list_tag[j] = new_examples_tag;      
      updatePYPModel(1, eng, examples_list_re[j], new_examples_re, reduce_lm);
      examples_list_re[j] = new_examples_re;      
      updatePYPModel(1, eng, examples_list_arc[j], new_examples_arc, arc_lm);
      examples_list_arc[j] = new_examples_arc;      
      std::cerr << ".";
    }

    std::cerr << "\n";
    if (num_iterations % 30 == 29) {
      //shift_lm->resample_hyperparameters(eng);      
      reduce_lm->resample_hyperparameters(eng);      
      arc_lm->resample_hyperparameters(eng);      
      tag_lm->resample_hyperparameters(eng);      
    }
    if ((num_iterations==0) || (num_iterations % 10 == 9)) {
      std::cerr << (num_iterations + 1) << " iterations";
      //std::cerr << " [Shift LLH=" << shift_lm->log_likelihood() << "]"; 
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]";    
      std::cerr << " [Reduce LLH=" << reduce_lm->log_likelihood() << "]";
      std::cerr << " [Arc LLH=" << arc_lm->log_likelihood() << "]";
      std::cerr << std::endl;
    }
  }
}

}
#endif
