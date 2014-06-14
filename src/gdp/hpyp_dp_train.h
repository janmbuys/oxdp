#ifndef _CGDP_DPT_H_
#define _CGDP_DPT_H_

#include <iostream>
#include <unordered_map>
#include <cstdlib>
#include <chrono>

#include "transition_parser.h"
#include "hpyplm/hpyplm.h"
#include "corpus/corpus.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

namespace oxlm {

//construct training examples for the individual models (binary decisions)
//use cannonical derivation (not based on current model probability)
inline void constructTrainExamples(const std::vector<Words>& corpus_sents, const std::vector<Words>& corpus_tags, const std::vector<WxList>& corpus_deps, std::vector<Words>* examples_sh, std::vector<Words>* examples_re, std::vector<Words>* examples_arc, std::vector<Words>* examples_tag) {
  std::cerr << "Constructing training examples...\n";

  // apply the oracle
  for (unsigned i = 0; (i < corpus_sents.size()); ++i) {
    ArcStandardParser parser(corpus_sents[i], corpus_tags[i]);
    ArcList gold_deps(corpus_sents[i].size());
    gold_deps.set_arcs(corpus_deps[i]);

    kAction a = kAction::sh;
    while (!parser.is_terminal_configuration() && (a != kAction::re)) {
      a = parser.oracleNext(gold_deps);  
      Words re_ctx = parser.tag_context();
      Words re_tuple(1, static_cast<WordId>(kAction::re));
      re_tuple.insert(re_tuple.end(), re_ctx.begin(), re_ctx.end());

      if (a == kAction::sh) {
        //word prediction
        Words sh_ctx = parser.word_tag_next_context();
        Words sh_tuple(1, parser.next_word());
        sh_tuple.insert(sh_tuple.end(), sh_ctx.begin(), sh_ctx.end());
        examples_sh->push_back(sh_tuple);
        
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
        Words arc_ctx = parser.tag_context();
        Words arc_tuple(1, static_cast<WordId>(kAction::la)); 
        arc_tuple.insert(arc_tuple.end(), arc_ctx.begin(), arc_ctx.end());
        examples_arc->push_back(arc_tuple);
        examples_re->push_back(re_tuple);  
        
        parser.leftArc();
      } else if (a == kAction::ra) {
        Words arc_ctx = parser.tag_context();
        Words arc_tuple(1, static_cast<WordId>(kAction::ra)); 
        arc_tuple.insert(arc_tuple.end(), arc_ctx.begin(), arc_ctx.end());
        examples_arc->push_back(arc_tuple);
        examples_re->push_back(re_tuple);  
      
        parser.rightArc();
      } 
    }   

    //for now, don't do anything after getting stuck (just stop training from that examples)
    // if (a == kAction::re) 
      //std::cerr << "oracle failure" << std::endl;
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

}
#endif
