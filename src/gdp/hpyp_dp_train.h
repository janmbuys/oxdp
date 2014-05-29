#ifndef _CGDP_DPT_H_
#define _CGDP_DPT_H_

#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "transition_parser.h"
#include "hpyplm/hpyplm.h"
#include "corpus/corpus.h"
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

namespace oxlm {

//three-way action decision
void train_raw(std::string train_file, Dict& dict, std::set<WordId>& vocabs, std::vector<Words>& corpussh, std::vector<Words>& corpusre) {
  std::vector<Words> corpuss;
  std::vector<WxList> corpusd;

  std::string sentence_file = train_file + ".sentences";
  std::cerr << "Reading sentences...\n";
  ReadFromFile(sentence_file, &dict, &corpuss, &vocabs);
  std::cerr << "Corpus size: " << corpuss.size() << " sentences\t (" << vocabs.size() << " word types)\n";
  
  std::string dependencies_file = train_file + ".dependencies";
  std::cerr << "Reading dependencies...\n";
  ReadFromDependencyFile(dependencies_file, &corpusd);

  // apply the oracle
  for (unsigned i = 0; i < corpuss.size(); ++i) {
    ArcStandardParser parser(corpuss[i], 2);
    bool has_parse = parser.sentence_oracle(corpusd[i]);
    if (is_projective_dependency(corpusd[i])) {
      if (has_parse) {
        //std::cout << parser.actions_str() << std::endl;
        //add the extracted training examples
        WordsList sh_ctxs = parser.shift_contexts();        
        for (unsigned j = 0; j < parser.sentence_length(); ++j) {
          //corpussh.push_back(Words(1, parser.get_sentence()[j]));
          Words ins(1, parser.get_sentence()[j]);
          //corpussh.back()->insert(corpussh.end(), sh_ctxs[j].begin(), sh_ctxs[j].end());
          ins.insert(ins.end(), sh_ctxs[j].begin(), sh_ctxs[j].end());
          corpussh.push_back(ins);
        }
              
        WordsList act_ctxs = parser.action_contexts();
        for (unsigned j = 0; j < act_ctxs.size(); ++j) {
          Words ins(1, static_cast<WordId>(parser.action_predictions()[j]));
          ins.insert(ins.end(), act_ctxs[j].begin(), act_ctxs[j].end());
          //for (auto a: ins)
          //    std::cout << a << " ";
          //std::cout << std::endl;
          corpusre.push_back(ins);
        }
            
      } else
            std::cerr << "oracle failure" << std::endl;
    } 
  }
}

//with binary decisions
void train_raw(std::string train_file, Dict& dict, unsigned ctx_size, std::vector<Words>& corpussh, std::vector<Words>& corpusre, std::vector<Words>& corpusarc, std::vector<Words>& corpustag) {
  std::vector<Words> corpuss;
  std::vector<Words> corpuspos;
  std::vector<WxList> corpusd;

  /*
  std::string sentence_file = train_file + ".sentences";
  std::cerr << "Reading corpus...\n";
  ReadFromFile(sentence_file, &dict, &corpuss, &vocabs);
  std::cerr << "E-corpus size: " << corpuss.size() << " sentences\t (" << vocabs.size() << " word types)\n";
  
  std::string dependencies_file = train_file + ".dependencies";
  std::cerr << "Reading corpus...\n";
  ReadFromDependencyFile(dependencies_file, &corpusd);
  std::cerr << "E-corpus size: " << corpusd.size() << " sentences\n";  */

  std::cerr << "Reading training data...\n";
  ReadFromConllFile(train_file, &dict, &corpuss, &corpuspos, &corpusd, false);
  std::cerr << "Corpus size: " << corpusd.size() << " sentences\t (" << dict.size() << " word types, " << dict.pos_size() << " tags)\n";

  // apply the oracle
  for (unsigned i = 0; (i < corpuss.size()); ++i) {
    ArcStandardParser parser(corpuss[i], corpuspos[i], ctx_size);
    /*for (auto d: corpuss[i])
      std::cout << d << " ";
    std::cout << std::endl;
    
    for (auto d: corpusd[i])
      std::cout << d << " ";
    std::cout << std::endl;  */

    bool has_parse = parser.sentence_oracle(corpusd[i]);
    if (is_projective_dependency(corpusd[i])) {
      if (has_parse) {
        WordsList sh_ctxs = parser.shift_contexts();
        if (sh_ctxs.size() != parser.sentence_length())
          std::cerr << "unmatching context lengths" << std::endl;

        //TODO update to use pos context
        for (unsigned j = 0; j < parser.sentence_length(); ++j) {
          Words ins(1, parser.get_sentence()[j]);
          ins.insert(ins.end(), sh_ctxs[j].begin(), sh_ctxs[j].end());
          corpussh.push_back(ins);
        }
      
        WordsList tag_ctxs = parser.tag_contexts();
        for (unsigned j = 0; j < parser.sentence_length(); ++j) {
          Words ins(1, parser.get_postags()[j]);
          ins.insert(ins.end(), tag_ctxs[j].begin(), tag_ctxs[j].end());
          corpustag.push_back(ins);
        }

        //reduce contexts = action contexts
        WordsList re_ctxs = parser.action_contexts();
        for (unsigned j = 0; j < re_ctxs.size(); ++j) {
          Words ins(1, static_cast<WordId>(parser.reduce_predictions()[j]));
          ins.insert(ins.end(), re_ctxs[j].begin(), re_ctxs[j].end());
          corpusre.push_back(ins);
        }

        //left/right arc contexts
        WordsList arc_ctxs = parser.arc_contexts();
        for (unsigned j = 0; j < arc_ctxs.size(); ++j) {
          Words ins(1, static_cast<WordId>(parser.arc_predictions()[j]));
          ins.insert(ins.end(), arc_ctxs[j].begin(), arc_ctxs[j].end());
          corpusarc.push_back(ins);
        }
            
      } else
        std::cerr << "oracle failure" << std::endl;
    } 
  }
}

template <unsigned N>
void train_lm(int samples, MT19937& eng, Dict& dict, std::vector<Words>& corpus, PYPLM<N>& lm) {
  std::vector<WordId> ctx(N - 1, 0);

  for (int sample=0; sample < samples; ++sample) {
    for (const auto& s : corpus) {
      WordId w = s[0];
      for (unsigned i = 0; ((i < N - 1) && (i < s.size() - 1)); i++)
        ctx[i] = s[i+1];
      if (sample > 0) 
        lm.decrement(w, ctx, eng);
      lm.increment(w, ctx, eng);
    }

    if (sample % 30u == 29) 
      lm.resample_hyperparameters(eng);      
    if (sample % 10 == 9) {
      std::cerr << (sample + 1) << " iterations [LLH=" << lm.log_likelihood() << "]" << std::endl;
    }  
  }
}

}
#endif
