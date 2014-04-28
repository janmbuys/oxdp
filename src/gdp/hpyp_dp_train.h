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

#define kORDER 3  //default 4
#define nPARTICLES 10

namespace oxlm {

void train_raw(std::string train_file, Dict& dict, std::set<WordId>& vocabs) {
  const WordId kSOS = dict.Convert("ROOT"); 
  std::vector<WordId> ctx(kORDER - 1, kSOS);
  std::vector<Words> corpuss;
  std::vector<WxList> corpusd;

  std::string sentence_file = train_file + ".sentences";
  std::cerr << "Reading corpus...\n";
  ReadFromFile(sentence_file, &dict, &corpuss, &vocabs);
  std::cerr << "E-corpus size: " << corpuss.size() << " sentences\t (" << vocabs.size() << " word types)\n";
  
  std::string dependencies_file = train_file + ".dependencies";
  std::cerr << "Reading corpus...\n";
  ReadFromDependencyFile(dependencies_file, &corpusd);
  std::cerr << "E-corpus size: " << corpusd.size() << " sentences\t (" << " word types)\n";

  //TODO apply the oracle
}

void train_shift(int samples, std::string wc_train_file, MT19937& eng, Dict& dict, std::set<WordId>& vocabs, PYPLM<kORDER>& shift_lm) {
  const WordId kSOS = dict.Convert("ROOT"); 
  std::vector<WordId> ctx(kORDER - 1, kSOS);
  std::vector<Words> corpuss;

  std::cerr << "Reading corpus...\n";
  ReadFromFile(wc_train_file, &dict, &corpuss, &vocabs);
  std::cerr << "E-corpus size: " << corpuss.size() << " sentences\t (" << vocabs.size() << " word types)\n";
  //shift_lm = new PYPLM<kORDER>(vocabs.size(), 1, 1, 1, 1);

  for (int sample=0; sample < samples; ++sample) {
    for (const auto& s : corpuss) {
      WordId w = s[0];
      ctx = std::vector<WordId>(s.begin()+1, s.end());
      if (sample > 0) shift_lm.decrement(w, ctx, eng);
      shift_lm.increment(w, ctx, eng);
    }
    if (sample % 10 == 9) {
      std::cerr << " [LLH=" << shift_lm.log_likelihood() << "]" << std::endl;
      if (sample % 30u == 29) shift_lm.resample_hyperparameters(eng);
    } else { std::cerr << '.' << std::flush; }
  }
}

void train_action(int samples, std::string ac_train_file, MT19937& eng, Dict& dict, std::set<WordId>& vocabr, PYPLM<kORDER>& action_lm) {
  const WordId kSOS = dict.Convert("ROOT"); 
  std::vector<WordId> ctx(kORDER - 1, kSOS);
  std::vector<Words> corpusr;

  std::cerr << "Reading corpus...\n";
  ReadFromActionFile(ac_train_file, &dict, &corpusr, &vocabr);
  std::cerr << "E-corpus size: " << corpusr.size() << " sentences\t (" << vocabr.size() << " word types)\n";

  for (int sample=0; sample < samples; ++sample) {
    for (const auto& s : corpusr) {
      WordId w = s[0];  //index over actions
      ctx = std::vector<WordId>(s.begin()+1, s.end());
      if (sample > 0) action_lm.decrement(w, ctx, eng);
      action_lm.increment(w, ctx, eng);
    }
    if (sample % 10 == 9) {
      std::cerr << " [LLH=" << action_lm.log_likelihood() << "]" << std::endl;
      if (sample % 30u == 29) action_lm.resample_hyperparameters(eng);
    } else { std::cerr << '.' << std::flush; }
  }
}

}
#endif
