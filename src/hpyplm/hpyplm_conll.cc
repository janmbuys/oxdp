#include <iostream>
#include <cstdlib>
#include <chrono>

#include "hpyplm.h"
#include "corpus/corpus.h"
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

using namespace oxlm;

typedef std::vector<WordIndex> WxList;
typedef std::vector<Words> WordsList;

int main(int argc, char** argv) {
  const unsigned kOrder = 4;
  MT19937 eng;

  if (argc != 2) {
      std::cerr << argv[0] << " <nsamples>\n";
    return 1;
  }

  int num_samples = atoi(argv[1]);

  Dict dict("<s>", "</s>");
  const WordId kSOS = dict.convert("<s>", true);
  const WordId kEOS = dict.convert("</s>", true);

  /*Training */
  std::string train_file = "english-wsj/english_wsj_train.conll";

  std::vector<Words> corpus_sents;
  std::vector<Words> corpus_tags;
  std::vector<WxList> corpus_deps;
  
  std::cerr << "Reading training corpus...\n";
  dict.readFromConllFile(train_file, &corpus_sents, &corpus_tags, &corpus_deps, false);
  std::cerr << "Corpus size: " << corpus_sents.size() << " sentences\t (" << dict.size() << " word types, " << dict.tag_size() << " tags)\n";

  //define pyp model
  PYPLM<kOrder> lm(dict.size() + 1, 1, 1, 1, 1);
 
  std::cerr << "\nStarting training\n";
  auto tr_start = std::chrono::steady_clock::now();
                  
  std::vector<WordId> ctx(kOrder - 1, kSOS);
  for (int sample = 0; sample < num_samples; ++sample) {
    
    for (const auto& s: corpus_sents) {
      ctx.resize(kOrder - 1);
      for (unsigned i = 0; i <= s.size(); ++i) {
        WordId w = (i < s.size() ? s[i] : kEOS);
        if (sample > 0) 
          lm.decrement(w, ctx, eng);
        lm.increment(w, ctx, eng);
        ctx.push_back(w);
      }
    }

    if (sample % 10 == 9) {
      std::cerr << (sample + 1) << " iterations\n";
      if (sample % 30u == 29) 
        lm.resample_hyperparameters(eng);
      std::cerr << " [LLH=" << lm.log_likelihood() << "]\n";
    } else { 
      std::cerr << '.'; 
    }
  }

  auto tr_dur = std::chrono::steady_clock::now() - tr_start;
  std::cerr << "Training done...time " << std::chrono::duration_cast<std::chrono::seconds>(tr_dur).count() << "s\n";  
   
 /*Testing */

  std::string test_file = "english-wsj/english_wsj_dev.conll";
  
  std::vector<Words> test_sents;
  std::vector<Words> test_tags;
  std::vector<WxList> test_deps;

  std::cerr << "Reading test corpus...\n";
  dict.readFromConllFile(test_file, &test_sents, &test_tags, &test_deps, true);
  std::cerr << "Corpus size: " << test_sents.size() << " sentences\n";

  double llh = 0;
  int cnt = 0;
  
  for (auto& s : test_sents) {
    ctx.resize(kOrder - 1);
    for (unsigned i = 1; i <= s.size(); ++i) {
      WordId w = (i < s.size() ? s[i] : kEOS);
      double lp = std::log(lm.prob(w, ctx));
      
      ctx.push_back(w);
      llh -= lp;

      if (i < s.size()) {
        cnt++;
      }
    }
  }
 
  double cross_entropy = llh / (std::log(2) * cnt);
  double perplexity = std::pow(2, cross_entropy);

  std::cerr << "Total length: " << cnt << std::endl;
  std::cerr << "Log prob: " << llh << std::endl;
  std::cerr << "Cross-entropy: " << cross_entropy << std::endl;
  std::cerr << "Perplexity: " << perplexity << std::endl;

  return 0;
}


