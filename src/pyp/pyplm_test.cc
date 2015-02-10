#include <iostream>
#include <cstdlib>
#include <chrono>

#include "corpus/dict.h"
#include "corpus/sentence_corpus.h"
#include "utils/m.h"
#include "utils/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"
#include "pyp/pyplm.h"

#define kORDER 4  //default 4

using namespace oxlm;

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << argv[0] << " <training.txt> <test.txt> <nsamples>\n\nEstimate a " 
         << kORDER << "-gram HPYP LM and report perplexity\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }

  boost::shared_ptr<Dict> dict = boost::make_shared<Dict>();
  MT19937 eng;

  std::string train_file = argv[1];
  std::string test_file = argv[2];
  int samples = atoi(argv[3]);
  
  SentenceCorpus training_corpus;
  std::cerr << "Reading corpus...\n";
  training_corpus.readFile(train_file, dict, false);
  std::cerr << "Training corpus size: " << training_corpus.size() << " sentences\t (" 
            << training_corpus.numTokens() << " tokens " << dict->size() << " word types)\n";
  
  SentenceCorpus test_corpus;
  test_corpus.readFile(test_file, dict, true);
  std::cerr << "Read test corpus\n";
  
  PYPLM<kORDER> lm(dict->size(), 1, 1, 1, 1);

  //training  
  std::cerr << "\nStarting training\n";
  auto tr_start = std::chrono::steady_clock::now();

  std::vector<WordId> ctx(kORDER - 1, dict->sos());
  int train_cnt = 0;
  for (int sample = 0; sample < samples; ++sample) {
    for (int j = 0; j < training_corpus.size(); ++j) {
      ctx.resize(kORDER - 1);
      Sentence s = training_corpus.sentence_at(j);
      for (unsigned i = 0; i < s.size(); ++i) {
        WordId w = s.word_at(i);
        if (sample > 0) 
          lm.decrement(w, ctx, eng);
        lm.increment(w, ctx, eng);
       
        ctx.push_back(w);
        train_cnt++;
      }
    }

    std::cerr << " objective = " << (lm.log_likelihood()/training_corpus.numTokens()) << std::endl;
    std::cerr << train_cnt << " training instances\n";
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

  //Testing
  double llh = 0;
  int cnt = 0;
  for (int j = 0; j < test_corpus.size(); ++j) {
    ctx.resize(kORDER - 1);
    Sentence s = test_corpus.sentence_at(j);
    for (unsigned i = 0; i < s.size(); ++i) {
      WordId w = s.word_at(i);
      double lp = std::log(lm.prob(w, ctx));
      ctx.push_back(w);
      llh -= lp;
      ++cnt;
    }
  }

  double cross_entropy = llh / cnt;
  double perplexity = std::exp(cross_entropy);

  std::cerr << "Number of test tokens: " << cnt << std::endl;
  std::cerr << "Log likelihood: " << llh << std::endl;
  std::cerr << "Cross-entropy: " << cross_entropy << std::endl;
  std::cerr << "Perplexity: " << perplexity << std::endl;

  return 0;
}

