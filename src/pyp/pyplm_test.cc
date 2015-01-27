#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "corpus/dict.h"
#include "corpus/sentence_corpus.h"
#include "utils/m.h"
#include "utils/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"
#include "pyp/pyplm.h"

#define kORDER 4  //default 4

using namespace std;
using namespace oxlm;


int main(int argc, char** argv) {
  if (argc != 4) {
    cerr << argv[0] << " <training.txt> <test.txt> <nsamples>\n\nEstimate a " 
         << kORDER << "-gram HPYP LM and report perplexity\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }

  boost::shared_ptr<Dict> dict = boost::make_shared<Dict>();
  MT19937 eng;

  string train_file = argv[1];
  string test_file = argv[2];
  int samples = atoi(argv[3]);
  
  SentenceCorpus training_corpus;
  cerr << "Reading corpus...\n";
  training_corpus.readFile(train_file, dict, false);
  cerr << "Training corpus size: " << training_corpus.size() << " sentences " << training_corpus.numTokens() << " tokens " << dict->size() << " word types\n";
  
  SentenceCorpus test_corpus;
  test_corpus.readFile(test_file, dict, true);
  cerr << "Read test corpus\n";
  
  PYPLM<kORDER> lm(dict->size(), 1, 1, 1, 1);

  vector<WordId> ctx(kORDER - 1, dict->sos());
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
        
        //std::cout << w << ": "; 
        //for (int k = 0; k < ctx.size(); ++k)
        //for (int k = ctx.size() - (kORDER -1); k < ctx.size(); ++k)
        //  std::cout << ctx.at(k) << " ";
        //std::cout << std::endl;
        
        ctx.push_back(w);
        train_cnt++;
      }
    }

    cerr << " objective = " << (lm.log_likelihood()/training_corpus.numTokens()) << endl;
    cerr << "train count " << train_cnt << endl;
    if (sample % 10 == 9) {
      cerr << " [LLH=" << lm.log_likelihood() << "]" << endl;
      if (sample % 30u == 29) 
        lm.resample_hyperparameters(eng);
    }
  }

  double llh = 0;
  int cnt = 0;
  for (int j = 0; j < test_corpus.size(); ++j) {
    ctx.resize(kORDER - 1);
    Sentence s = test_corpus.sentence_at(j);
    for (unsigned i = 0; i < s.size(); ++i) {
      WordId w = s.word_at(i);
      double lp = log(lm.prob(w, ctx));
      ctx.push_back(w);
      llh -= lp;
      ++cnt;
    }
  }

  double llh2 = llh / log(2);
  cerr << "   Log e prob: " << llh << endl;
  //cerr << "   Log_2 prob: " << llh2 << endl;
  cerr << "        Count: " << cnt << endl;
  cerr << "Cross-entropy: " << (llh2 / cnt) << endl;
  //cerr << "   Perplexity: " << pow(2, llh2 / cnt) << endl;
  cerr << "   Perplexity: " << exp(llh / cnt) << endl;
  return 0;
}


