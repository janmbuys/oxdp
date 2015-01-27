#include <iostream>
#include <cstdlib>
#include <chrono>

#include "pyp/pyplm.h"
#include "corpus/dict.h"
#include "corpus/sentence_corpus.h"
#include "corpus/data_set.h"
#include "utils/m.h"
#include "utils/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

using namespace oxlm;

//typedef std::vector<WordIndex> WxList;
//typedef std::vector<Words> WordsList;

int main(int argc, char** argv) {
  const unsigned kOrder = 4;
  MT19937 eng;

  if (argc != 2) {
      std::cerr << argv[0] << " <nsamples>\n";
    return 1;
  }

  int num_samples = atoi(argv[1]);

  boost::shared_ptr<Dict> dict = boost::make_shared<Dict>("<s>", "</s>");
  const WordId kSOS = dict->convert("<s>", true);
  const WordId kEOS = dict->convert("</s>", true);

  /*Training */
  std::string training_file = "english-wsj-conll08-nofunc-noedges-unk/english_wsj_train.conll.txt";
  
  std::cerr << "Reading training corpus...\n";
  boost::shared_ptr<SentenceCorpus> training_corpus = boost::make_shared<SentenceCorpus>();
  training_corpus->readFile(training_file, dict, false);

  std::cerr << "Corpus size: " << training_corpus->size() << " sentences\t (" 
            << dict_->size() << " word types)\n";  

  //define pyp model
  PYPLM<kOrder> lm(dict.size() + 1, 1, 1, 1, 1);
 
  std::cerr << "\nStarting training\n";
  auto tr_start = std::chrono::steady_clock::now();
                  
  std::vector<WordId> ctx(kOrder - 1, kSOS);
  for (int sample = 0; sample < num_samples; ++sample) {
    int train_cnt = 0;
    for (int k = 0; k < training_corpus->size(); ++k) {
      Sentence s = trianing_corpus->sentence_at(k);
      ctx.resize(kOrder - 1);
      for (unsigned i = 0; i < s.size(); ++i) {
        WordId w = (i < s.size() ? s.word_at(i) : kEOS);
        if (sample > 0) 
          lm.decrement(w, ctx, eng);
        lm.increment(w, ctx, eng);
        ctx.push_back(w);
        train_cnt++;
      }
    }

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
   
 /*Testing */

  std::string test_file = "english-wsj/english_wsj_dev.conll";
  boost::shared_ptr<SentenceCorpus> test_corpus = boost::make_shared<SentenceCorpus>();
  std::cerr << "Reading test corpus...\n";
  test_corpus->readFile(test_file, dict, true);
  std::cerr << "Corpus size: " << training_corpus->size() << " sentences\t (" 
            << dict_->size() << " word types)\n";  

  double llh = 0;
  int cnt = 0;
  
  for (int k = 0; k < training_corpus->size(); ++k) {
    Sentence s = trianing_corpus->sentence_at(k);
    ctx.resize(kOrder - 1);
    for (unsigned i = 1; i <= s.size(); ++i) {
      WordId w = (i < s.size() ? s.word_at(i) : kEOS);
      double lp = std::log(lm.prob(w, ctx));
      
      ctx.push_back(w);
      llh -= lp;
      //if (i < s.size()) 
      cnt++;
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


