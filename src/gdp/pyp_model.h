#ifndef _PYP_MODEL_H_
#define _PYP_MODEL_H_

#include <numeric>
#include <algorithm>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "corpus/dict.h"
#include "corpus/sentence_corpus.h"
#include "corpus/data_set.h"
#include "corpus/model_config.h"

#include "pyp/constants.h"
#include "pyp/pyp_weights.h"

#include "gdp/ngram_model.h"

namespace oxlm {

// Train and evaluate a PYP n-gram language model.  
class PypModel {
 public:
  PypModel();

  PypModel(const boost::shared_ptr<ModelConfig>& config);

  void learn();

  void evaluate() const;

  void evaluate(const boost::shared_ptr<SentenceCorpus>& test_corpus,
                int minibatch_counter, Real& log_likelihood,
                Real& best_perplexity) const;

  void evaluate(const boost::shared_ptr<SentenceCorpus>& test_corpus,
                Real& accumulator) const;

 private:
  boost::shared_ptr<ModelConfig> config_;
  boost::shared_ptr<Dict> dict_;
  boost::shared_ptr<PypWeights<wordLMOrder>> weights_;
  boost::shared_ptr<NGramModel<PypWeights<wordLMOrder>>> model_;
};

}  // naemspace oxlm

#endif

