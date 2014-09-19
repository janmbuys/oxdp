#ifndef _PYP_MODEL_H_
#define _PYP_MODEL_H_

#include <numeric>
#include <algorithm>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "corpus/dict.h"
#include "corpus/sentence_corpus.h"
#include "corpus/data_set.h"
#include "corpus/ngram_model.h"

#include "pyp/utils.h"
#include "pyp/pyp_weights_interface.h"
#include "pyp/model_config.h"
#include "pyp/pyp_weights.h"

namespace oxlm {

//identity more abstractions later
class PypModel {
  public:
  PypModel();

  PypModel(const boost::shared_ptr<ModelConfig>& config);

  void learn();

  void evaluate() const;

  void evaluate(const boost::shared_ptr<SentenceCorpus>& test_corpus, int minibatch_counter, 
                   double& log_likelihood, double& best_perplexity) const;

  void evaluate(const boost::shared_ptr<SentenceCorpus>& test_corpus, double& accumulator) const;

  private:
  boost::shared_ptr<ModelConfig> config_;
  boost::shared_ptr<Dict> dict_;
  boost::shared_ptr<PypWeightsInterface> weights_;
  boost::shared_ptr<NGramModel> model_;
};

}

#endif

