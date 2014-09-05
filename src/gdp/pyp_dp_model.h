#ifndef _GDP_PYP_MODEL_H_
#define _GDP_PYP_MODEL_H_

#include <numeric>
#include <algorithm>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "pyp/pyp_parsed_weights_interface.h"
#include "gdp/model_config.h"
#include "corpus/dict.h"
#include "corpus/parsed_corpus.h"
#include "pyp/parsed_pyp_weights.h"
#include "pyp/parsed_lex_pyp_weights.h"

namespace oxlm {

//identity more abstractions later
class PypDpModel {
  public:
  PypDpModel();

  PypDpModel(const boost::shared_ptr<ModelConfig>& config);

  void learn();

  void evaluate() const;

  void evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, int minibatch_counter, 
                   double& log_likelihood, double& best_perplexity) const;

  void evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, double& accumulator) const;

  private:
  boost::shared_ptr<ModelConfig> config_;
  boost::shared_ptr<Dict> dict_;
  boost::shared_ptr<PypParsedWeightsInterface> weights_; //see if this works
  size_t num_actions_;
};

}

#endif

