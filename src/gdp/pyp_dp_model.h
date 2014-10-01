#ifndef _GDP_PYP_MODEL_H_
#define _GDP_PYP_MODEL_H_

#include <numeric>
#include <algorithm>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "corpus/dict.h"
#include "corpus/parsed_corpus.h"
#include "corpus/parse_data_set.h"

#include "pyp/utils.h"
#include "pyp/pyp_parsed_weights_interface.h"
#include "pyp/model_config.h"
#include "pyp/parsed_pyp_weights.h"
#include "pyp/parsed_lex_pyp_weights.h"

#include "gdp/transition_parser.h"
#include "gdp/transition_parse_model_interface.h"
#include "gdp/arc_standard_parse_model.h"
#include "gdp/arc_eager_parse_model.h"
#include "gdp/eisner_parse_model.h"
#include "gdp/accuracy_counts.h"

namespace oxlm {

//identity more abstractions later
class PypDpModel {
  public:
  PypDpModel();

  PypDpModel(const boost::shared_ptr<ModelConfig>& config);

  void learn_semi_supervised();

  void learn();

  void evaluate() const;

  void evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, int minibatch_counter, 
                   Real& log_likelihood, Real& best_perplexity) const;

  void evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, Real& accumulator) const;

  private:
  boost::shared_ptr<ModelConfig> config_;
  boost::shared_ptr<Dict> dict_;
  boost::shared_ptr<PypParsedWeightsInterface> weights_;
  boost::shared_ptr<ParseModelInterface> parse_model_;
};

}

#endif

