#ifndef _GDP_PYP_MODEL_H_
#define _GDP_PYP_MODEL_H_

#include <numeric>
#include <algorithm>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "corpus/dict.h"
#include "corpus/parsed_corpus.h"
#include "corpus/parse_data_set.h"
#include "corpus/model_config.h"

#include "pyp/constants.h"
#include "pyp/parsed_pyp_weights.h"
#include "pyp/parsed_lex_pyp_weights.h"
#include "pyp/parsed_chlex_pyp_weights.h"

#include "gdp/transition_parser.h"
//#include "gdp/arc_standard_parse_model.h"
#include "gdp/arc_standard_labelled_parse_model.h"
//#include "gdp/arc_eager_parse_model.h"
#include "gdp/arc_eager_labelled_parse_model.h"
#include "gdp/eisner_parse_model.h"
#include "gdp/accuracy_counts.h"

namespace oxlm {

template<class ParseModel, class ParsedWeights>
class PypDpModel {
 public:
  PypDpModel();

  PypDpModel(const boost::shared_ptr<ModelConfig>& config);

  void learn();

  void evaluate() const;

  void evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, int minibatch_counter, 
                   Real& log_likelihood, Real& best_perplexity) const;

  void evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, Real& accumulator) const;

 private:
  boost::shared_ptr<ModelConfig> config_;
  boost::shared_ptr<Dict> dict_;
  boost::shared_ptr<Dict> ch_dict_;
  boost::shared_ptr<ParsedWeights> weights_;
  boost::shared_ptr<ParseModel> parse_model_;
};

}

#endif

