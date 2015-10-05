#pragma once

#include <boost/shared_ptr.hpp>

#include "corpus/dict.h"
#include "corpus/sentence_corpus.h"
#include "corpus/data_set.h"
#include "corpus/model_config.h"

#include "lbl/factored_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/minibatch_words.h"
#include "lbl/model_utils.h"
#include "lbl/parsed_factored_weights.h"
#include "lbl/parsed_weights.h"
#include "lbl/discriminative_weights.h"
#include "lbl/utils.h"
#include "lbl/weights.h"

#include "corpus/dict.h"
#include "corpus/parsed_corpus.h"
#include "corpus/parse_data_set.h"
#include "corpus/model_config.h"

#include "pyp/constants.h"
#include "pyp/parsed_pyp_weights.h"
#include "pyp/parsed_lex_pyp_weights.h"

#include "gdp/transition_parser.h"
#include "gdp/arc_standard_labelled_parse_model.h"
#include "gdp/accuracy_counts.h"

namespace oxlm {

template<class ParseModel, class ParsedWeights, class Metadata>
class LblDpModel {
 public:
  LblDpModel();

  LblDpModel(const boost::shared_ptr<ModelConfig>& config);

  boost::shared_ptr<Dict> getDict() const;

  boost::shared_ptr<ModelConfig> getConfig() const;

  void learn();

  void update(
      const MinibatchWords& global_words,
      const boost::shared_ptr<ParsedWeights>& global_gradient,
      const boost::shared_ptr<ParsedWeights>& adagrad,
      bool sentences_only);

  Real regularize(
      const MinibatchWords& global_words,
      const boost::shared_ptr<ParsedWeights>& global_gradient,
      Real minibatch_factor,
      bool sentences_only);

  void evaluate();

  void evaluate(
      const boost::shared_ptr<ParsedCorpus>& corpus, Real& accumulator);

  void evaluate_sentence_vector(
      const boost::shared_ptr<ParsedCorpus>& corpus, Real& accumulator);

  Real predict(int word_id, const vector<int>& context) const;

  MatrixReal getWordVectors() const;

  MatrixReal getFeatureVectors() const;
  
  MatrixReal getSentenceVectors() const;

  void save() const;

  void load(const string& filename);

  void clearCache();

  bool operator==(
      const LblDpModel<ParseModel, ParsedWeights, Metadata>& other) const;

  void evaluate(
      const boost::shared_ptr<ParsedCorpus>& corpus, const Time& iteration_start,
      int minibatch_counter, Real& objective, Real& best_perplexity);

 private:
  boost::shared_ptr<ModelConfig> config;
  boost::shared_ptr<Dict> dict;
  boost::shared_ptr<Metadata> metadata;
  boost::shared_ptr<ParsedWeights> weights;
  boost::shared_ptr<ParseModel> parse_model;
};

/*
class ASParsedFactoredLblLM: public LblDpModel<ArcStandardParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata> {
 public:
  ASParsedFactoredLblLM() : LblDpModel<ArcStandardParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>() {}

  ASParsedFactoredLblLM(const boost::shared_ptr<ModelConfig>& config)
     : LblDpModel<ArcStandardParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>(config) {}
};

class ASParsedLblLM: public LblDpModel<ArcStandardParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata> {
 public:
  ASParsedLblLM() : LblDpModel<ArcStandardParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>() {}

  ASParsedLblLM(const boost::shared_ptr<ModelConfig>& config)
     : LblDpModel<ArcStandardParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>(config) {}
}; */ 

} // namespace oxlm
