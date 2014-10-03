#pragma once

#include <boost/shared_ptr.hpp>

#include "corpus/dict.h"
#include "corpus/sentence_corpus.h"
#include "corpus/data_set.h"
#include "corpus/model_config.h"

#include "lbl/factored_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/metadata.h"
#include "lbl/minibatch_words.h"
#include "lbl/model_utils.h"
#include "lbl/utils.h"
#include "lbl/weights.h"

#include "gdp/ngram_model.h"

namespace oxlm {

enum ModelType {
  NLM = 1,
  FACTORED_NLM = 2,
};

template<class GlobalWeights, class MinibatchWeights, class Metadata>
class LblModel {
 public:
  LblModel();

  LblModel(const boost::shared_ptr<ModelConfig>& config);

  boost::shared_ptr<Dict> getDict() const;

  boost::shared_ptr<ModelConfig> getConfig() const;

  void learn();

  void update(
      const MinibatchWords& global_words,
      const boost::shared_ptr<MinibatchWeights>& global_gradient,
      const boost::shared_ptr<GlobalWeights>& adagrad);

  Real regularize(
      const boost::shared_ptr<MinibatchWeights>& global_gradient,
      Real minibatch_factor);

  void evaluate(
      const boost::shared_ptr<SentenceCorpus>& corpus, Real& accumulator) const;

  Real predict(int word_id, const vector<int>& context) const;

  MatrixReal getWordVectors() const;

  void save() const;

  void load(const string& filename);

  void clearCache();

  bool operator==(
      const LblModel<GlobalWeights, MinibatchWeights, Metadata>& other) const;

 private:
  void evaluate(
      const boost::shared_ptr<SentenceCorpus>& corpus, const Time& iteration_start,
      int minibatch_counter, Real& objective, Real& best_perplexity) const;

  boost::shared_ptr<ModelConfig> config;
  boost::shared_ptr<Dict> dict;
  boost::shared_ptr<Metadata> metadata;
  boost::shared_ptr<GlobalWeights> weights;
  boost::shared_ptr<NGramModel<GlobalWeights>> ngram_model;
};

class LblLM: public LblModel<Weights, Weights, Metadata> {
 public:
  LblLM() : LblModel<Weights, Weights, Metadata>() {}

  LblLM(const boost::shared_ptr<ModelConfig>& config)
      : LblModel<Weights, Weights, Metadata>(config) {}
};

class FactoredLblLM: public LblModel<FactoredWeights, FactoredWeights, FactoredMetadata> {
 public:
  FactoredLblLM() : LblModel<FactoredWeights, FactoredWeights, FactoredMetadata>() {}

  FactoredLblLM(const boost::shared_ptr<ModelConfig>& config)
      : LblModel<FactoredWeights, FactoredWeights, FactoredMetadata>(config) {}
};

} // namespace oxlm
