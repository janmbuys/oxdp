#include "lbl/parsed_factored_metadata.h"

#include <boost/make_shared.hpp>

#include "lbl/model_utils.h"

namespace oxlm {

ParsedFactoredMetadata::ParsedFactoredMetadata() {}

ParsedFactoredMetadata::ParsedFactoredMetadata(
    const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict)
    : FactoredMetadata(config, dict),
      actionBias(VectorReal::Zero(config->numActions())) {}

ParsedFactoredMetadata::ParsedFactoredMetadata(
    const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict,
    const boost::shared_ptr<WordToClassIndex>& index)
    : FactoredMetadata(config, dict, index),
      actionBias(VectorReal::Zero(config->numActions())) {}

void ParsedFactoredMetadata::initialize(
    const boost::shared_ptr<ParsedCorpus>& corpus) {
  FactoredMetadata::initialize(corpus);
  vector<int> action_counts = corpus->actionCounts();

  // Initialize action biases with unigram distribution. Alternatively,
  // could initialize uniformly. TODO confirm which works best.
   actionBias = VectorReal::Ones(action_counts.size()) * 0.2;
  /*VectorReal counts = VectorReal::Zero(action_counts.size());
  for (size_t i = 0; i < action_counts.size(); ++i) {
    counts(i) = action_counts[i];
  }
  actionBias = counts.array() / counts.sum(); */
}

VectorReal ParsedFactoredMetadata::getActionBias() const { return actionBias; }

bool ParsedFactoredMetadata::operator==(
    const ParsedFactoredMetadata& other) const {
  return Metadata::operator==(other) && actionBias == other.actionBias;
}

}  // namespace oxlm
