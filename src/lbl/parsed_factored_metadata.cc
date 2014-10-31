#include "lbl/parsed_factored_metadata.h"

#include <boost/make_shared.hpp>

#include "lbl/model_utils.h"

namespace oxlm {

ParsedFactoredMetadata::ParsedFactoredMetadata() {}

ParsedFactoredMetadata::ParsedFactoredMetadata(
    const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict)
    : FactoredMetadata(config, dict) {}

ParsedFactoredMetadata::ParsedFactoredMetadata(
    const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict,
    const boost::shared_ptr<WordToClassIndex>& index)
    : FactoredMetadata(config, dict, index),
      actionBias(VectorReal::Zero(config->num_actions)) {}

void ParsedFactoredMetadata::initialize(const boost::shared_ptr<ParsedCorpus>& corpus) {
  FactoredMetadata::initialize(corpus);  
  vector<int> action_counts = corpus->actionCounts();
  //std::cout << action_counts.size() << " actions" << std::endl;
  if (config->labelled_parser) {
    VectorReal counts = VectorReal::Zero(action_counts.size());
    for (size_t i = 0; i < action_counts.size(); ++i) 
      counts(i) = action_counts[i];
    actionBias = counts.array() / counts.sum();
  } else {
    //assume for now arcstandard
    VectorReal counts = VectorReal::Zero(config->num_actions);
    counts(0) = action_counts[0];
    for (int k = 0; k < config->num_labels; ++k) {
      counts(1) += action_counts[k + 1];
      counts(2) += action_counts[k + config->num_labels + 1];
    }
    actionBias = counts.array() / counts.sum();
  }
}

VectorReal ParsedFactoredMetadata::getActionBias() const {
  return actionBias;
}

bool ParsedFactoredMetadata::operator==(const ParsedFactoredMetadata& other) const {
  return Metadata::operator==(other)
      && actionBias == other.actionBias;
}

} // namespace oxlm
