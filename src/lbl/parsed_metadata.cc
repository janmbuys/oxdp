#include "lbl/parsed_metadata.h"

#include <boost/make_shared.hpp>

#include "lbl/model_utils.h"

namespace oxlm {

ParsedMetadata::ParsedMetadata() {}

ParsedMetadata::ParsedMetadata(
    const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict)
    : Metadata(config, dict),
      actionBias(VectorReal::Zero(config->numActions())) {}

void ParsedMetadata::initialize(const boost::shared_ptr<ParsedCorpus>& corpus) {
  Metadata::initialize(corpus);  
  vector<int> action_counts = corpus->actionCounts();
    
  VectorReal counts = VectorReal::Zero(action_counts.size());
  for (size_t i = 0; i < action_counts.size(); ++i) 
    counts(i) = action_counts[i];
  actionBias = counts.array() / counts.sum();
}

VectorReal ParsedMetadata::getActionBias() const {
  return actionBias;
}

bool ParsedMetadata::operator==(const ParsedMetadata& other) const {
  return Metadata::operator==(other)
      && actionBias == other.actionBias;
}

} // namespace oxlm
