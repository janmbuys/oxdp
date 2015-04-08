#include "lbl/discriminative_metadata.h"

namespace oxlm {

DiscriminativeMetadata::DiscriminativeMetadata() {}

DiscriminativeMetadata::DiscriminativeMetadata(const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict): 
    config(config) {}

void DiscriminativeMetadata::initialize(const boost::shared_ptr<ParsedCorpus>& corpus) {
  vector<int> action_counts = corpus->actionCounts();
  VectorReal counts = VectorReal::Zero(config->numActions());
  for (size_t i = 0; i < config->numActions(); ++i) 
    counts(i) = action_counts[i];

  unigram = counts.array() / counts.sum();
  smoothed_unigram = (counts.array() + 1) / (counts.sum() + counts.size()); //plus one smoothing
}

VectorReal DiscriminativeMetadata::getUnigram() const {
  return unigram;
}

VectorReal DiscriminativeMetadata::getSmoothedUnigram() const {
  return smoothed_unigram;
}

bool DiscriminativeMetadata::operator==(const DiscriminativeMetadata& other) const {
  return *config == *other.config
      && unigram == other.unigram;
}

} // namespace oxlm
