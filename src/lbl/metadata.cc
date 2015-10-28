#include "lbl/metadata.h"

namespace oxlm {

Metadata::Metadata() {}

Metadata::Metadata(const boost::shared_ptr<ModelConfig>& config,
                   boost::shared_ptr<Dict>& dict)
    : config(config) {}

void Metadata::initialize(const boost::shared_ptr<CorpusInterface>& corpus) {
  VectorReal counts = VectorReal::Zero(config->vocab_size);
  std::vector<int> corpus_counts = corpus->unigramCounts();
  for (size_t i = 0; i < config->vocab_size; ++i) {
    counts(i) = corpus_counts[i];
  }

  unigram = counts.array() / counts.sum();
  smoothed_unigram = (counts.array() + 1) /
                     (counts.sum() + counts.size());
}

VectorReal Metadata::getUnigram() const { return unigram; }

VectorReal Metadata::getSmoothedUnigram() const { return smoothed_unigram; }

bool Metadata::operator==(const Metadata& other) const {
  return *config == *other.config && unigram == other.unigram;
}

}  // namespace oxlm
