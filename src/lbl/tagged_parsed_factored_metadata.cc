#include "lbl/tagged_parsed_factored_metadata.h"

#include <boost/make_shared.hpp>

#include "lbl/model_utils.h"

namespace oxlm {

TaggedParsedFactoredMetadata::TaggedParsedFactoredMetadata() {}

TaggedParsedFactoredMetadata::TaggedParsedFactoredMetadata(
    const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict)
    : ParsedFactoredMetadata(config, dict),
      tagBias(VectorReal::Zero(config->num_tags)) {}

TaggedParsedFactoredMetadata::TaggedParsedFactoredMetadata(
    const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict,
    const boost::shared_ptr<WordToClassIndex>& index)
    : ParsedFactoredMetadata(config, dict, index),
      tagBias(VectorReal::Zero(config->num_tags)) {}

void TaggedParsedFactoredMetadata::initialize(
    const boost::shared_ptr<ParsedCorpus>& corpus) {
  ParsedFactoredMetadata::initialize(corpus);
  vector<int> tag_counts = corpus->tagCounts();

  VectorReal counts = VectorReal::Zero(tag_counts.size());
  for (size_t i = 0; i < tag_counts.size(); ++i) {
    counts(i) = tag_counts[i];
  }
  tagBias = counts.array() / counts.sum();
}

VectorReal TaggedParsedFactoredMetadata::getTagBias() const { return tagBias; }

bool TaggedParsedFactoredMetadata::operator==(
    const TaggedParsedFactoredMetadata& other) const {
  return Metadata::operator==(other) && tagBias == other.tagBias;
}

}  // namespace oxlm
