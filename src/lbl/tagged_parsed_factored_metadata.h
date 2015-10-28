#pragma once

#include <boost/shared_ptr.hpp>

#include "corpus/parsed_corpus.h"

#include "lbl/parsed_factored_metadata.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"
#include "utils/serialization_helpers.h"

namespace oxlm {

// Metadata for TaggedParsedFactoredWeights.  
class TaggedParsedFactoredMetadata : public ParsedFactoredMetadata {
 public:
  TaggedParsedFactoredMetadata();

  TaggedParsedFactoredMetadata(const boost::shared_ptr<ModelConfig>& config,
                               boost::shared_ptr<Dict>& dict);

  TaggedParsedFactoredMetadata(
      const boost::shared_ptr<ModelConfig>& config,
      boost::shared_ptr<Dict>& dict,
      const boost::shared_ptr<WordToClassIndex>& index);

  void initialize(const boost::shared_ptr<ParsedCorpus>& corpus);

  VectorReal getTagBias() const;

  bool operator==(const TaggedParsedFactoredMetadata& other) const;

 private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& boost::serialization::base_object<Metadata>(*this);

    ar& tagBias;
  }

 protected:
  VectorReal tagBias;
};

}  // namespace oxlm
