#pragma once

#include <boost/serialization/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include "corpus/parsed_corpus.h"

#include "lbl/factored_metadata.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"
#include "utils/serialization_helpers.h"

namespace oxlm {

class ParsedFactoredMetadata : public FactoredMetadata {
 public:
  ParsedFactoredMetadata();

  ParsedFactoredMetadata(const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict);

  ParsedFactoredMetadata(
      const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict,
      const boost::shared_ptr<WordToClassIndex>& index);

  void initialize(const boost::shared_ptr<ParsedCorpus>& corpus);

  VectorReal getActionBias() const;

  bool operator==(const ParsedFactoredMetadata& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<Metadata>(*this);

    ar & actionBias;
  }

 protected:
  VectorReal actionBias;
};

} // namespace oxlm
