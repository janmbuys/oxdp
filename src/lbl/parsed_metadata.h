#pragma once

#include <boost/serialization/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include "corpus/parsed_corpus.h"

#include "lbl/metadata.h"
#include "lbl/utils.h"
#include "lbl/word_to_class_index.h"
#include "utils/serialization_helpers.h"

namespace oxlm {

class ParsedMetadata : public Metadata {
 public:
  ParsedMetadata();

  ParsedMetadata(const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict);

  void initialize(const boost::shared_ptr<ParsedCorpus>& corpus);

  VectorReal getActionBias() const;

  bool operator==(const ParsedMetadata& other) const;

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
