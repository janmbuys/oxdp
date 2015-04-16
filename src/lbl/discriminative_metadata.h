#pragma once

#include <boost/serialization/singleton.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include "corpus/dict.h"
#include "corpus/parsed_corpus.h"
#include "corpus/model_config.h"

#include "lbl/utils.h"
#include "lbl/model_utils.h"
#include "utils/serialization_helpers.h"

namespace oxlm {

class DiscriminativeMetadata {
 public:
  DiscriminativeMetadata();

  DiscriminativeMetadata(const boost::shared_ptr<ModelConfig>& config, boost::shared_ptr<Dict>& dict);

  void initialize(const boost::shared_ptr<ParsedCorpus>& corpus);

  VectorReal getUnigram() const;

  VectorReal getSmoothedUnigram() const;

  bool operator==(const DiscriminativeMetadata& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & config;
    ar & unigram;
  }

 protected:
  boost::shared_ptr<ModelConfig> config;
  VectorReal unigram;
  VectorReal smoothed_unigram;
};

} // namespace oxlm