#pragma once

#include <boost/make_shared.hpp>
#include <boost/thread/tss.hpp>

#include "lbl/class_distribution.h"
#include "lbl/factored_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/word_distributions.h"
#include "lbl/word_to_class_index.h"

#include "corpus/parsed_weights_interface.h"
namespace oxlm {

class ParsedFactoredWeights : public FactoredWeights, public ParsedWeightsInterface {
 public:
  ParsedFactoredWeights();

  ParsedFactoredWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<FactoredMetadata>& metadata,
      bool init);

  ParsedFactoredWeights(const ParsedFactoredWeights& other);

  virtual size_t numParameters() const;

  Real predictWord(int word, Words context) const override;

  Real predictTag(int tag, Words context) const override;
  
  Real predictAction(int action, Words context) const override;

  int numWords() const override;
  
  int numTags() const override;

  int numActions() const override;
 
 private:
  void allocate();

  void setModelParameters();

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << metadata;

    ar << boost::serialization::base_object<const FactoredWeights>(*this);

    ar << index;

    ar << size;
    ar << boost::serialization::make_array(data, size);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> metadata;

    ar >> boost::serialization::base_object<FactoredWeights>(*this);

    ar >> index;

    ar >> size;
    data = new Real[size];
    ar >> boost::serialization::make_array(data, size);

    setModelParameters();
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 private:
  int size;
  Real* data;
  vector<Mutex> mutexes;
};

} //namespace oxlm
