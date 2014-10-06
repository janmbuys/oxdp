#pragma once

#include <boost/make_shared.hpp>
#include <boost/thread/tss.hpp>

#include "corpus/parse_data_set.h"

#include "lbl/class_distribution.h"
#include "lbl/factored_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/word_distributions.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class ParsedFactoredWeights : public FactoredWeights {
 public:
  ParsedFactoredWeights();

  ParsedFactoredWeights(
      const boost::shared_ptr<ModelConfig>& config,
      const boost::shared_ptr<FactoredMetadata>& metadata,
      bool init);

  ParsedFactoredWeights(const ParsedFactoredWeights& other);

  size_t numParameters() const override;

  void getGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<FactoredWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  virtual Real getObjective(
      const boost::shared_ptr<ParseDataSet>& examples) const;

  bool checkGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<FactoredWeights>& gradient,
      double eps);

  void estimateGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<FactoredWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;
  
  Real predictWord(int word, Words context) const;

  Real predictTag(int tag, Words context) const;
  
  Real predictAction(int action, Words context) const;

  int numWords() const;
  
  int numTags() const;

  int numActions() const;
 
 protected:
 Real getObjective(
      const boost::shared_ptr<ParseDataSet>& examples,
      vector<vector<int>>& contexts,
      vector<MatrixReal>& context_vectors,
      MatrixReal& prediction_vectors,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs) const;

  void getProbabilities(
    const boost::shared_ptr<ParseDataSet>& examples,
      const vector<vector<int>>& contexts,
      const MatrixReal& prediction_vectors,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs) const;

  MatrixReal getWeightedRepresentations(
      const boost::shared_ptr<ParseDataSet>& examples,
      const MatrixReal& prediction_vectors,
      const MatrixReal& class_probs,
      const vector<VectorReal>& word_probs) const;

  void getFullGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& prediction_vectors,
      const MatrixReal& weighted_representations,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs,
      const boost::shared_ptr<FactoredWeights>& gradient,
      MinibatchWords& words) const;

  std::vector<Words> getNoiseWords(
      const boost::shared_ptr<ParseDataSet>& examples) const;

  void estimateProjectionGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const MatrixReal& prediction_vectors,
      const boost::shared_ptr<FactoredWeights>& gradient,
      MatrixReal& weighted_representations,
      Real& objective,
      MinibatchWords& words) const;

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
    //ar << boost::serialization::make_array(data, size);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> metadata;

    ar >> boost::serialization::base_object<FactoredWeights>(*this);

    ar >> index;

    ar >> size;
    //data = new Real[size];
    //ar >> boost::serialization::make_array(data, size);

    setModelParameters();
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 private:
  int size;
  //Real* data;
  vector<Mutex> mutexes;
};

} //namespace oxlm
