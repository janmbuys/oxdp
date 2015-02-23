#pragma once

#include <boost/make_shared.hpp>
#include <boost/thread/tss.hpp>

#include "corpus/parse_data_set.h"

#include "lbl/class_distribution.h"
#include "lbl/parsed_factored_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/word_distributions.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class ParsedFactoredWeights : public FactoredWeights {
 public:
  ParsedFactoredWeights();

  ParsedFactoredWeights(
      const boost::shared_ptr<ModelConfig>& config,
      const boost::shared_ptr<ParsedFactoredMetadata>& metadata,
      bool init);

  ParsedFactoredWeights(const ParsedFactoredWeights& other);

  size_t numParameters() const override;

  void getGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<ParsedFactoredWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  virtual Real getObjective(
      const boost::shared_ptr<ParseDataSet>& examples) const;

  bool checkGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<ParsedFactoredWeights>& gradient,
      double eps);

  void estimateGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<ParsedFactoredWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const; 
  
  Real predictWord(int word, Context context) const;

  Reals predictWord(Context context) const;

  Real predictTag(int tag, Context context) const;
  
  Reals predictTag(Context context) const;

  Real predictAction(int action, Context context) const;
  
  Reals predictAction(Context context) const;

  void syncUpdate(
      const MinibatchWords& words,
      const boost::shared_ptr<ParsedFactoredWeights>& gradient);

  void updateSquared(
      const MinibatchWords& global_words,
      const boost::shared_ptr<ParsedFactoredWeights>& global_gradient);

  void updateAdaGrad(
      const MinibatchWords& global_words,
      const boost::shared_ptr<ParsedFactoredWeights>& global_gradient,
      const boost::shared_ptr<ParsedFactoredWeights>& adagrad);

  Real regularizerUpdate(
      const boost::shared_ptr<ParsedFactoredWeights>& global_gradient,
      Real minibatch_factor);

  void clear(const MinibatchWords& words, bool parallel_update);

  int numWords() const;
  
  int numTags() const;

  int numActions() const;
 
  void clearCache();

  bool operator==(const ParsedFactoredWeights& other) const;

  virtual ~ParsedFactoredWeights();

 protected:
   Real getObjective(
      const boost::shared_ptr<ParseDataSet>& examples,
      vector<vector<int>>& word_contexts,
      vector<vector<int>>& action_contexts,
      vector<WordsList>& word_features,
      vector<WordsList>& action_features,
      vector<MatrixReal>& word_context_vectors,
      vector<MatrixReal>& action_context_vectors,
      MatrixReal& word_prediction_vectors,
      MatrixReal& action_prediction_vectors,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs,
      MatrixReal& action_probs) const;

  void getProbabilities(
      const boost::shared_ptr<ParseDataSet>& examples,
      const vector<vector<int>>& word_contexts,
      const vector<vector<int>>& action_contexts,
      const MatrixReal& word_prediction_vectors,
      const MatrixReal& action_prediction_vectors,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs,
      MatrixReal& action_probs) const;

  MatrixReal getActionWeightedRepresentations(
      const boost::shared_ptr<ParseDataSet>& examples,
      const MatrixReal& action_prediction_vectors,
      const MatrixReal& action_probs) const;

  void getFullGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const vector<vector<int>>& word_contexts,
      const vector<vector<int>>& action_contexts,
      const vector<WordsList>& word_features,
      const vector<WordsList>& action_features,
      const vector<MatrixReal>& word_context_vectors,
      const vector<MatrixReal>& action_context_vectors,
      const MatrixReal& word_prediction_vectors,
      const MatrixReal& action_prediction_vectors,
      const MatrixReal& word_weighted_representations,
      const MatrixReal& action_weighted_representations,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs,
      MatrixReal& action_probs,
      const boost::shared_ptr<ParsedFactoredWeights>& gradient,
      MinibatchWords& words) const;

  std::vector<Words> getNoiseWords(
      const boost::shared_ptr<ParseDataSet>& examples) const; 

  void estimateProjectionGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const MatrixReal& prediction_vectors,
      const boost::shared_ptr<ParsedFactoredWeights>& gradient,
      MatrixReal& weighted_representations,
      Real& objective,
      MinibatchWords& words) const; 

 private:
  void allocate();

  void setModelParameters();

  Block getBlock() const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << metadata;

    ar << boost::serialization::base_object<const FactoredWeights>(*this);

    ar << size;
    ar << boost::serialization::make_array(data, size);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> metadata;

    ar >> boost::serialization::base_object<FactoredWeights>(*this);

    ar >> size;
    data = new Real[size];
    ar >> boost::serialization::make_array(data, size);

    setModelParameters();
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 protected:
  boost::shared_ptr<ParsedFactoredMetadata> metadata;

  WordVectorsType K;
  WeightsType     L;
  WeightsType     PW;
 
  mutable ContextCache actionNormalizerCache;

 private:
  int size;
  Real* data;
  vector<Mutex> mutexes;
};

} //namespace oxlm
