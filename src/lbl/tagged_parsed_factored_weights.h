#pragma once

#include <boost/make_shared.hpp>
#include <boost/thread/tss.hpp>

#include "corpus/parse_data_set.h"

#include "lbl/class_distribution.h"
#include "lbl/tagged_parsed_factored_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/parsed_factored_weights.h"
#include "lbl/word_distributions.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class TaggedParsedFactoredWeights : public ParsedFactoredWeights {
 public:
  TaggedParsedFactoredWeights();

  TaggedParsedFactoredWeights(
      const boost::shared_ptr<ModelConfig>& config,
      const boost::shared_ptr<TaggedParsedFactoredMetadata>& metadata,
      bool init);

  TaggedParsedFactoredWeights(const TaggedParsedFactoredWeights& other);

  size_t numParameters() const override;

  VectorReal getSentenceVectorGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      Real& objective) const;

  void getGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<TaggedParsedFactoredWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  virtual Real getObjective(
      const boost::shared_ptr<ParseDataSet>& examples) const;

  bool checkGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<TaggedParsedFactoredWeights>& gradient,
      double eps);

  void estimateGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<TaggedParsedFactoredWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const; 
  
  //Real predictWord(int word, Context context) const;

 // Reals predictWord(Context context) const;

  Reals predictWordOverTags(int word, Context context) const;

  Real predictTag(int tag, Context context) const;
  
  Reals predictTag(Context context) const;

  //Real predictAction(int action, Context context) const;
  
  //Reals predictAction(Context context) const;

  void syncUpdate(
      const MinibatchWords& words,
      const boost::shared_ptr<TaggedParsedFactoredWeights>& gradient);

  void updateSquared(
      const MinibatchWords& global_words,
      const boost::shared_ptr<TaggedParsedFactoredWeights>& global_gradient);

  void updateAdaGrad(
      const MinibatchWords& global_words,
      const boost::shared_ptr<TaggedParsedFactoredWeights>& global_gradient,
      const boost::shared_ptr<TaggedParsedFactoredWeights>& adagrad);

  Real regularizerUpdate(
      const boost::shared_ptr<TaggedParsedFactoredWeights>& global_gradient,
      Real minibatch_factor);

  void clear(const MinibatchWords& words, bool parallel_update);

  int numWords() const;
  
  int numTags() const;

  int numActions() const;
 
  void clearCache();

  bool operator==(const TaggedParsedFactoredWeights& other) const;

  virtual ~TaggedParsedFactoredWeights();

 protected:
   Real getObjective(
      const boost::shared_ptr<ParseDataSet>& examples,
      vector<WordsList>& word_contexts,
      vector<WordsList>& action_contexts,
      vector<WordsList>& tag_contexts,
      vector<MatrixReal>& word_context_vectors,
      vector<MatrixReal>& action_context_vectors,
      vector<MatrixReal>& tag_context_vectors,
      MatrixReal& word_prediction_vectors,
      MatrixReal& action_prediction_vectors,
      MatrixReal& tag_prediction_vectors,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs,
      MatrixReal& action_probs,
      MatrixReal& tag_probs) const;

  void getProbabilities(
      const boost::shared_ptr<ParseDataSet>& examples,
      const vector<WordsList>& word_contexts,
      const vector<WordsList>& action_contexts,
      const vector<WordsList>& tag_contexts,
      const MatrixReal& word_prediction_vectors,
      const MatrixReal& action_prediction_vectors,
      const MatrixReal& tag_prediction_vectors,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs,
      MatrixReal& action_probs,
      MatrixReal& tag_probs) const;

  MatrixReal getTagWeightedRepresentations(
      const boost::shared_ptr<ParseDataSet>& examples,
      const MatrixReal& tag_prediction_vectors,
      const MatrixReal& tag_probs) const;

  void getFullGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const vector<WordsList>& word_contexts,
      const vector<WordsList>& action_contexts,
      const vector<WordsList>& tag_contexts,
      const vector<MatrixReal>& word_context_vectors,
      const vector<MatrixReal>& action_context_vectors,
      const vector<MatrixReal>& tag_context_vectors,
      const MatrixReal& word_prediction_vectors,
      const MatrixReal& action_prediction_vectors,
      const MatrixReal& tag_prediction_vectors,
      const MatrixReal& word_weighted_representations,
      const MatrixReal& action_weighted_representations,
      const MatrixReal& tag_weighted_representations,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs,
      MatrixReal& action_probs,
      MatrixReal& tag_probs,
      const boost::shared_ptr<TaggedParsedFactoredWeights>& gradient,
      MinibatchWords& words) const;

 private:
  void allocate();

  void setModelParameters();

  Block getBlock() const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << metadata;

    ar << boost::serialization::base_object<const ParsedFactoredWeights>(*this);

    ar << size;
    ar << boost::serialization::make_array(data, size);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> metadata;

    ar >> boost::serialization::base_object<ParsedFactoredWeights>(*this);

    ar >> size;
    data = new Real[size];
    ar >> boost::serialization::make_array(data, size);

    setModelParameters();
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 protected:
  boost::shared_ptr<TaggedParsedFactoredMetadata> metadata;

  WordVectorsType U;
  WeightsType     V;
  WeightsType     TW;
 
  mutable ContextCache tagNormalizerCache;

 private:
  int size;
  Real* data;
  vector<Mutex> mutexes;
};

} //namespace oxlm
