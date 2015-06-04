#pragma once

#include <mutex>
#include <vector>

#include <boost/make_shared.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/thread/tss.hpp>

#include "corpus/parse_data_set.h"

#include "lbl/context_cache.h"
#include "lbl/discriminative_metadata.h"
#include "lbl/minibatch_words.h"
#include "lbl/utils.h"
#include "lbl/word_distributions.h"

namespace oxlm {

typedef Eigen::Map<MatrixReal> ContextTransformType;
typedef vector<ContextTransformType> ContextTransformsType;
typedef Eigen::Map<MatrixReal> WordVectorsType;
typedef Eigen::Map<VectorReal> WeightsType;

typedef boost::shared_ptr<mutex> Mutex;
typedef pair<size_t, size_t> Block;

class DiscriminativeWeights {
 public:
  DiscriminativeWeights();

  DiscriminativeWeights(
      const boost::shared_ptr<ModelConfig>& config,
      const boost::shared_ptr<DiscriminativeMetadata>& metadata,
      bool init);

  DiscriminativeWeights(const DiscriminativeWeights& other);

  virtual size_t numParameters() const;

  void getGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<DiscriminativeWeights>& gradient,
      Real& objective,
      MinibatchWords& words,
      bool sentences_only = false) const;

  virtual Real getObjective(
      const boost::shared_ptr<ParseDataSet>& examples) const;

  bool checkGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<DiscriminativeWeights>& gradient,
      double eps);

  void estimateGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const boost::shared_ptr<DiscriminativeWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;


  VectorReal getSentenceVectorGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      Real& objective) const;

  Real predictWord(int word, Context context) const;

  Reals predictWordOverTags(WordId word, Context context) const;

  Reals predictWord(Context context) const;

  Real predictTag(int tag, Context context) const;
  
  Reals predictTag(Context context) const;

  Real predictAction(int action, Context context) const;
  
  Reals predictAction(Context context) const;

  int numWords() const;
  
  int numTags() const;

  int numActions() const;

  void syncUpdate(
      const MinibatchWords& words,
      const boost::shared_ptr<DiscriminativeWeights>& gradient,
      bool sentences_only = false);

  void updateSentenceVectorGradient(const VectorReal& sentence_vector_gradient);

  void updateSquared(
      const MinibatchWords& global_words,
      const boost::shared_ptr<DiscriminativeWeights>& global_gradient,
      bool sentences_only = false);

  void updateAdaGrad(
      const MinibatchWords& global_words,
      const boost::shared_ptr<DiscriminativeWeights>& global_gradient,
      const boost::shared_ptr<DiscriminativeWeights>& adagrad,
      bool sentences_only = false);

  Real regularizerUpdate(
      const MinibatchWords& global_words,
      const boost::shared_ptr<DiscriminativeWeights>& global_gradient,
      Real minibatch_factor,
      bool sentences_only = false);

  void clear(const MinibatchWords& words, bool parallel_update);

  void resetSentenceVector();

  Real predict(int word, Context context) const;

  Reals predict(Context context) const;

  int vocabSize() const;

  void clearCache();

  MatrixReal getWordVectors() const;

  MatrixReal getFeatureVectors() const;

  MatrixReal getSentenceVectors() const;

  bool operator==(const DiscriminativeWeights& other) const;

  virtual ~DiscriminativeWeights();

 protected:

  Real getObjective(
      const boost::shared_ptr<ParseDataSet>& examples,
      vector<WordsList>& contexts,
      vector<MatrixReal>& context_vectors,
      MatrixReal& prediction_vectors,
      MatrixReal& word_probs) const;

  void getContextVectors(
      const boost::shared_ptr<ParseDataSet>& examples,
      vector<WordsList>& contexts,
      vector<MatrixReal>& context_vectors) const;
  
  void setContextWords(
      const vector<WordsList>& contexts,
      MinibatchWords& words) const;

  MatrixReal getPredictionVectors(
      size_t prediction_size,
      const vector<MatrixReal>& context_vectors) const;

  MatrixReal getContextProduct(
      int index, const MatrixReal& representations,
      bool transpose = false) const;

  MatrixReal getProbabilities(
      const boost::shared_ptr<ParseDataSet>& examples,
      const MatrixReal& prediction_vectors) const;

  MatrixReal getWeightedRepresentations(
      const boost::shared_ptr<ParseDataSet>& examples,
      const MatrixReal& prediction_vectors,
      const MatrixReal& word_probs) const;

  void getFullGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const vector<WordsList>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& prediction_vectors,
      const MatrixReal& weighted_representations,
      MatrixReal& word_probs,
      const boost::shared_ptr<DiscriminativeWeights>& gradient,
      MinibatchWords& words) const;

  void getContextGradient(
      size_t prediction_size,
      const vector<WordsList>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& weighted_representations,
      const boost::shared_ptr<DiscriminativeWeights>& gradient) const;

  virtual vector<vector<int>> getNoiseWords(
      const boost::shared_ptr<ParseDataSet>& examples) const;

  void estimateProjectionGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      const MatrixReal& prediction_vectors,
      const boost::shared_ptr<DiscriminativeWeights>& gradient,
      MatrixReal& weighted_representations,
      Real& objective,
      MinibatchWords& words) const;

  VectorReal getPredictionVector(const Context& context) const;

 private:
  void allocate();

  void setModelParameters();

  Block getBlock(int start, int size) const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << config;
    ar << metadata;

    ar << size;
    ar << boost::serialization::make_array(data, size);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> config;

    ar >> metadata;

    ar >> size;
    data = new Real[size];
    ar >> boost::serialization::make_array(data, size);

    setModelParameters();
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 protected:
  boost::shared_ptr<ModelConfig> config;
  boost::shared_ptr<DiscriminativeMetadata> metadata;

  ContextTransformsType C;
  WordVectorsType       P;
  WordVectorsType       Q;
  WordVectorsType       R;
  WeightsType           B;
  WeightsType           W;

  mutable ContextCache normalizerCache;

 private:
  int size;
  Real* data;
  vector<Mutex> mutexesC;
  vector<Mutex> mutexesP;
  vector<Mutex> mutexesQ;
  vector<Mutex> mutexesR;
  Mutex mutexB;

  mutable boost::thread_specific_ptr<WordDistributions> wordDists;
};

} // namespace oxlm
