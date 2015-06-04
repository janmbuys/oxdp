#include "lbl/tagged_parsed_factored_weights.h"

#include <iomanip>

#include <boost/make_shared.hpp>

#include "lbl/operators.h"

namespace oxlm {

TaggedParsedFactoredWeights::TaggedParsedFactoredWeights()
    : data(NULL), U(0, 0, 0), V(0, 0), TW(0, 0) {}

TaggedParsedFactoredWeights::TaggedParsedFactoredWeights(
    const boost::shared_ptr<ModelConfig>& config,
    const boost::shared_ptr<TaggedParsedFactoredMetadata>& metadata,
    bool init)
    : ParsedFactoredWeights(config, metadata, init), metadata(metadata),
      data(NULL), U(0, 0, 0), V(0, 0), TW(0, 0) { 
  allocate();

  if (init) {
   // Initialize model weights randomly.
    mt19937 gen(1);
    normal_distribution<Real> gaussian(0, 0.1);
    for (int i = 0; i < size; ++i) {
      TW(i) = gaussian(gen);
    }

    V = metadata->getTagBias();
  } else {
    TW.setZero();
  }
}

TaggedParsedFactoredWeights::TaggedParsedFactoredWeights(const TaggedParsedFactoredWeights& other)
    : ParsedFactoredWeights(other), metadata(other.metadata),
    data(NULL), U(0, 0, 0), V(0, 0), TW(0, 0) { 
  allocate();
  memcpy(data, other.data, size * sizeof(Real));  
}

size_t TaggedParsedFactoredWeights::numParameters() const {
  return ParsedFactoredWeights::numParameters() + size;
}

void TaggedParsedFactoredWeights::allocate() {
  //set vector sizes
  int num_tags = config->num_tags;
  int word_width = config->representation_size;
  int U_size = num_tags * word_width;
  int V_size = num_tags;

  size = U_size + V_size;
  data = new Real[size]; 

  for (int i = 0; i < config->threads; ++i) {
    mutexes.push_back(boost::make_shared<mutex>());
  }

  setModelParameters();
}

void TaggedParsedFactoredWeights::setModelParameters() {
  //new the model parameters
  int num_tags = config->num_tags;
  int word_width = config->representation_size;
  int U_size = num_tags * word_width;
  int V_size = num_tags;

  new (&TW) WeightsType(data, size);

  new (&U) WordVectorsType(data, word_width, num_tags);
  new (&V) WeightsType(data + U_size, V_size);
}

Reals TaggedParsedFactoredWeights::predictWordOverTags(int word, Context context) const {
  //don't actually predict tags at the moment
  Reals weights(numTags(), 0);
  for (int i = 0; i < numTags(); ++i) {
    context.features.back()[0] = config->tag_to_feature[i]; //TODO check
    weights[i] = predictWord(word, context);
  }
  return weights;

  //return Reals(numTags(), ParsedFactoredWeights::predict(word, context));
}

Real TaggedParsedFactoredWeights::predictTag(int tag, Context context) const {
  VectorReal prediction_vector = getPredictionVector(context);
  Real prob = 0;

  //TODO bug in cache
  //auto ret = actionNormalizerCache.get(context.words);
  //if (ret.second) {
  //  prob = (K.col(action).dot(prediction_vector) + L(action) - ret.first);
  //} else {  
    Real normalizer = 0;
    VectorReal tag_probs = logSoftMax(
        U.transpose() * prediction_vector + V, normalizer);
    //actionNormalizerCache.set(context.words, normalizer);
    prob = tag_probs(tag);
  //}

  return -prob;
}
 
Reals TaggedParsedFactoredWeights::predictTag(Context context) const {
  VectorReal prediction_vector = getPredictionVector(context);
  Reals probs(numTags(), 0);

  Real normalizer = 0;
  VectorReal tag_probs = logSoftMax(
      U.transpose() * prediction_vector + V, normalizer);
  //actionNormalizerCache.set(context.words, normalizer);
  for (int i = 0; i < numTags(); ++i) 
    probs[i] = -tag_probs(i);  
  
  return probs;
}
  
int TaggedParsedFactoredWeights::numWords() const {
  return ParsedFactoredWeights::vocabSize();
}

int TaggedParsedFactoredWeights::numTags() const {
  return config->num_tags;
}

int TaggedParsedFactoredWeights::numActions() const {
  return config->numActions();
}

VectorReal TaggedParsedFactoredWeights::getSentenceVectorGradient(const boost::shared_ptr<ParseDataSet>& examples, Real& objective) const {
  VectorReal sentence_gradient = VectorReal::Zero(config->representation_size);

  vector<WordsList> word_contexts;
  vector<WordsList> action_contexts;
  vector<WordsList> tag_contexts;
  vector<MatrixReal> word_context_vectors;
  vector<MatrixReal> action_context_vectors;
  vector<MatrixReal> tag_context_vectors;
  MatrixReal word_prediction_vectors;
  MatrixReal action_prediction_vectors;
  MatrixReal tag_prediction_vectors;
  MatrixReal class_probs;
  vector<VectorReal> word_probs;
  MatrixReal action_probs;
  MatrixReal tag_probs;

  getObjective(examples, word_contexts, action_contexts, tag_contexts,
          word_context_vectors, action_context_vectors, tag_context_vectors,
          word_prediction_vectors, action_prediction_vectors, tag_prediction_vectors,
          class_probs, word_probs, action_probs, tag_probs); 

  MatrixReal word_weighted_representations = ParsedFactoredWeights::getWeightedRepresentations(
      examples->word_examples(), word_prediction_vectors, class_probs, word_probs);
  MatrixReal action_weighted_representations = ParsedFactoredWeights::getActionWeightedRepresentations(
      examples, action_prediction_vectors, action_probs);
  MatrixReal tag_weighted_representations = getTagWeightedRepresentations(
      examples, tag_prediction_vectors, tag_probs);

  int context_width = config->ngram_order - 1;
  MatrixReal word_context_gradients = getContextProduct(context_width, word_weighted_representations, true);
  MatrixReal action_context_gradients = getContextProduct(context_width, action_weighted_representations, true);
  MatrixReal tag_context_gradients = getContextProduct(context_width, tag_weighted_representations, true);
  for (size_t i = 0; i < examples->word_example_size(); ++i) 
    sentence_gradient += word_context_gradients.col(i); 
  for (size_t i = 0; i < examples->action_example_size(); ++i) 
    sentence_gradient += action_context_gradients.col(i); 
  for (size_t i = 0; i < examples->tag_example_size(); ++i) 
    sentence_gradient += tag_context_gradients.col(i); 

  return sentence_gradient;
}

void TaggedParsedFactoredWeights::getGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<TaggedParsedFactoredWeights>& gradient,
    Real& objective,
    MinibatchWords& words,
      bool sentences_only) const {
  vector<WordsList> word_contexts;
  vector<WordsList> action_contexts;
  vector<WordsList> tag_contexts;
  vector<MatrixReal> word_context_vectors;
  vector<MatrixReal> action_context_vectors;
  vector<MatrixReal> tag_context_vectors;
  MatrixReal word_prediction_vectors;
  MatrixReal action_prediction_vectors;
  MatrixReal tag_prediction_vectors;
  MatrixReal class_probs;
  vector<VectorReal> word_probs;
  MatrixReal action_probs;
  MatrixReal tag_probs;
  objective += getObjective(examples, word_contexts, action_contexts, tag_contexts,
          word_context_vectors, action_context_vectors, tag_context_vectors,
          word_prediction_vectors, action_prediction_vectors, tag_prediction_vectors,
          class_probs, word_probs, action_probs, tag_probs); 

  setContextWords(word_contexts, words); 
  setContextWords(action_contexts, words); 
  setContextWords(tag_contexts, words); 

  MatrixReal word_weighted_representations = ParsedFactoredWeights::getWeightedRepresentations(
      examples->word_examples(), word_prediction_vectors, class_probs, word_probs);
  
  MatrixReal action_weighted_representations = ParsedFactoredWeights::getActionWeightedRepresentations(
      examples, action_prediction_vectors, action_probs);

  MatrixReal tag_weighted_representations = getTagWeightedRepresentations(
      examples, tag_prediction_vectors, tag_probs);

  getFullGradient(
      examples, word_contexts, action_contexts, tag_contexts,
      word_context_vectors, action_context_vectors, tag_context_vectors,
      word_prediction_vectors, action_prediction_vectors, tag_prediction_vectors,
      word_weighted_representations, action_weighted_representations, tag_weighted_representations,
      class_probs, word_probs, action_probs, tag_probs, gradient, words, sentences_only);
}

bool TaggedParsedFactoredWeights::checkGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<TaggedParsedFactoredWeights>& gradient,
    double eps) {
if (!ParsedFactoredWeights::checkGradient(examples, gradient, eps)) {
    return false;
  }

  std::cout << size << std::endl;
  for (int i = 0; i < size; ++i) {
    TW(i) += eps;
    Real objective_plus = getObjective(examples);
    TW(i) -= eps;

    TW(i) -= eps;
    Real objective_minus = getObjective(examples);
    TW(i) += eps;

    double est_gradient = (objective_plus - objective_minus) / (2 * eps);
    if (fabs(gradient->TW(i) - est_gradient) > eps) {
      return false;
    }
  }

  return true;
}

void TaggedParsedFactoredWeights::estimateGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<TaggedParsedFactoredWeights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  ParsedFactoredWeights::estimateGradient(examples, gradient, objective, words);
  //TODO estimate tag gradient
} 

Real TaggedParsedFactoredWeights::getObjective(
    const boost::shared_ptr<ParseDataSet>& examples) const {
  vector<WordsList> word_contexts;
  vector<WordsList> action_contexts;
  vector<WordsList> tag_contexts;
  vector<MatrixReal> word_context_vectors;
  vector<MatrixReal> action_context_vectors;
  vector<MatrixReal> tag_context_vectors;
  MatrixReal word_prediction_vectors;
  MatrixReal action_prediction_vectors;
  MatrixReal tag_prediction_vectors;
  MatrixReal class_probs;
  vector<VectorReal> word_probs;
  MatrixReal action_probs;
  MatrixReal tag_probs;
  return getObjective(
      examples, word_contexts, action_contexts, tag_contexts, 
      word_context_vectors, action_context_vectors, tag_context_vectors, word_prediction_vectors, 
      action_prediction_vectors, tag_prediction_vectors, class_probs, word_probs, action_probs, tag_probs);
}

Real TaggedParsedFactoredWeights::getObjective(
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
      MatrixReal& tag_probs) const {
  getContextVectors(examples->word_examples(), word_contexts, word_context_vectors);
  getContextVectors(examples->action_examples(), action_contexts, action_context_vectors);
  getContextVectors(examples->tag_examples(), tag_contexts, tag_context_vectors);
  
  word_prediction_vectors = getPredictionVectors(examples->word_example_size(), word_context_vectors); 
  action_prediction_vectors = getPredictionVectors(examples->action_example_size(), 
                                                   action_context_vectors); 
  tag_prediction_vectors = getPredictionVectors(examples->tag_example_size(), tag_context_vectors); 
  getProbabilities(
      examples, word_contexts, action_contexts, tag_contexts, word_prediction_vectors, 
      action_prediction_vectors, tag_prediction_vectors,
      class_probs, word_probs, action_probs, tag_probs);

  Real objective = 0;
  for (size_t i = 0; i < examples->word_example_size(); ++i) {
    int word_id = examples->word_at(i);
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);

    objective -= class_probs(class_id, i);
    objective -= word_probs[i](word_class_id);
  }

  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    int action_id = examples->action_at(i);
    objective -= action_probs(action_id, i);
  } 

  for (size_t i = 0; i < examples->tag_example_size(); ++i) {
    int tag_id = examples->tag_at(i);
    //std::cout << tag_id << std::endl;
    objective -= tag_probs(tag_id, i);
  }

  //convert out of log-space to probabilities
  for (size_t i = 0; i < examples->word_example_size(); ++i) {
    class_probs.col(i).array() = class_probs.col(i).array().exp();      
    word_probs[i].array() = word_probs[i].array().exp();
  }

  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    action_probs.col(i).array() = action_probs.col(i).array().exp();      
  }
      
  for (size_t i = 0; i < examples->tag_example_size(); ++i) {
    tag_probs.col(i).array() = tag_probs.col(i).array().exp();      
  }

  return objective;
}

void TaggedParsedFactoredWeights::getProbabilities(
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
      MatrixReal& tag_probs) const {
  ParsedFactoredWeights::getProbabilities(examples, word_contexts, action_contexts, word_prediction_vectors, action_prediction_vectors, class_probs, word_probs, action_probs);
  
  tag_probs = U.transpose() * tag_prediction_vectors 
                + V * MatrixReal::Ones(1, examples->tag_example_size());
  for (size_t i = 0; i < examples->tag_example_size(); ++i) {
    tag_probs.col(i) = logSoftMax(tag_probs.col(i));
  }
}

MatrixReal TaggedParsedFactoredWeights::getTagWeightedRepresentations(
    const boost::shared_ptr<ParseDataSet>& examples,
    const MatrixReal& tag_prediction_vectors,
    const MatrixReal& tag_probs) const {
  MatrixReal weighted_representations = U * tag_probs;

  for (size_t i = 0; i < examples->tag_example_size(); ++i) {
    int tag_id = examples->tag_at(i);
    weighted_representations.col(i) -= U.col(tag_id);
  }

  weighted_representations.array() *= activationDerivative(config->activation, tag_prediction_vectors);
  return weighted_representations;
}

void TaggedParsedFactoredWeights::getFullGradient(
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
      MinibatchWords& words,
      bool sentences_only) const {
  for (size_t i = 0; i < examples->word_example_size(); ++i) {
    int word_id = examples->word_at(i); 
    int class_id = index->getClass(word_id);
    int word_class_id = index->getWordIndexInClass(word_id);
    class_probs(class_id, i) -= 1;
    word_probs[i](word_class_id) -= 1;
  }

  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    int action_id = examples->action_at(i); 
    action_probs(action_id, i) -= 1;
  }

  for (size_t i = 0; i < examples->tag_example_size(); ++i) {
    int tag_id = examples->tag_at(i); 
    tag_probs(tag_id, i) -= 1;
  }

  if (!sentences_only) {
    gradient->S += word_prediction_vectors * class_probs.transpose();
    gradient->T += class_probs.rowwise().sum();
    for (size_t i = 0; i < examples->word_example_size(); ++i) {
      int word_id = examples->word_at(i); 
      int class_id = index->getClass(word_id);
      int class_start = index->getClassMarker(class_id);
      int class_size = index->getClassSize(class_id);

      for (int j = 0; j < class_size; ++j) {
        words.addOutputWord(class_start + j);
      }

      gradient->B.segment(class_start, class_size) += word_probs[i];
      gradient->R.block(0, class_start, gradient->R.rows(), class_size) +=
          word_prediction_vectors.col(i) * word_probs[i].transpose();
    }

    gradient->K += action_prediction_vectors * action_probs.transpose();
    gradient->L += action_probs.rowwise().sum();

    gradient->U += tag_prediction_vectors * tag_probs.transpose();
    gradient->V += tag_probs.rowwise().sum();
  }

  getContextGradient(
      examples->word_example_size(), word_contexts, word_context_vectors, word_weighted_representations, gradient, sentences_only);
  getContextGradient(
      examples->action_example_size(), action_contexts, action_context_vectors, action_weighted_representations, gradient, sentences_only);
  getContextGradient(
      examples->tag_example_size(), tag_contexts, tag_context_vectors, tag_weighted_representations, gradient, sentences_only);
}

void TaggedParsedFactoredWeights::syncUpdate(
    const MinibatchWords& words,
    const boost::shared_ptr<TaggedParsedFactoredWeights>& gradient,
      bool sentences_only) {
  ParsedFactoredWeights::syncUpdate(words, gradient, sentences_only);
  if (sentences_only)
    return;

  size_t block_size = TW.size() / mutexes.size() + 1;
  size_t block_start = 0;
  for (size_t i = 0; i < mutexes.size(); ++i) {
    block_size = min(block_size, TW.size() - block_start);
    lock_guard<mutex> lock(*mutexes[i]);
    TW.segment(block_start, block_size) +=
        gradient->TW.segment(block_start, block_size);
    block_start += block_size;
  }
}

Block TaggedParsedFactoredWeights::getBlock() const {
  int thread_id = omp_get_thread_num();
  size_t block_size = TW.size() / config->threads + 1;
  size_t block_start = thread_id * block_size;
  block_size = min(block_size, TW.size() - block_start);
  return make_pair(block_start, block_size);
}

void TaggedParsedFactoredWeights::updateSquared(
    const MinibatchWords& global_words,
    const boost::shared_ptr<TaggedParsedFactoredWeights>& global_gradient,
      bool sentences_only) {
  ParsedFactoredWeights::updateSquared(global_words, global_gradient, sentences_only);
  if (sentences_only)
    return;

  Block block = getBlock();
  TW.segment(block.first, block.second).array() +=
      global_gradient->TW.segment(block.first, block.second).array().square();
}

void TaggedParsedFactoredWeights::updateAdaGrad(
    const MinibatchWords& global_words,
    const boost::shared_ptr<TaggedParsedFactoredWeights>& global_gradient,
    const boost::shared_ptr<TaggedParsedFactoredWeights>& adagrad,
      bool sentences_only) {
  ParsedFactoredWeights::updateAdaGrad(global_words, global_gradient, adagrad, sentences_only);
  if (sentences_only)
    return;

  Block block = getBlock();
  TW.segment(block.first, block.second) -=
      global_gradient->TW.segment(block.first, block.second).binaryExpr(
          adagrad->TW.segment(block.first, block.second),
          CwiseAdagradUpdateOp<Real>(config->step_size));
}

Real TaggedParsedFactoredWeights::regularizerUpdate(
    const MinibatchWords& global_words,
    const boost::shared_ptr<TaggedParsedFactoredWeights>& global_gradient,
    Real minibatch_factor,
      bool sentences_only) {
  Real ret = ParsedFactoredWeights::regularizerUpdate(global_words, global_gradient, minibatch_factor, sentences_only);
  if (sentences_only)
    return ret;

  Block block = getBlock();
  Real sigma = minibatch_factor * config->step_size * config->l2_lbl;
  TW.segment(block.first, block.second) -=
      TW.segment(block.first, block.second) * sigma;

  Real squares = TW.segment(block.first, block.second).array().square().sum();
  ret += 0.5 * minibatch_factor * config->l2_lbl * squares;

  return ret;
}

void TaggedParsedFactoredWeights::clear(const MinibatchWords& words, bool parallel_update) {
  ParsedFactoredWeights::clear(words, parallel_update);

  if (parallel_update) {
    Block block = getBlock();
    TW.segment(block.first, block.second).setZero();
  } else {
    TW.setZero();
  }
}

void TaggedParsedFactoredWeights::clearCache() {
  ParsedFactoredWeights::clearCache();
  tagNormalizerCache.clear();
}

bool TaggedParsedFactoredWeights::operator==(const TaggedParsedFactoredWeights& other) const {
  return Weights::operator==(other)
      && *metadata == *other.metadata
      && size == other.size
      && TW == other.TW;
}

TaggedParsedFactoredWeights::~TaggedParsedFactoredWeights() {
  delete data;
}

} //namespace oxlm
