#include "lbl/parsed_weights.h"

#include <iomanip>

#include <boost/make_shared.hpp>

#include "lbl/operators.h"

namespace oxlm {

ParsedWeights::ParsedWeights()
    : data(NULL), K(0, 0, 0), L(0, 0), PW(0, 0) {}

ParsedWeights::ParsedWeights(
    const boost::shared_ptr<ModelConfig>& config,
    const boost::shared_ptr<ParsedMetadata>& metadata,
    bool init)
    : Weights(config, metadata, init), metadata(metadata),
      data(NULL), K(0, 0, 0), L(0, 0), PW(0, 0) { 
  allocate();

  if (init) {
   // Initialize model weights randomly.
    mt19937 gen(1);
    normal_distribution<Real> gaussian(0, 0.1);
    for (int i = 0; i < size; ++i) {
      PW(i) = gaussian(gen);
    }

    L = metadata->getActionBias();
  } else {
    PW.setZero();
  }
}

ParsedWeights::ParsedWeights(const ParsedWeights& other)
    : Weights(other), metadata(other.metadata),
    data(NULL), K(0, 0, 0), L(0, 0), PW(0, 0) { 
  allocate();
  memcpy(data, other.data, size * sizeof(Real));  
}

size_t ParsedWeights::numParameters() const {
  return Weights::numParameters() + size;
}

void ParsedWeights::allocate() {
  //set vector sizes
  int num_actions = config->numActions();
  int word_width = config->representation_size;
  int K_size = num_actions * word_width;
  int L_size = num_actions;

  size = K_size + L_size;
  data = new Real[size]; 

  for (int i = 0; i < config->threads; ++i) {
    mutexes.push_back(boost::make_shared<mutex>());
  }

  setModelParameters();
}

void ParsedWeights::setModelParameters() {
  //new the model parameters
  int num_actions = config->numActions();
  int word_width = config->representation_size;
  int K_size = num_actions * word_width;
  int L_size = num_actions;

  new (&PW) WeightsType(data, size);

  new (&K) WordVectorsType(data, word_width, num_actions);
  new (&L) WeightsType(data + K_size, L_size);
}

Real ParsedWeights::predictWord(int word, Words context) const {
  //implement as unlexicalised model
    return 0.0;
}

Reals ParsedWeights::predictWord(Words context) const {
  return Reals(numWords(), 0.0);
}

Real ParsedWeights::predictTag(int tag, Words context) const {
  return Weights::predict(tag, context);
}
 
Reals ParsedWeights::predictTag(Words context) const {
  return Weights::predict(context);
}
  
Real ParsedWeights::predictAction(WordId action, Words context) const {
  VectorReal prediction_vector = getPredictionVector(context);
  Real prob = 0;

  //TODO debug cache
  //auto ret = actionNormalizerCache.get(context);
  //if (ret.second) {
  //  prob = (K.col(action).dot(prediction_vector) + L(action) - ret.first);
  //} else {  
    Real normalizer = 0;
    VectorReal action_probs = logSoftMax(
        K.transpose() * prediction_vector + L, normalizer);
    //actionNormalizerCache.set(context, normalizer);
    prob = action_probs(action);
  //}

  return -prob;
}

Reals ParsedWeights::predictAction(Words context) const {
  VectorReal prediction_vector = getPredictionVector(context);
  Reals probs(numActions(), 0);

  Real normalizer = 0;
  VectorReal action_probs = logSoftMax(
      K.transpose() * prediction_vector + L, normalizer);
  //actionNormalizerCache.set(context, normalizer);
  for (int i = 0; i < numActions(); ++i) 
    probs[i] = -action_probs(i);  
  
  return probs;
}

int ParsedWeights::numWords() const {
  return 1;
}

int ParsedWeights::numTags() const {
  return Weights::vocabSize();
}

int ParsedWeights::numActions() const {
  return config->numActions();
}
 
void ParsedWeights::getGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<ParsedWeights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  vector<vector<int>> word_contexts;
  vector<vector<int>> action_contexts;
  vector<MatrixReal> word_context_vectors;
  vector<MatrixReal> action_context_vectors;
  MatrixReal word_prediction_vectors;
  MatrixReal action_prediction_vectors;
  MatrixReal word_probs;
  MatrixReal action_probs;
  objective += getObjective(examples, word_contexts, action_contexts, word_context_vectors, 
          action_context_vectors, word_prediction_vectors, action_prediction_vectors,
          word_probs, action_probs); 

  setContextWords(word_contexts, words); 
  setContextWords(action_contexts, words); 

  MatrixReal word_weighted_representations = Weights::getWeightedRepresentations(
      examples->tag_examples(), word_prediction_vectors, word_probs);
  
  MatrixReal action_weighted_representations = getActionWeightedRepresentations(
      examples, action_prediction_vectors, action_probs);

  getFullGradient(
      examples, word_contexts, action_contexts, word_context_vectors, action_context_vectors,
      word_prediction_vectors, action_prediction_vectors, word_weighted_representations,
      action_weighted_representations, word_probs, action_probs, gradient, words);
}

bool ParsedWeights::checkGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<ParsedWeights>& gradient,
    double eps) {
if (!Weights::checkGradient(examples->tag_examples(), gradient, eps)) {
    return false;
  }

  std::cout << size << std::endl;
  for (int i = 0; i < size; ++i) {
    PW(i) += eps;
    Real objective_plus = getObjective(examples);
    PW(i) -= eps;

    PW(i) -= eps;
    Real objective_minus = getObjective(examples);
    PW(i) += eps;

    double est_gradient = (objective_plus - objective_minus) / (2 * eps);
    if (fabs(gradient->PW(i) - est_gradient) > eps) {
      return false;
    }
  }

  return true;
}

void ParsedWeights::estimateGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<ParsedWeights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  Weights::estimateGradient(examples->tag_examples(), gradient, objective, words);
  //TODO estimate action gradient
} 

Real ParsedWeights::getObjective(
    const boost::shared_ptr<ParseDataSet>& examples) const {
  vector<vector<int>> word_contexts;
  vector<vector<int>> action_contexts;
  vector<MatrixReal> word_context_vectors;
  vector<MatrixReal> action_context_vectors;
  MatrixReal word_prediction_vectors;
  MatrixReal action_prediction_vectors;
  MatrixReal word_probs;
  MatrixReal action_probs;
  return getObjective(
      examples, word_contexts, action_contexts, word_context_vectors, action_context_vectors, 
      word_prediction_vectors, action_prediction_vectors, word_probs, action_probs);
}

Real ParsedWeights::getObjective(
    const boost::shared_ptr<ParseDataSet>& examples,
    vector<vector<int>>& word_contexts,
    vector<vector<int>>& action_contexts,
    vector<MatrixReal>& word_context_vectors,
    vector<MatrixReal>& action_context_vectors,
    MatrixReal& word_prediction_vectors,
    MatrixReal& action_prediction_vectors,
    MatrixReal& word_probs,
    MatrixReal& action_probs) const {
  //computing the hidden layer (in forward pass) twice when predicting both word and action,
  //but else we need another way to store training examples 
  getContextVectors(examples->tag_examples(), word_contexts, word_context_vectors);
  getContextVectors(examples->action_examples(), action_contexts, action_context_vectors);

  word_prediction_vectors = getPredictionVectors(examples->tag_example_size(), word_context_vectors); 
  action_prediction_vectors = getPredictionVectors(examples->action_example_size(), 
                                                   action_context_vectors); 

  getProbabilities(
      examples, word_contexts, action_contexts, word_prediction_vectors, action_prediction_vectors,
      word_probs, action_probs);

  Real objective = 0;
  for (size_t i = 0; i < examples->tag_example_size(); ++i) {
    int tag_id = examples->tag_at(i);
    objective -= word_probs(tag_id, i);    
  }

  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    int action_id = examples->action_at(i);
    objective -= action_probs(action_id, i);
  } 

  //convert out of log-space to probabilities
  for (size_t i = 0; i < examples->tag_example_size(); ++i) {
    word_probs.col(i).array() = word_probs.col(i).array().exp();      
  }

  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    action_probs.col(i).array() = action_probs.col(i).array().exp();      
  }
     
  return objective;
}

void ParsedWeights::getProbabilities(
    const boost::shared_ptr<ParseDataSet>& examples,
    const vector<vector<int>>& word_contexts,
    const vector<vector<int>>& action_contexts,
    const MatrixReal& word_prediction_vectors,
    const MatrixReal& action_prediction_vectors,
    MatrixReal& word_probs,
    MatrixReal& action_probs) const {
  word_probs = Weights::getProbabilities(examples->tag_examples(), word_prediction_vectors);
  
  action_probs = K.transpose() * action_prediction_vectors 
                + L * MatrixReal::Ones(1, examples->action_example_size());
  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    action_probs.col(i) = logSoftMax(action_probs.col(i));
  }
}

MatrixReal ParsedWeights::getActionWeightedRepresentations(
    const boost::shared_ptr<ParseDataSet>& examples,
    const MatrixReal& action_prediction_vectors,
    const MatrixReal& action_probs) const {
  MatrixReal weighted_representations = K * action_probs;

  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    int action_id = examples->action_at(i);
    weighted_representations.col(i) -= K.col(action_id);
  }

  weighted_representations.array() *= activationDerivative(config->activation, action_prediction_vectors);

  return weighted_representations;
}

void ParsedWeights::getFullGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const vector<vector<int>>& word_contexts,
    const vector<vector<int>>& action_contexts,
    const vector<MatrixReal>& word_context_vectors,
    const vector<MatrixReal>& action_context_vectors,
    const MatrixReal& word_prediction_vectors,
    const MatrixReal& action_prediction_vectors,
    const MatrixReal& word_weighted_representations,
    const MatrixReal& action_weighted_representations,
    MatrixReal& word_probs,
    MatrixReal& action_probs,
    const boost::shared_ptr<ParsedWeights>& gradient,
    MinibatchWords& words) const {
  //reimplementing factored_weights computations
  for (size_t i = 0; i < examples->tag_example_size(); ++i) {
    int tag_id = examples->tag_at(i); 
    word_probs(tag_id, i) -= 1;
  }

  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    int action_id = examples->action_at(i); 
    action_probs(action_id, i) -= 1;
  }

  for (size_t tag_id = 0; tag_id < numTags(); ++tag_id) {
    words.addOutputWord(tag_id);
  }

  gradient->R += word_prediction_vectors * word_probs.transpose();
  gradient->B += word_probs.rowwise().sum();
  
  gradient->K += action_prediction_vectors * action_probs.transpose();
  gradient->L += action_probs.rowwise().sum();

  getContextGradient(
      examples->tag_example_size(), word_contexts, word_context_vectors, word_weighted_representations, gradient);
  getContextGradient(
      examples->action_example_size(), action_contexts, action_context_vectors, action_weighted_representations, gradient);
}

std::vector<Words> ParsedWeights::getNoiseWords(
    const boost::shared_ptr<ParseDataSet>& examples) const {
  Weights::getNoiseWords(examples->tag_examples());
} 

void ParsedWeights::estimateProjectionGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const MatrixReal& prediction_vectors,
    const boost::shared_ptr<ParsedWeights>& gradient,
    MatrixReal& weighted_representations,
    Real& objective,
    MinibatchWords& words) const {
  Weights::estimateProjectionGradient(examples->tag_examples(), prediction_vectors, gradient, weighted_representations, objective, words);
  //TODO estimate for actions
} 

void ParsedWeights::syncUpdate(
    const MinibatchWords& words,
    const boost::shared_ptr<ParsedWeights>& gradient) {
  Weights::syncUpdate(words, gradient);

  size_t block_size = PW.size() / mutexes.size() + 1;
  size_t block_start = 0;
  for (size_t i = 0; i < mutexes.size(); ++i) {
    block_size = min(block_size, PW.size() - block_start);
    lock_guard<mutex> lock(*mutexes[i]);
    PW.segment(block_start, block_size) +=
        gradient->PW.segment(block_start, block_size);
    block_start += block_size;
  }
}

Block ParsedWeights::getBlock() const {
  int thread_id = omp_get_thread_num();
  size_t block_size = PW.size() / config->threads + 1;
  size_t block_start = thread_id * block_size;
  block_size = min(block_size, PW.size() - block_start);
  return make_pair(block_start, block_size);
}

void ParsedWeights::updateSquared(
    const MinibatchWords& global_words,
    const boost::shared_ptr<ParsedWeights>& global_gradient) {
  Weights::updateSquared(global_words, global_gradient);

  Block block = getBlock();
  PW.segment(block.first, block.second).array() +=
      global_gradient->PW.segment(block.first, block.second).array().square();
}

void ParsedWeights::updateAdaGrad(
    const MinibatchWords& global_words,
    const boost::shared_ptr<ParsedWeights>& global_gradient,
    const boost::shared_ptr<ParsedWeights>& adagrad) {
  Weights::updateAdaGrad(global_words, global_gradient, adagrad);

  Block block = getBlock();
  PW.segment(block.first, block.second) -=
      global_gradient->PW.segment(block.first, block.second).binaryExpr(
          adagrad->PW.segment(block.first, block.second),
          CwiseAdagradUpdateOp<Real>(config->step_size));
}

Real ParsedWeights::regularizerUpdate(
    const boost::shared_ptr<ParsedWeights>& global_gradient,
    Real minibatch_factor) {
  Real ret = Weights::regularizerUpdate(global_gradient, minibatch_factor);

  Block block = getBlock();
  Real sigma = minibatch_factor * config->step_size * config->l2_lbl;
  PW.segment(block.first, block.second) -=
      PW.segment(block.first, block.second) * sigma;

  Real squares = PW.segment(block.first, block.second).array().square().sum();
  ret += 0.5 * minibatch_factor * config->l2_lbl * squares;

  return ret;
}

void ParsedWeights::clear(const MinibatchWords& words, bool parallel_update) {
  Weights::clear(words, parallel_update);

  if (parallel_update) {
    Block block = getBlock();
    PW.segment(block.first, block.second).setZero();
  } else {
    PW.setZero();
  }
}

void ParsedWeights::clearCache() {
  Weights::clearCache();
  actionNormalizerCache.clear();
}

bool ParsedWeights::operator==(const ParsedWeights& other) const {
  return Weights::operator==(other)
      && *metadata == *other.metadata
      && size == other.size
      && PW == other.PW;
}

ParsedWeights::~ParsedWeights() {
  delete data;
}

} //namespace oxlm
