#include "lbl/weights.h"

#include <iomanip>
#include <random>

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/operators.h"

namespace oxlm {

Weights::Weights() : data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {}

Weights::Weights(
    const boost::shared_ptr<ModelData>& config, const boost::shared_ptr<Metadata>& metadata)
    : config(config), metadata(metadata),
      data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  W.setZero();
}

Weights::Weights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<Metadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : config(config), metadata(metadata),
      data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();

  // Initialize model weights randomly.
  mt19937 gen(1);
  normal_distribution<Real> gaussian(0, 0.1);
  for (int i = 0; i < size; ++i) {
    W(i) = gaussian(gen);
  }

  // Initialize bias with unigram probabilities.
  VectorReal counts = VectorReal::Zero(config->vocab_size);
  for (size_t i = 0; i < training_corpus->size(); ++i) {
    //don't include start of sentence in unigram distribution
    for (size_t k = 1; k < training_corpus->at(i).size(); ++k)
      counts(training_corpus->at(i)[k]) += 1;
  }
  //unigram distribution with plus one smoothing 
  B = ((counts.array() + 1) / (counts.sum() + counts.size())).log();

  cout << "===============================" << endl;
  cout << " Model parameters: " << endl;
  cout << "  Context vocab size = " << config->vocab_size << endl;
  cout << "  Output vocab size = " << config->vocab_size << endl;
  cout << "  Total parameters = " << size << endl;
  cout << "===============================" << endl;
}

Weights::Weights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<Metadata>& metadata,
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices)
    : config(config), metadata(metadata),
      data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  W.setZero();
}

Weights::Weights(const Weights& other)
    : config(other.config), metadata(other.metadata),
      data(NULL), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  memcpy(data, other.data, size * sizeof(Real));
}

void Weights::allocate() {
  int num_context_words = config->vocab_size;
  int num_output_words = config->vocab_size;
  int word_width = config->word_representation_size;
  int context_width = config->ngram_order - 1;

  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_output_words;

  size = Q_size + R_size + context_width * C_size + B_size;
  data = new Real[size];

  setModelParameters();
}

void Weights::setModelParameters() {
  int num_context_words = config->vocab_size;
  int num_output_words = config->vocab_size;
  int word_width = config->word_representation_size;
  int context_width = config->ngram_order - 1;

  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_output_words;

  new (&W) WeightsType(data, size);

  new (&Q) WordVectorsType(data, word_width, num_context_words);
  new (&R) WordVectorsType(data + Q_size, word_width, num_output_words);

  Real* start = data + Q_size + R_size;
  for (int i = 0; i < context_width; ++i) {
    if (config->diagonal_contexts) {
      C.push_back(ContextTransformType(start, word_width, 1));
    } else {
      C.push_back(ContextTransformType(start, word_width, word_width));
    }
    start += C_size;
  }

  new (&B) WeightsType(start, B_size);
}

boost::shared_ptr<Weights> Weights::getGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    Real& objective) const {
  vector<WordId> words;
  vector<vector<WordId>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal word_probs;
  objective = getObjective(
      corpus, indices, words, contexts, context_vectors,
      prediction_vectors, word_probs);

  //word_probs and prediction_vectors have been calculated in getObjective
  MatrixReal weighted_representations = getWeightedRepresentations(
      words, prediction_vectors, word_probs);

  return getFullGradient(
      words, contexts, context_vectors, prediction_vectors,
      weighted_representations, word_probs);
}

void Weights::getContextVectors(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    vector<WordId>& words,
    vector<vector<WordId>>& contexts,
    vector<MatrixReal>& context_vectors) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->word_representation_size;
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, context_width);
  size_t prediction_size = 0; //number of contexts we want to make predictions with
  //exclude 0 from predictions
  for (size_t i = 0; i < indices.size(); ++i) 
    prediction_size += corpus->at(indices[i]).size() - 1;

  words.resize(prediction_size);
  contexts.resize(prediction_size);
  context_vectors.resize(
      context_width, MatrixReal::Zero(word_width, prediction_size));
  size_t l = 0; //cumalative index
  for (size_t i = 0; i < indices.size(); ++i) {
    for (size_t k = 1; k < corpus->at(indices[i]).size(); ++k) {
      words[l] = corpus->at(indices[i]).at(k);
      contexts[l] = processor->extract(indices[i], k);
      for (int j = 0; j < context_width; ++j) {
        context_vectors[j].col(l) = Q.col(contexts[l][j]);
      }
      ++l;
    }
  }
}

MatrixReal Weights::getPredictionVectors(
    const size_t prediction_size,
    const vector<MatrixReal>& context_vectors) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->word_representation_size;
  MatrixReal prediction_vectors = MatrixReal::Zero(word_width, prediction_size);

  for (int i = 0; i < context_width; ++i) {
    prediction_vectors += getContextProduct(i, context_vectors[i]);
  }

  if (config->sigmoid) {
    for (size_t i = 0; i < prediction_size; ++i) {
      prediction_vectors.col(i) = sigmoid(prediction_vectors.col(i));
    }
  }

  return prediction_vectors;
}

MatrixReal Weights::getContextProduct(
    int index, const MatrixReal& representations, bool transpose) const {
  if (config->diagonal_contexts) {
    return C[index].asDiagonal() * representations;
  } else {
    if (transpose) {
      return C[index].transpose() * representations;
    } else {
      return C[index] * representations;
    }
  }
}

MatrixReal Weights::getProbabilities(
    const size_t prediction_size, const MatrixReal& prediction_vectors) const {
  MatrixReal word_probs = R.transpose() * prediction_vectors + B * MatrixReal::Ones(1, prediction_size);
  for (size_t i = 0; i < prediction_size; ++i) {
    word_probs.col(i) = softMax(word_probs.col(i));
  }

  return word_probs;
}

MatrixReal Weights::getWeightedRepresentations(
    const vector<WordId>& words,
    const MatrixReal& prediction_vectors,
    const MatrixReal& word_probs) const {
  MatrixReal weighted_representations = R * word_probs;

  for (size_t i = 0; i < words.size(); ++i) {
    weighted_representations.col(i) -= R.col(words[i]);
  }

  if (config->sigmoid) {
    weighted_representations.array() *= sigmoidDerivative(prediction_vectors);
  }

  return weighted_representations;
}

boost::shared_ptr<Weights> Weights::getFullGradient(
    const vector<WordId>& words,
    const vector<vector<WordId>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& prediction_vectors,
    const MatrixReal& weighted_representations,
    MatrixReal& word_probs) const {
  boost::shared_ptr<Weights> gradient =
      boost::make_shared<Weights>(config, metadata);

  for (size_t i = 0; i < words.size(); ++i) {
    word_probs(words[i], i) -= 1;
  }

  gradient->R = prediction_vectors * word_probs.transpose();
  gradient->B = word_probs.rowwise().sum();

  getContextGradient(
      words.size(), contexts, context_vectors, weighted_representations, gradient);

  return gradient;
}

void Weights::getContextGradient(
    const size_t prediction_size,
    const vector<vector<WordId>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& weighted_representations,
    const boost::shared_ptr<Weights>& gradient) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->word_representation_size;
  MatrixReal context_gradients = MatrixReal::Zero(word_width, prediction_size);
  for (int j = 0; j < context_width; ++j) {
    context_gradients = getContextProduct(j, weighted_representations, true);
    for (size_t i = 0; i < prediction_size; ++i) {
      gradient->Q.col(contexts[i][j]) += context_gradients.col(i);
    }

    if (config->diagonal_contexts) {
      gradient->C[j] = context_vectors[j].cwiseProduct(weighted_representations).rowwise().sum();
    } else {
      gradient->C[j] = weighted_representations * context_vectors[j].transpose();
    }
  }
}

bool Weights::checkGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<Weights>& gradient,
    double eps) {
  vector<vector<WordId>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal word_probs;

  for (int i = 0; i < size; ++i) {
    W(i) += eps;
    Real objective_plus = getObjective(corpus, indices);
    W(i) -= eps;

    W(i) -= eps;
    Real objective_minus = getObjective(corpus, indices);
    W(i) += eps;

    double est_gradient = (objective_plus - objective_minus) / (2 * eps);
    if (fabs(gradient->W(i) - est_gradient) > eps) {
      cout << i << " " << gradient->W(i) << " " << est_gradient << endl;
      return false;
    }
  }

  return true;
}

Real Weights::getObjective(
    const boost::shared_ptr<Corpus>& corpus, const vector<int>& indices) const {
  vector<WordId> words;
  vector<vector<WordId>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal word_probs;
  return getObjective(
      corpus, indices, words, contexts, context_vectors, prediction_vectors, word_probs);
}

Real Weights::getObjective(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    vector<WordId>& words,
    vector<vector<WordId>>& contexts,
    vector<MatrixReal>& context_vectors,
    MatrixReal& prediction_vectors,
    MatrixReal& word_probs) const {
  getContextVectors(corpus, indices, words, contexts, context_vectors);
  prediction_vectors = getPredictionVectors(words.size(), context_vectors);
  word_probs = getProbabilities(words.size(), prediction_vectors);

  Real objective = 0;
  for (size_t i = 0; i < words.size(); ++i) {
    objective -= log(word_probs(words[i], i));
  }

  return objective;
}

vector<vector<int>> Weights::getNoiseWords(
    const vector<WordId>& words,
    const boost::shared_ptr<Corpus>& corpus) const {
  vector<vector<int>> noise_words(words.size());
  for (size_t i = 0; i < words.size(); ++i) {
    for (int j = 0; j < config->noise_samples; ++j) {
      int k = rand() % corpus->size();
      noise_words[i].push_back(corpus->at(k).at(rand() % corpus->at(k).size()));
    }
  }

  return noise_words;
}

void Weights::estimateProjectionGradient(
    const vector<WordId>& words,
    const boost::shared_ptr<Corpus>& corpus,
    const MatrixReal& prediction_vectors,
    const boost::shared_ptr<Weights>& gradient,
    MatrixReal& weighted_representations,
    Real& objective) const {
  int noise_samples = config->noise_samples;
  int word_width = config->word_representation_size;
  VectorReal unigram = metadata->getUnigram();
  vector<vector<int>> noise_words = getNoiseWords(words, corpus);

  objective = 0;
  weighted_representations = MatrixReal::Zero(word_width, words.size());
  for (size_t i = 0; i < words.size(); ++i) {
    int word_id = words[i]; 
    Real log_pos_prob = R.col(word_id).dot(prediction_vectors.col(i)) + B(word_id);
    Real pos_prob = exp(log_pos_prob);
    assert(pos_prob <= numeric_limits<Real>::max());

    Real pos_weight = (noise_samples * unigram(word_id)) / (pos_prob + noise_samples * unigram(word_id));
    weighted_representations.col(i) -= pos_weight * R.col(word_id);

    objective -= log(1 - pos_weight);

    gradient->R.col(word_id) -= pos_weight * prediction_vectors.col(i);
    gradient->B(word_id) -= pos_weight;

    for (int j = 0; j < noise_samples; ++j) {
      int noise_word_id = noise_words[i][j];
      Real log_neg_prob = R.col(noise_word_id).dot(prediction_vectors.col(i)) + B(noise_word_id);
      Real neg_prob = exp(log_neg_prob);
      assert(neg_prob <= numeric_limits<Real>::max());

      Real neg_weight = neg_prob / (neg_prob + noise_samples * unigram(noise_word_id));
      weighted_representations.col(i) += neg_weight * R.col(noise_word_id);

      objective -= log(1 - neg_weight);

      gradient->R.col(noise_word_id) += neg_weight * prediction_vectors.col(i);
      gradient->B(noise_word_id) += neg_weight;
    }
  }
}

boost::shared_ptr<Weights> Weights::estimateGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    Real& objective) const {
  vector<WordId> words;
  vector<vector<WordId>> contexts;
  vector<MatrixReal> context_vectors;
  getContextVectors(corpus, indices, words, contexts, context_vectors);

  MatrixReal prediction_vectors =
      getPredictionVectors(words.size(), context_vectors);

  boost::shared_ptr<Weights> gradient =
      boost::make_shared<Weights>(config, metadata);
  MatrixReal weighted_representations;
  estimateProjectionGradient(
      words, corpus, prediction_vectors, gradient,
      weighted_representations, objective);

  if (config->sigmoid) {
    weighted_representations.array() *= sigmoidDerivative(prediction_vectors);
  }

  getContextGradient(
      words.size(), contexts, context_vectors, weighted_representations, gradient);

  return gradient;
}

void Weights::update(const boost::shared_ptr<Weights>& gradient) {
  W += gradient->W;
}

void Weights::updateSquared(const boost::shared_ptr<Weights>& global_gradient) {
  W.array() += global_gradient->W.array().square();
}

void Weights::updateAdaGrad(
    const boost::shared_ptr<Weights>& global_gradient,
    const boost::shared_ptr<Weights>& adagrad) {
  W -= global_gradient->W.binaryExpr(
      adagrad->W, CwiseAdagradUpdateOp<Real>(config->step_size));
}

Real Weights::regularizerUpdate(
    const boost::shared_ptr<Weights>& global_gradient, Real minibatch_factor) {
  Real sigma = minibatch_factor * config->step_size * config->l2_lbl;
  W -= W * sigma;
  return 0.5 * minibatch_factor * config->l2_lbl * W.array().square().sum();
}

VectorReal Weights::getPredictionVector(const vector<int>& context) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->word_representation_size;

  VectorReal prediction_vector = VectorReal::Zero(word_width);
  for (int i = 0; i < context_width; ++i) {
    if (config->diagonal_contexts) {
      prediction_vector += C[i].asDiagonal() * Q.col(context[i]);
    } else {
      prediction_vector += C[i] * Q.col(context[i]);
    }
  }

  return config->sigmoid ? sigmoid(prediction_vector) : prediction_vector;
}

Real Weights::predict(int word_id, vector<int> context) const {
  VectorReal prediction_vector = getPredictionVector(context);

  auto ret = normalizerCache.get(context);
  if (ret.second) {
    return R.col(word_id).dot(prediction_vector) + B(word_id) - ret.first;
  } else {
    Real normalizer = 0;
    VectorReal word_probs = logSoftMax(
        R.transpose() * prediction_vector + B, normalizer);
    normalizerCache.set(context, normalizer);
    return word_probs(word_id);
  }
}

void Weights::clearCache() {
  normalizerCache.clear();
}

MatrixReal Weights::getWordVectors() const {
  return R;
}

bool Weights::operator==(const Weights& other) const {
  return *config == *other.config
      && *metadata == *other.metadata
      && size == other.size
      && W == other.W;
}

Weights::~Weights() {
  delete data;
}

} // namespace oxlm
