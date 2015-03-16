#include "lbl/weights.h"

#include <iomanip>
#include <random>

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/operators.h"

namespace oxlm {

Weights::Weights() : data(NULL), P(0, 0, 0), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {}

Weights::Weights(
    const boost::shared_ptr<ModelConfig>& config, 
    const boost::shared_ptr<Metadata>& metadata,
    bool init)
    : config(config), metadata(metadata),
      data(NULL), P(0, 0, 0), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  
  if (init) {
  // Initialize model weights randomly.
  mt19937 gen(1);
  normal_distribution<Real> gaussian(0, 0.1);
  for (int i = 0; i < size; ++i) {
    W(i) = gaussian(gen);
  }

  // Initialize bias with unigram probabilities.
  B = metadata->getSmoothedUnigram().array().log();

  cout << "===============================" << endl;
  cout << " Model parameters: " << endl;
  cout << "  Context vocab size = " << config->vocab_size << endl;
  cout << "  Feature vocab size = " << config->num_tags << endl;
  cout << "  Output vocab size = " << config->vocab_size << endl;
  cout << "  Total parameters = " << numParameters() << endl;
  cout << "===============================" << endl;
  } else {
    W.setZero();
  }
}

Weights::Weights(const Weights& other)
    : config(other.config), metadata(other.metadata),
      data(NULL), P(0, 0, 0), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  memcpy(data, other.data, size * sizeof(Real));
}

void Weights::allocate() {
  int num_feature_words = config->num_tags;
  int num_context_words = config->vocab_size;
  int num_output_words = config->vocab_size;
  int word_width = config->representation_size;
  int context_width = config->ngram_order - 1;

  int P_size = word_width * num_feature_words;
  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_output_words;

  size = P_size + Q_size + R_size + context_width * C_size + B_size;
  data = new Real[size];

  for (int i = 0; i < num_feature_words; ++i) {
    mutexesP.push_back(boost::make_shared<mutex>());
  }
  for (int i = 0; i < num_context_words; ++i) {
    mutexesQ.push_back(boost::make_shared<mutex>());
  }
  for (int i = 0; i < num_output_words; ++i) {
    mutexesR.push_back(boost::make_shared<mutex>());
  }
  for (int i = 0; i < context_width; ++i) {
    mutexesC.push_back(boost::make_shared<mutex>());
  }
  mutexB = boost::make_shared<mutex>();

  setModelParameters();
}

void Weights::setModelParameters() {
  int num_feature_words = config->num_tags;
  int num_context_words = config->vocab_size;
  int num_output_words = config->vocab_size;
  int word_width = config->representation_size;
  int context_width = config->ngram_order - 1;

  int P_size = word_width * num_feature_words;
  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_output_words;

  new (&W) WeightsType(data, size);

  new (&P) WordVectorsType(data, word_width, num_feature_words);
  new (&Q) WordVectorsType(data + P_size, word_width, num_context_words);
  new (&R) WordVectorsType(data + P_size + Q_size, word_width, num_output_words);

  Real* start = data + P_size + Q_size + R_size;
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

size_t Weights::numParameters() const {
  return size;
}

void Weights::getGradient(
    const boost::shared_ptr<DataSet>& examples,
    const boost::shared_ptr<Weights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  vector<vector<int>> contexts;
  vector<WordsList> features;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal word_probs;
  objective += getObjective(examples,
      contexts, features, context_vectors, prediction_vectors, word_probs);

  setContextWords(contexts, words);
  setFeatureWords(features, words);

  MatrixReal weighted_representations = getWeightedRepresentations(
      examples, prediction_vectors, word_probs);

  getFullGradient(
      examples, contexts, features, context_vectors, prediction_vectors,
      weighted_representations, word_probs, gradient, words);
}

void Weights::getContextVectors(
    const boost::shared_ptr<DataSet>& examples,
    vector<vector<int>>& contexts,
    vector<WordsList>& features,
    vector<MatrixReal>& context_vectors) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->representation_size;

  contexts.resize(examples->size());
  features.resize(examples->size());
  context_vectors.resize(
      context_width, MatrixReal::Zero(word_width, examples->size()));
  for (size_t i = 0; i < examples->size(); ++i) {
    contexts[i] = examples->contextAt(i).words;
    features[i] = examples->contextAt(i).features;
    for (int j = 0; j < context_width; ++j) {
      if (!config->compositional || config->pos_annotated) 
        context_vectors[j].col(i) = Q.col(contexts[i][j]);
      if (config->compositional) {
        for (auto feat: features[i][j]) {
          context_vectors[j].col(i) += P.col(feat);
        }
      }  
    }
  }
}

void Weights::setContextWords(
    const vector<vector<int>>& contexts,
    MinibatchWords& words) const {
  for (const auto& context: contexts) {
    for (int word_id: context) {
      words.addContextWord(word_id);
    }
  }
}

void Weights::setFeatureWords(
    const vector<WordsList>& features,
    MinibatchWords& words) const {
  for (const auto& context: features) {
    for (const auto& item: context) {
      for (int word_id: item) {
        words.addFeatureWord(word_id);
      }
    }
  }
}

MatrixReal Weights::getPredictionVectors(
    size_t prediction_size,
    const vector<MatrixReal>& context_vectors) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->representation_size;
  MatrixReal prediction_vectors = MatrixReal::Zero(word_width, prediction_size);

  for (int i = 0; i < context_width; ++i) {
    prediction_vectors += getContextProduct(i, context_vectors[i]);
  }

  return applyActivation<MatrixReal>(config->activation, prediction_vectors);
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
    const boost::shared_ptr<DataSet>& examples,
    const MatrixReal& prediction_vectors) const {
  MatrixReal word_probs = R.transpose() * prediction_vectors 
                          + B * MatrixReal::Ones(1, examples->size());
  for (size_t i = 0; i < examples->size(); ++i) {    
    word_probs.col(i) = logSoftMax(word_probs.col(i));
  }

  return word_probs;
}

MatrixReal Weights::getWeightedRepresentations(
    const boost::shared_ptr<DataSet>& examples,
    const MatrixReal& prediction_vectors,
    const MatrixReal& word_probs) const {
  MatrixReal weighted_representations = R * word_probs;

  for (size_t i = 0; i < examples->size(); ++i) {
    weighted_representations.col(i) -= R.col(examples->wordAt(i));
  }

  weighted_representations.array() *= activationDerivative(config->activation, prediction_vectors);
  return weighted_representations;
}

void Weights::getFullGradient(
    const boost::shared_ptr<DataSet>& examples,
    const vector<vector<int>>& contexts,
    const vector<WordsList>& features,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& prediction_vectors,
    const MatrixReal& weighted_representations,
    MatrixReal& word_probs,
    const boost::shared_ptr<Weights>& gradient,
    MinibatchWords& words) const {
  for (size_t i = 0; i < examples->size(); ++i) {
    word_probs(examples->wordAt(i), i) -= 1;
  }

  for (size_t word_id = 0; word_id < config->vocab_size; ++word_id) {
    words.addOutputWord(word_id);
  }

  gradient->R += prediction_vectors * word_probs.transpose();
  gradient->B += word_probs.rowwise().sum();

  getContextGradient(
      examples->size(), contexts, features, context_vectors, weighted_representations, gradient);
}

void Weights::getContextGradient(
    size_t prediction_size,
    const vector<vector<int>>& contexts,
    const vector<WordsList>& features,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& weighted_representations,
    const boost::shared_ptr<Weights>& gradient) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->representation_size;
  MatrixReal context_gradients = MatrixReal::Zero(word_width, prediction_size);
  for (int j = 0; j < context_width; ++j) {
    context_gradients = getContextProduct(j, weighted_representations, true);
    for (size_t i = 0; i < prediction_size; ++i) {
      if (!config->compositional || config->pos_annotated) 
        gradient->Q.col(contexts[i][j]) += context_gradients.col(i);
      if (config->compositional) {
        for (auto feat: features[i][j]) {
          gradient->P.col(feat) += context_gradients.col(i);
        }
      } 
    }

    if (config->diagonal_contexts) {
      gradient->C[j] += context_vectors[j].cwiseProduct(weighted_representations).rowwise().sum();
    } else {
      gradient->C[j] += weighted_representations * context_vectors[j].transpose();
    }
  }
  //std::cout << std::endl;
}

bool Weights::checkGradient(
    const boost::shared_ptr<DataSet>& examples,
    const boost::shared_ptr<Weights>& gradient,
    double eps) {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal word_probs;
  int P_size = config->representation_size * config->num_tags;
  //std::cout << P_size << std::endl;

  for (int i = 0; i < size; ++i) {
    W(i) += eps;
    Real objective_plus = getObjective(examples);
    W(i) -= eps;

    W(i) -= eps;
    Real objective_minus = getObjective(examples);
    W(i) += eps;

    double est_gradient = (objective_plus - objective_minus) / (2 * eps);
    if (fabs(gradient->W(i) - est_gradient) > eps) {
      std::cout << i << " " <<  gradient->W(i) << " " << est_gradient << " ";
      //return false;
    }
  }

  return true;
}

Real Weights::getObjective(
    const boost::shared_ptr<DataSet>& examples) const {
  vector<vector<int>> contexts;
  vector<WordsList> features;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal word_probs;
  return getObjective(
      examples, contexts, features, context_vectors, prediction_vectors, word_probs);
}

Real Weights::getObjective(
    const boost::shared_ptr<DataSet>& examples,
    vector<vector<int>>& contexts,
    vector<WordsList>& features,
    vector<MatrixReal>& context_vectors,
    MatrixReal& prediction_vectors,
    MatrixReal& word_probs) const {
  getContextVectors(examples, contexts, features, context_vectors);
  prediction_vectors = getPredictionVectors(examples->size(), context_vectors);
  
  word_probs = getProbabilities(examples, prediction_vectors);

  Real objective = 0;
  for (size_t i = 0; i < examples->size(); ++i) {
    Real word_likelihood = -word_probs(examples->wordAt(i), i);    
    objective += word_likelihood;
  }

  for (size_t i = 0; i < examples->size(); ++i) {
    word_probs.col(i).array() = word_probs.col(i).array().exp();
  } 

  return objective;
}

vector<vector<int>> Weights::getNoiseWords(
    const boost::shared_ptr<DataSet>& examples) const {
  if (!wordDists.get()) {
    wordDists.reset(new WordDistributions(metadata->getUnigram()));
  }
    
  vector<vector<int>> noise_words(examples->size());
  for (size_t i = 0; i < examples->size(); ++i) {
    for (int j = 0; j < config->noise_samples; ++j) {
      noise_words[i].push_back(wordDists->sample());
    }
  }

  return noise_words;
}

void Weights::estimateProjectionGradient(
    const boost::shared_ptr<DataSet>& examples,
    const MatrixReal& prediction_vectors,
    const boost::shared_ptr<Weights>& gradient,
    MatrixReal& weighted_representations,
    Real& objective,
    MinibatchWords& words) const {
  int noise_samples = config->noise_samples;
  int word_width = config->representation_size;
  VectorReal unigram = metadata->getUnigram();
  vector<vector<int>> noise_words = getNoiseWords(examples);

  for (size_t i = 0; i < examples->size(); ++i) {
    words.addOutputWord(examples->wordAt(i));
    for (int word_id: noise_words[i]) {
      words.addOutputWord(word_id);
    }
  }

  weighted_representations = MatrixReal::Zero(word_width, examples->size());
  for (size_t i = 0; i < examples->size(); ++i) {
    int word_id = examples->wordAt(i);
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

void Weights::estimateGradient(
    const boost::shared_ptr<DataSet>& examples,
    const boost::shared_ptr<Weights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  vector<vector<int>> contexts;
  vector<WordsList> features;
  vector<MatrixReal> context_vectors;
  getContextVectors(examples, contexts, features, context_vectors);

  setContextWords(contexts, words);
  setFeatureWords(features, words);

  MatrixReal prediction_vectors =
      getPredictionVectors(examples->size(), context_vectors);

  MatrixReal weighted_representations;
  estimateProjectionGradient(
      examples, prediction_vectors, gradient,
      weighted_representations, objective, words);

  weighted_representations.array() *= activationDerivative(config->activation, prediction_vectors);

  getContextGradient(
      examples->size(), contexts, features, context_vectors, weighted_representations, gradient);
}

void Weights::syncUpdate(
    const MinibatchWords& words,
    const boost::shared_ptr<Weights>& gradient) {
  for (int word_id: words.getFeatureWordsSet()) {
    lock_guard<mutex> lock(*mutexesP[word_id]);
    P.col(word_id) += gradient->P.col(word_id);
  }

  for (int word_id: words.getContextWordsSet()) {
    lock_guard<mutex> lock(*mutexesQ[word_id]);
    Q.col(word_id) += gradient->Q.col(word_id);
  }

  for (int word_id: words.getOutputWordsSet()) {
    lock_guard<mutex> lock(*mutexesR[word_id]);
    R.col(word_id) += gradient->R.col(word_id);
  }

  for (int i = 0; i < C.size(); ++i) {
    lock_guard<mutex> lock(*mutexesC[i]);
    C[i] += gradient->C[i];
  }

  lock_guard<mutex> lock(*mutexB);
  B += gradient->B;
}

Block Weights::getBlock(int start, int size) const {
  int thread_id = omp_get_thread_num();
  size_t block_size = size / config->threads + 1;
  size_t block_start = start + thread_id * block_size;
  block_size = min(block_size, start + size - block_start);

  return make_pair(block_start, block_size);
}

void Weights::updateSquared(
    const MinibatchWords& global_words,
    const boost::shared_ptr<Weights>& global_gradient) {
  for (int word_id: global_words.getFeatureWords()) {
    P.col(word_id).array() += global_gradient->P.col(word_id).array().square();
  }

  for (int word_id: global_words.getContextWords()) {
    Q.col(word_id).array() += global_gradient->Q.col(word_id).array().square();
  }

  for (int word_id: global_words.getOutputWords()) {
    R.col(word_id).array() += global_gradient->R.col(word_id).array().square();
  }

  Block block = getBlock(P.size() + Q.size() + R.size(), W.size() - (P.size() + Q.size() + R.size()));
  W.segment(block.first, block.second).array() +=
      global_gradient->W.segment(block.first, block.second).array().square();
}

void Weights::updateAdaGrad(
    const MinibatchWords& global_words,
    const boost::shared_ptr<Weights>& global_gradient,
    const boost::shared_ptr<Weights>& adagrad) {
  for (int word_id: global_words.getFeatureWords()) {
    P.col(word_id) -= global_gradient->P.col(word_id).binaryExpr(
        adagrad->P.col(word_id), CwiseAdagradUpdateOp<Real>(config->step_size));
  }

  for (int word_id: global_words.getContextWords()) {
    Q.col(word_id) -= global_gradient->Q.col(word_id).binaryExpr(
        adagrad->Q.col(word_id), CwiseAdagradUpdateOp<Real>(config->step_size));
  }

  for (int word_id: global_words.getOutputWords()) {
    R.col(word_id) -= global_gradient->R.col(word_id).binaryExpr(
        adagrad->R.col(word_id), CwiseAdagradUpdateOp<Real>(config->step_size));
  }

  Block block = getBlock(P.size() + Q.size() + R.size(), W.size() - (P.size() + Q.size() + R.size()));
  W.segment(block.first, block.second) -=
      global_gradient->W.segment(block.first, block.second).binaryExpr(
          adagrad->W.segment(block.first, block.second),
          CwiseAdagradUpdateOp<Real>(config->step_size));
}

Real Weights::regularizerUpdate(
    const boost::shared_ptr<Weights>& global_gradient,
    Real minibatch_factor) {
  Real sigma = minibatch_factor * config->step_size * config->l2_lbl;
  Block block = getBlock(0, W.size());
  W.segment(block.first, block.second) -=
      W.segment(block.first, block.second) * sigma;

  Real sum = W.segment(block.first, block.second).array().square().sum();
  return 0.5 * minibatch_factor * config->l2_lbl * sum;
}

void Weights::clear(const MinibatchWords& words, bool parallel_update) {
  if (parallel_update) {
    for (int word_id: words.getFeatureWords()) {
      P.col(word_id).setZero();
    }

    for (int word_id: words.getContextWords()) {
      Q.col(word_id).setZero();
    }

    for (int word_id: words.getOutputWords()) {
      R.col(word_id).setZero();
    }

    Block block = getBlock(P.size() + Q.size() + R.size(), W.size() - (P.size() + Q.size() + R.size()));
    W.segment(block.first, block.second).setZero();
  } else {
    for (int word_id: words.getFeatureWordsSet()) {
      P.col(word_id).setZero();
    }

    for (int word_id: words.getContextWordsSet()) {
      Q.col(word_id).setZero();
    }

    for (int word_id: words.getOutputWordsSet()) {
      R.col(word_id).setZero();
    }

    W.segment(P.size() + Q.size() + R.size(), W.size() - (P.size() + Q.size() + R.size())).setZero();
  }
}

VectorReal Weights::getPredictionVector(const Context& context) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->representation_size;

  VectorReal prediction_vector = VectorReal::Zero(word_width);
  for (int i = 0; i < context_width; ++i) {
    VectorReal in_vector = VectorReal::Zero(word_width);
    if (!config->compositional || config->pos_annotated) 
      in_vector = Q.col(context.words[i]); //.array();
    if (config->compositional) {
      for (auto feat: context.features[i]) 
        in_vector += P.col(feat);
    } 
    if (config->diagonal_contexts) {
      //prediction_vector += C[i].array() * in_vector.array(); 
      prediction_vector += C[i].asDiagonal() * in_vector; 
    } else {
      prediction_vector += C[i] * in_vector;
    }
  }

  return applyActivation<VectorReal>(config->activation, prediction_vector);
}

Real Weights::predict(int word, Context context) const {
  VectorReal prediction_vector = getPredictionVector(context);
  Real prob = 0;

  //auto ret = normalizerCache.get(context.words);
  //if (ret.second) {
  //  prob = (R.col(word).dot(prediction_vector) + B(word) - ret.first);
  //} else {  
    Real normalizer = 0;
    VectorReal word_probs = logSoftMax(
        R.transpose() * prediction_vector + B, normalizer);
    //normalizerCache.set(context.words, normalizer);
    prob = word_probs(word);
  //}

  //return negative log likelihood
  return -prob;
}

Reals Weights::predict(Context context) const {
  VectorReal prediction_vector = getPredictionVector(context);
  Reals probs(vocabSize(), 0);

  Real normalizer = 0;
  VectorReal word_probs = logSoftMax(
      R.transpose() * prediction_vector + B, normalizer);
  //normalizerCache.set(context.words, normalizer);
  for (int i = 0; i < vocabSize(); ++i)
    probs[i] = -word_probs(i);
  
  return probs;
}

int Weights::vocabSize() const {
  return config->vocab_size;
}

void Weights::clearCache() {
  normalizerCache.clear();
}

MatrixReal Weights::getWordVectors() const {
  return R; //why not Q?
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
