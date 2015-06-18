#include "lbl/discriminative_weights.h"

#include <iomanip>
#include <random>

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/operators.h"

namespace oxlm {

DiscriminativeWeights::DiscriminativeWeights() : data(NULL), P(0, 0, 0), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {}

DiscriminativeWeights::DiscriminativeWeights(
    const boost::shared_ptr<ModelConfig>& config, 
    const boost::shared_ptr<DiscriminativeMetadata>& metadata,
    bool init)
    : config(config), metadata(metadata),
      data(NULL), P(0, 0, 0), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  
  if (init) {
  // Initialize model weights randomly.
  mt19937 gen; //(1);
  normal_distribution<Real> gaussian(0, 0.1);
  for (int i = 0; i < size; ++i) {
    W(i) = gaussian(gen);
  }

  // Initialize bias with unigram probabilities.
  B = metadata->getSmoothedUnigram().array().log();

  cout << "===============================" << endl;
  cout << " Model parameters: " << endl;
  cout << "  Context vocab size = " << config->num_features << endl;
  cout << "  Output vocab size = " << config->numActions() << endl;
  cout << "  Total parameters = " << numParameters() << endl;
  cout << "===============================" << endl;
  } else {
    W.setZero();
  }
}

DiscriminativeWeights::DiscriminativeWeights(const DiscriminativeWeights& other)
    : config(other.config), metadata(other.metadata),
      data(NULL), P(0, 0, 0), Q(0, 0, 0), R(0, 0, 0), B(0, 0), W(0, 0) {
  allocate();
  memcpy(data, other.data, size * sizeof(Real));
}

void DiscriminativeWeights::allocate() {
  int num_context_words = config->num_features;
  int num_output_words = config->numActions();
  int word_width = config->representation_size;
  int context_width = config->ngram_order - 1;

  int P_size = word_width * 1;
  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_output_words;

  size = P_size + Q_size + R_size + context_width * C_size + B_size;
  data = new Real[size];

  for (int i = 0; i < 1; ++i) {
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

void DiscriminativeWeights::setModelParameters() {
  int num_context_words = config->num_features;
  int num_output_words = config->numActions();
  int word_width = config->representation_size;
  int context_width = config->ngram_order - 1;

  int P_size = word_width * 1;
  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_output_words;

  new (&W) WeightsType(data, size);

  new (&P) WordVectorsType(data, word_width, 1);
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

size_t DiscriminativeWeights::numParameters() const {
  return size;
}

Real DiscriminativeWeights::predictWord(int word, Context context) const {
  return 0.0;
}

Reals DiscriminativeWeights::predictWord(Context context) const {
  return Reals(numWords(), 0.0);
}
 
Reals DiscriminativeWeights::predictWordOverTags(int word, Context context) const {
  return Reals(numTags(), 0.0);
}
 
Real DiscriminativeWeights::predictTag(int tag, Context context) const {
  return 0.0;
}
 
Reals DiscriminativeWeights::predictTag(Context context) const {
  return Reals(numTags(), 0.0);
}
  
Real DiscriminativeWeights::predictAction(WordId action, Context context) const {
  return predict(action, context);
}

Reals DiscriminativeWeights::predictAction(Context context) const {
  return predict(context);
}

int DiscriminativeWeights::numWords() const {
  return 1;
}

int DiscriminativeWeights::numTags() const {
  return 1;
}

int DiscriminativeWeights::numActions() const {
  return config->numActions();
}
 
VectorReal DiscriminativeWeights::getSentenceVectorGradient(
      const boost::shared_ptr<ParseDataSet>& examples,
      Real& objective) const {
  return VectorReal::Zero(config->representation_size);
}

void DiscriminativeWeights::getGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<DiscriminativeWeights>& gradient,
    Real& objective,
    MinibatchWords& words,
    bool sentences_only) const {
  vector<WordsList> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal word_probs;
  objective += getObjective(examples,
      contexts, context_vectors, prediction_vectors, word_probs);

  setContextWords(contexts, words);

  MatrixReal weighted_representations = getWeightedRepresentations(
      examples, prediction_vectors, word_probs);

  getFullGradient(
      examples, contexts, context_vectors, prediction_vectors,
      weighted_representations, word_probs, gradient, words);
}

void DiscriminativeWeights::getContextVectors(
    const boost::shared_ptr<ParseDataSet>& examples,
    vector<WordsList>& contexts,
    vector<MatrixReal>& context_vectors) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->representation_size;

  contexts.resize(examples->action_example_size());
  context_vectors.resize(
      context_width, MatrixReal::Zero(word_width, examples->action_example_size()));
  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    contexts[i] = examples->action_context_at(i).features;
    for (int j = 0; j < context_width; ++j) {
      for (auto feat: contexts[i][j]) 
        context_vectors[j].col(i) += Q.col(feat);
    }  
  }
}

void DiscriminativeWeights::setContextWords(
    const vector<WordsList>& contexts,
    MinibatchWords& words) const {
  for (const auto& context: contexts) {
   for (const auto& item: context) {
      for (int word_id: item) {
        words.addContextWord(word_id);
      }
    }
  }
}

MatrixReal DiscriminativeWeights::getPredictionVectors(
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

MatrixReal DiscriminativeWeights::getContextProduct(
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

MatrixReal DiscriminativeWeights::getProbabilities(
    const boost::shared_ptr<ParseDataSet>& examples,
    const MatrixReal& prediction_vectors) const {
  MatrixReal word_probs = R.transpose() * prediction_vectors 
                          + B * MatrixReal::Ones(1, examples->action_example_size());
  for (size_t i = 0; i < examples->action_example_size(); ++i) {    
    word_probs.col(i) = logSoftMax(word_probs.col(i));
  }

  return word_probs;
}

MatrixReal DiscriminativeWeights::getWeightedRepresentations(
    const boost::shared_ptr<ParseDataSet>& examples,
    const MatrixReal& prediction_vectors,
    const MatrixReal& word_probs) const {
  MatrixReal weighted_representations = R * word_probs;

  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    weighted_representations.col(i) -= R.col(examples->action_at(i));
  }

  weighted_representations.array() *= activationDerivative(config->activation, prediction_vectors);
  return weighted_representations;
}

void DiscriminativeWeights::getFullGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const vector<WordsList>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& prediction_vectors,
    const MatrixReal& weighted_representations,
    MatrixReal& word_probs,
    const boost::shared_ptr<DiscriminativeWeights>& gradient,
    MinibatchWords& words) const {
  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    word_probs(examples->action_at(i), i) -= 1;
  }

  for (size_t word_id = 0; word_id < config->numActions(); ++word_id) {
    words.addOutputWord(word_id);
  }

  gradient->R += prediction_vectors * word_probs.transpose();
  gradient->B += word_probs.rowwise().sum();

  getContextGradient(
      examples->action_example_size(), contexts, context_vectors, weighted_representations, gradient);
}

void DiscriminativeWeights::getContextGradient(
    size_t prediction_size,
    const vector<WordsList>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& weighted_representations,
    const boost::shared_ptr<DiscriminativeWeights>& gradient) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->representation_size;
  MatrixReal context_gradients = MatrixReal::Zero(word_width, prediction_size);
  for (int j = 0; j < context_width; ++j) {
    context_gradients = getContextProduct(j, weighted_representations, true);
    for (size_t i = 0; i < prediction_size; ++i) {
      for (auto feat: contexts[i][j]) 
        gradient->Q.col(feat) += context_gradients.col(i);
    }

    if (config->diagonal_contexts) {
      gradient->C[j] += context_vectors[j].cwiseProduct(weighted_representations).rowwise().sum();
    } else {
      gradient->C[j] += weighted_representations * context_vectors[j].transpose();
    }
  }
  //std::cout << std::endl;
}

bool DiscriminativeWeights::checkGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<DiscriminativeWeights>& gradient,
    double eps) {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal word_probs;

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

Real DiscriminativeWeights::getObjective(
    const boost::shared_ptr<ParseDataSet>& examples) const {
  vector<WordsList> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal word_probs;
  return getObjective(
      examples, contexts, context_vectors, prediction_vectors, word_probs);
}

Real DiscriminativeWeights::getObjective(
    const boost::shared_ptr<ParseDataSet>& examples,
    vector<WordsList>& contexts,
    vector<MatrixReal>& context_vectors,
    MatrixReal& prediction_vectors,
    MatrixReal& word_probs) const {
  getContextVectors(examples, contexts, context_vectors);
  prediction_vectors = getPredictionVectors(examples->action_example_size(), context_vectors);
  
  word_probs = getProbabilities(examples, prediction_vectors);

  Real objective = 0;
  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    Real word_likelihood = -word_probs(examples->action_at(i), i);    
    objective += word_likelihood;
  }

  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    word_probs.col(i).array() = word_probs.col(i).array().exp();
  } 

  return objective;
}

vector<vector<int>> DiscriminativeWeights::getNoiseWords(
    const boost::shared_ptr<ParseDataSet>& examples) const {
  if (!wordDists.get()) {
    wordDists.reset(new WordDistributions(metadata->getUnigram()));
  }
    
  vector<vector<int>> noise_words(examples->action_example_size());
  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    for (int j = 0; j < config->noise_samples; ++j) {
      noise_words[i].push_back(wordDists->sample());
    }
  }

  return noise_words;
}

void DiscriminativeWeights::estimateProjectionGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const MatrixReal& prediction_vectors,
    const boost::shared_ptr<DiscriminativeWeights>& gradient,
    MatrixReal& weighted_representations,
    Real& objective,
    MinibatchWords& words) const {
  int noise_samples = config->noise_samples;
  int word_width = config->representation_size;
  VectorReal unigram = metadata->getUnigram();
  vector<vector<int>> noise_words = getNoiseWords(examples);

  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    words.addOutputWord(examples->action_at(i));
    for (int word_id: noise_words[i]) {
      words.addOutputWord(word_id);
    }
  }

  weighted_representations = MatrixReal::Zero(word_width, examples->action_example_size());
  for (size_t i = 0; i < examples->action_example_size(); ++i) {
    int word_id = examples->action_at(i);
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

void DiscriminativeWeights::estimateGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<DiscriminativeWeights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  vector<WordsList> contexts;
  vector<MatrixReal> context_vectors;
  getContextVectors(examples, contexts, context_vectors);

  setContextWords(contexts, words);

  MatrixReal prediction_vectors =
      getPredictionVectors(examples->action_example_size(), context_vectors);

  MatrixReal weighted_representations;
  estimateProjectionGradient(
      examples, prediction_vectors, gradient,
      weighted_representations, objective, words);

  weighted_representations.array() *= activationDerivative(config->activation, prediction_vectors);

  getContextGradient(
      examples->action_example_size(), contexts, context_vectors, weighted_representations, gradient);
}

void DiscriminativeWeights::syncUpdate(
    const MinibatchWords& words,
    const boost::shared_ptr<DiscriminativeWeights>& gradient,
    bool sentences_only) {

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

Block DiscriminativeWeights::getBlock(int start, int size) const {
  int thread_id = omp_get_thread_num();
  size_t block_size = size / config->threads + 1;
  size_t block_start = start + thread_id * block_size;
  block_size = min(block_size, start + size - block_start);

  return make_pair(block_start, block_size);
}

void DiscriminativeWeights::updateSentenceVectorGradient(const VectorReal& sentence_vector_gradient) {}

void DiscriminativeWeights::updateSquared(
    const MinibatchWords& global_words,
    const boost::shared_ptr<DiscriminativeWeights>& global_gradient,
    bool sentences_only) {

  for (int word_id: global_words.getContextWords()) {
    if (config->rms_prop)
      Q.col(word_id).array() = Q.col(word_id).array()*0.9 + global_gradient->Q.col(word_id).array().square()*0.1;
    else
      Q.col(word_id).array() += global_gradient->Q.col(word_id).array().square();
  }

  for (int word_id: global_words.getOutputWords()) {
    if (config->rms_prop)
      R.col(word_id).array() = R.col(word_id).array()*0.9 + global_gradient->R.col(word_id).array().square()*0.1;
    else 
      R.col(word_id).array() += global_gradient->R.col(word_id).array().square();
  }

  Block block = getBlock(P.size() + Q.size() + R.size(), W.size() - (P.size() + Q.size() + R.size()));
  if (config->rms_prop)
    W.segment(block.first, block.second).array() = W.segment(block.first, block.second).array()*0.9
      + global_gradient->W.segment(block.first, block.second).array().square()*0.1;
  else
    W.segment(block.first, block.second).array() +=
      global_gradient->W.segment(block.first, block.second).array().square();
}

void DiscriminativeWeights::updateAdaGrad(
    const MinibatchWords& global_words,
    const boost::shared_ptr<DiscriminativeWeights>& global_gradient,
    const boost::shared_ptr<DiscriminativeWeights>& adagrad,
    bool sentences_only) {
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

Real DiscriminativeWeights::regularizerUpdate(
    const MinibatchWords& global_words,
    const boost::shared_ptr<DiscriminativeWeights>& global_gradient,
    Real minibatch_factor,
    bool sentences_only) {
  Real sigma = minibatch_factor * config->step_size * config->l2_lbl;
  Block block = getBlock(0, W.size());
  W.segment(block.first, block.second) -=
      W.segment(block.first, block.second) * sigma;

  Real sum = W.segment(block.first, block.second).array().square().sum();
  return 0.5 * minibatch_factor * config->l2_lbl * sum;
}

void DiscriminativeWeights::clear(const MinibatchWords& words, bool parallel_update) {
  if (parallel_update) {

    for (int word_id: words.getContextWords()) {
      Q.col(word_id).setZero();
    }

    for (int word_id: words.getOutputWords()) {
      R.col(word_id).setZero();
    }

    Block block = getBlock(P.size() + Q.size() + R.size(), W.size() - (P.size() + Q.size() + R.size()));
    W.segment(block.first, block.second).setZero();
  } else {

    for (int word_id: words.getContextWordsSet()) {
      Q.col(word_id).setZero();
    }

    for (int word_id: words.getOutputWordsSet()) {
      R.col(word_id).setZero();
    }

    W.segment(P.size() + Q.size() + R.size(), W.size() - (P.size() + Q.size() + R.size())).setZero();
  }
}

VectorReal DiscriminativeWeights::getPredictionVector(const Context& context) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->representation_size;

  VectorReal prediction_vector = VectorReal::Zero(word_width);
  for (int i = 0; i < context_width; ++i) {
    VectorReal in_vector = VectorReal::Zero(word_width);
    for (auto feat: context.features[i]) 
      in_vector += Q.col(feat);
     
    if (config->diagonal_contexts) {
      //prediction_vector += C[i].array() * in_vector.array(); 
      prediction_vector += C[i].asDiagonal() * in_vector; 
    } else {
      prediction_vector += C[i] * in_vector;
    }
  }

  return applyActivation<VectorReal>(config->activation, prediction_vector);
}

Real DiscriminativeWeights::predict(int word, Context context) const {
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

Reals DiscriminativeWeights::predict(Context context) const {
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

void DiscriminativeWeights::resetSentenceVector() {}

int DiscriminativeWeights::vocabSize() const {
  return config->numActions();
}

void DiscriminativeWeights::clearCache() {
  normalizerCache.clear();
}

MatrixReal DiscriminativeWeights::getWordVectors() const {
  return R; //why not Q?
}

MatrixReal DiscriminativeWeights::getFeatureVectors() const {
  return Q; 
}

MatrixReal DiscriminativeWeights::getSentenceVectors() const {
  return MatrixReal(); 
}

bool DiscriminativeWeights::operator==(const DiscriminativeWeights& other) const {
  return *config == *other.config
      && *metadata == *other.metadata
      && size == other.size
      && W == other.W;
}

DiscriminativeWeights::~DiscriminativeWeights() {
  delete data;
}

} // namespace oxlm
