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
  cout << "  Context vocab size = " << config->num_features << endl;
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
  int num_sentence_vectors = 1;
  if (config->sentence_vector)
    num_sentence_vectors += config->num_train_sentences;
    
  int num_context_words = config->num_features;
  int num_output_words = config->vocab_size;
  //if (config->output_compositional)
  //  num_output_words = config->num_features;
  int word_width = config->representation_size;
  int context_width = config->ngram_order - 1;

  int P_size = word_width * num_sentence_vectors;
  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_output_words;

  size = P_size + Q_size + R_size + context_width * C_size + B_size;
  data = new Real[size];

  for (int i = 0; i < num_sentence_vectors; ++i) {
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
  int num_sentence_vectors = 1;
  if (config->sentence_vector)
    num_sentence_vectors += config->num_train_sentences;
  int num_context_words = config->num_features;
  int num_output_words = config->vocab_size;
  //if (config->output_compositional)
  //  num_output_words = config->num_features;
  int word_width = config->representation_size;
  int context_width = config->ngram_order - 1;

  int P_size = word_width * num_sentence_vectors;
  int Q_size = word_width * num_context_words;
  int R_size = word_width * num_output_words;
  int C_size = config->diagonal_contexts ? word_width : word_width * word_width;
  int B_size = num_output_words;

  new (&W) WeightsType(data, size);

  new (&P) WordVectorsType(data, word_width, num_sentence_vectors);
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
      weighted_representations, word_probs, gradient, words, sentences_only);
}

void Weights::getContextVectors(
    const boost::shared_ptr<DataSet>& examples,
    vector<WordsList>& contexts,
    vector<MatrixReal>& context_vectors) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->representation_size;

  MT19937 eng;
  contexts.resize(examples->size());
  context_vectors.resize(
      context_width, MatrixReal::Zero(word_width, examples->size()));
  //std::cout << examples->size() << std::endl;
  for (size_t i = 0; i < examples->size(); ++i) {
    contexts[i] = examples->contextAt(i).features;
    //std::cout << "i " << contexts[i].size() << std::endl;
    
    if (config->whole_feature_dropout > 0) {
      for (int j = 0; (j < context_width - 1) || (!config->sentence_vector && (j == context_width - 1)); ++j) {
       for (int k = 0; k < contexts[i][j].size(); ++k) {
         if (sample_uniform01<Real, MT19937> (eng) < config->whole_feature_dropout) {
           contexts[i][j].erase(contexts[i][j].begin() + k);
           --k;
         }
       }
      }
    }

    MatrixReal dropout_mask = MatrixReal::Zero(word_width, word_width);
    if (config->feature_dropout > 0) {
      for (int k = 0; k < word_width; ++k) {
        if (sample_uniform01<Real, MT19937> (eng) >= config->feature_dropout) {
          dropout_mask(k, k) = 1.0;
        }
      }
    }

    //contexts[i].push_back(examples->contextAt(i).features[j]);

    for (int j = 0; j < context_width; ++j) {
      VectorReal feature_vector = VectorReal::Zero(word_width);

      if (config->sentence_vector && (j == context_width - 1)) {
        //context_vectors[j].col(i) += P.col(contexts[i][j][0]); 
        feature_vector += P.col(contexts[i][j][0]); 
      } else {   
        for (auto feat: contexts[i][j]) {
          //context_vectors[j].col(i) += Q.col(feat);
          feature_vector += Q.col(feat);
        }
      }

      if (config->feature_dropout > 0) 
        feature_vector = dropout_mask * feature_vector;

      context_vectors[j].col(i) += feature_vector;
    }
  }
}

void Weights::setContextWords(
    const vector<WordsList>& contexts,
    MinibatchWords& words) const {
  for (const auto& context: contexts) {
   for (int i = 0; i < context.size(); ++i) {
      if (config->sentence_vector && (i == context.size() - 1))
        words.addSentenceWord(context[i][0]);
      else {
        for (int word_id: context[i]) {
          words.addContextWord(word_id);
        }
      }
    }
  }
}

//During training
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
    //std::cout << C[index].size() << " " << representations.col(0).size() << std::endl;
    return C[index].asDiagonal() * representations;
  } else {
    if (transpose) {
      //std::cout << C[index].col(0).size() << " " << representations.col(0).size() << std::endl;
      return C[index].transpose() * representations;
    } else {
      //std::cout << C[index].row(0).size() << " " << representations.col(0).size() << std::endl;
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
    const vector<WordsList>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& prediction_vectors,
    const MatrixReal& weighted_representations,
    MatrixReal& word_probs,
    const boost::shared_ptr<Weights>& gradient,
    MinibatchWords& words,
    bool sentences_only) const {
  for (size_t i = 0; i < examples->size(); ++i) {
    word_probs(examples->wordAt(i), i) -= 1;
  }

  if (!sentences_only) {
    for (size_t word_id = 0; word_id < config->vocab_size; ++word_id) {
      words.addOutputWord(word_id);
    }

    gradient->R += prediction_vectors * word_probs.transpose();
    gradient->B += word_probs.rowwise().sum();
  }

  getContextGradient(
      examples->size(), contexts, context_vectors, weighted_representations, gradient, sentences_only);
}

void Weights::getContextGradient(
    size_t prediction_size,
    const vector<WordsList>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& weighted_representations,
    const boost::shared_ptr<Weights>& gradient,
    bool sentences_only) const {
  int context_width = config->ngram_order - 1;

  int word_width = config->representation_size;
  MatrixReal context_gradients = MatrixReal::Zero(word_width, prediction_size);
  for (int j = 0; j < context_width; ++j) {
    //std::cout << j << ": " << prediction_size << std::endl;
    context_gradients = getContextProduct(j, weighted_representations, true);
    for (size_t i = 0; i < prediction_size; ++i) {
      if (config->sentence_vector && (j == context_width - 1)) {
        gradient->P.col(contexts[i][j][0]) += context_gradients.col(i); 
      } else if (!sentences_only) {
        for (auto feat: contexts[i][j]) {
          gradient->Q.col(feat) += context_gradients.col(i);
        }
      }
    }
    
    //TODO maybe change back?
    //if (!sentences_only) {
      if (config->diagonal_contexts) {
        gradient->C[j] += context_vectors[j].cwiseProduct(weighted_representations).rowwise().sum();
      } else {
        gradient->C[j] += weighted_representations * context_vectors[j].transpose();
      }
    //}
  }
}

bool Weights::checkGradient(
    const boost::shared_ptr<DataSet>& examples,
    const boost::shared_ptr<Weights>& gradient,
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

Real Weights::getObjective(
    const boost::shared_ptr<DataSet>& examples) const {
  vector<WordsList> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal word_probs;
  return getObjective(
      examples, contexts, context_vectors, prediction_vectors, word_probs);
}

Real Weights::getObjective(
    const boost::shared_ptr<DataSet>& examples,
    vector<WordsList>& contexts,
    vector<MatrixReal>& context_vectors,
    MatrixReal& prediction_vectors,
    MatrixReal& word_probs) const {
  getContextVectors(examples, contexts, context_vectors);
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
  vector<WordsList> contexts;
  vector<MatrixReal> context_vectors;
  getContextVectors(examples, contexts, context_vectors);

  setContextWords(contexts, words);

  MatrixReal prediction_vectors =
      getPredictionVectors(examples->size(), context_vectors);

  MatrixReal weighted_representations;
  estimateProjectionGradient(
      examples, prediction_vectors, gradient,
      weighted_representations, objective, words);

  weighted_representations.array() *= activationDerivative(config->activation, prediction_vectors);

  getContextGradient(
      examples->size(), contexts, context_vectors, weighted_representations, gradient);
}

void Weights::syncUpdate(
    const MinibatchWords& words,
    const boost::shared_ptr<Weights>& gradient,
    bool sentences_only) {
  for (int word_id: words.getSentenceWordsSet()) {
    lock_guard<mutex> lock(*mutexesP[word_id]);
    P.col(word_id) += gradient->P.col(word_id);
  }

  if (!sentences_only) {
    for (int word_id: words.getContextWordsSet()) {
      lock_guard<mutex> lock(*mutexesQ[word_id]);
      Q.col(word_id) += gradient->Q.col(word_id);
    }

    for (int i = 0; i < C.size(); ++i) {
      lock_guard<mutex> lock(*mutexesC[i]);
      C[i] += gradient->C[i];
    }

    for (int word_id: words.getOutputWordsSet()) {
      lock_guard<mutex> lock(*mutexesR[word_id]);
      R.col(word_id) += gradient->R.col(word_id);
    }

    lock_guard<mutex> lock(*mutexB);
    B += gradient->B;
  }
}

Block Weights::getBlock(int start, int size) const {
  int thread_id = omp_get_thread_num();
  size_t block_size = size / config->threads + 1;
  size_t block_start = start + thread_id * block_size;
  block_size = min(block_size, start + size - block_start);

  return make_pair(block_start, block_size);
}

void Weights::updateSentenceVectorGradient(const VectorReal& sentence_vector_gradient) {
  VectorReal adagrad = sentence_vector_gradient.array().square();
  P.col(0) -= sentence_vector_gradient.binaryExpr(adagrad, CwiseAdagradUpdateOp<Real>(config->step_size));
  
  //regularize
  Real sigma = config->step_size * config->l2_lbl;
  P.col(0) -= P.col(0)*sigma;
}

void Weights::updateSquared(
    const MinibatchWords& global_words,
    const boost::shared_ptr<Weights>& global_gradient,
    bool sentences_only) {
  for (int word_id: global_words.getSentenceWords()) {
    if (config->rms_prop) {
      //P.col(word_id).array() = P.col(word_id).array()*0.9 + global_gradient->P.col(word_id).array().square()*0.1;
      P.col(word_id).array() *= 0.9;
      P.col(word_id).array() += global_gradient->P.col(word_id).array().square()*0.1;
    } else
      P.col(word_id).array() += global_gradient->P.col(word_id).array().square();

  }

  /* if (sentences_only) {
    int C_size = config->diagonal_contexts ? config->representation_size : config->representation_size * config->representation_size; 
  
    Block block = getBlock(P.size() + Q.size() + R.size(), (config->ngram_order - 1)*C_size);
    W.segment(block.first, block.second).array() +=
        global_gradient->W.segment(block.first, block.second).array().square(); 
    
  } else { */
  if (!sentences_only) {
    for (int word_id: global_words.getContextWords()) {
      if (config->rms_prop) {
        //Q.col(word_id).array() = Q.col(word_id).array()*0.9 + global_gradient->Q.col(word_id).array().square()*0.1;
        Q.col(word_id).array() *= 0.9;
        Q.col(word_id).array() += global_gradient->Q.col(word_id).array().square()*0.1;
      } else
        Q.col(word_id).array() += global_gradient->Q.col(word_id).array().square();
    }

    for (int word_id: global_words.getOutputWords()) {
      if (config->rms_prop) {
        //R.col(word_id).array() = R.col(word_id).array()*0.9 + global_gradient->R.col(word_id).array().square()*0.1;
        R.col(word_id).array() *= 0.9;
        R.col(word_id).array() += global_gradient->R.col(word_id).array().square()*0.1;
      } else 
        R.col(word_id).array() += global_gradient->R.col(word_id).array().square();
    }

    Block block = getBlock(P.size() + Q.size() + R.size(), W.size() - (P.size() + Q.size() + R.size()));
    if (config->rms_prop) {
      //W.segment(block.first, block.second).array() = W.segment(block.first, block.second).array()*0.9
      //  + global_gradient->W.segment(block.first, block.second).array().square()*0.1;
      W.segment(block.first, block.second).array() *= 0.9;
      W.segment(block.first, block.second).array() += global_gradient->W.segment(block.first, block.second).array().square()*0.1;
    } else
      W.segment(block.first, block.second).array() +=
        global_gradient->W.segment(block.first, block.second).array().square();
  }
}

void Weights::updateAdaGrad(
    const MinibatchWords& global_words,
    const boost::shared_ptr<Weights>& global_gradient,
    const boost::shared_ptr<Weights>& adagrad,
    bool sentences_only) {
  for (int word_id: global_words.getSentenceWords()) {
    P.col(word_id) -= global_gradient->P.col(word_id).binaryExpr(
        adagrad->P.col(word_id), CwiseAdagradUpdateOp<Real>(config->step_size));
  }

  /*if (sentences_only) {
    int C_size = config->diagonal_contexts ? config->representation_size : config->representation_size * config->representation_size; 
  
    Block block = getBlock(P.size() + Q.size() + R.size(), (config->ngram_order - 1)*C_size);
    W.segment(block.first, block.second) -=
        global_gradient->W.segment(block.first, block.second).binaryExpr(
            adagrad->W.segment(block.first, block.second),
            CwiseAdagradUpdateOp<Real>(config->step_size));
  } else {  */
  if (!sentences_only) {
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
}

Real Weights::regularizerUpdate(
    const MinibatchWords& global_words,
    const boost::shared_ptr<Weights>& global_gradient,
    Real minibatch_factor,
    bool sentences_only) {
  Real sigma = minibatch_factor * config->step_size * config->l2_lbl_sv;
  Real sum = 0;
  for (int word_id: global_words.getSentenceWords()) {
    P.col(word_id) -= P.col(word_id) * sigma;
    sum += P.col(word_id).array().square().sum();
  }
  
  //Block block = getBlock(0, P.size());
  //W.segment(block.first, block.second) -=
  //    W.segment(block.first, block.second) * sigma;
  //Real sum = W.segment(block.first, block.second).array().square().sum();
  Real total = 0.5 * minibatch_factor * config->l2_lbl_sv * sum; 
  
  sigma = minibatch_factor * config->step_size * config->l2_lbl;
  if (sentences_only) {
    /*int C_size = config->diagonal_contexts ? config->representation_size : config->representation_size * config->representation_size; 
    block = getBlock(P.size() + Q.size() + R.size(), (config->ngram_order - 1)*C_size);
    W.segment(block.first, block.second) -=
        W.segment(block.first, block.second) * sigma;

    sum = W.segment(block.first, block.second).array().square().sum();
    return total + 0.5 * minibatch_factor * config->l2_lbl * sum; */
    return total;
  } else {
    Block block = getBlock(P.size(), W.size() - P.size());
    W.segment(block.first, block.second) -=
        W.segment(block.first, block.second) * sigma;

    sum = W.segment(block.first, block.second).array().square().sum();
    return total + 0.5 * minibatch_factor * config->l2_lbl * sum;
  }
}

void Weights::resetSentenceVector() {
  P.col(0).setZero();
  mt19937 gen(1);
  normal_distribution<Real> gaussian(0, 0.1);
  for (int i = 0; i < config->representation_size; ++i) {
    P(i, 0) = gaussian(gen);
  }
}

void Weights::clear(const MinibatchWords& words, bool parallel_update) {
  if (parallel_update) {
    for (int word_id: words.getSentenceWords()) {
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
    for (int word_id: words.getSentenceWordsSet()) {
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

//At test time
VectorReal Weights::getPredictionVector(const Context& context) const {
  int context_width = config->ngram_order - 1;
  int word_width = config->representation_size;
//  if (config->sentence_vector)
//     contexts[i].push_back(Words(1, examples->sentenceIdAt(i)));

  VectorReal prediction_vector = VectorReal::Zero(word_width);
  for (int i = 0; i < context_width; ++i) {
    VectorReal in_vector = VectorReal::Zero(word_width);
    if (config->sentence_vector && i == (context_width - 1)) {
      //std::cout << context.features[i].size() << std::endl;
      //std::cout << context.features[i][0] << std::endl;
      if (config->feature_dropout > 0) 
        in_vector += P.col(context.features[i][0])*(1.0/(1 - config->feature_dropout)); 
      else
        in_vector += P.col(context.features[i][0]); //0 
    } else {
      for (auto feat: context.features[i]) {
        if ((config->feature_dropout > 0) || (config->whole_feature_dropout > 0))
          in_vector += Q.col(feat)*(1.0/(1 - ((1 - config->whole_feature_dropout)*config->feature_dropout + config->whole_feature_dropout))); 
        else
          in_vector += Q.col(feat);
      }
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

Reals Weights::predictViterbi(Context context) const {
  VectorReal prediction_vector = getPredictionVector(context);
  Reals probs(vocabSize(), std::numeric_limits<Real>::max());

  //Real normalizer = 0;
  VectorReal word_probs = R.transpose() * prediction_vector + B; //logSoftMax()
  //normalizerCache.set(context.words, normalizer);
  int word_id = word_probs.maxCoeff();
  probs[word_id] = -log(word_probs(word_id));
  
  return probs;
}

int Weights::vocabSize() const {
  return config->vocab_size;
}

void Weights::clearCache() {
  normalizerCache.clear();
}

MatrixReal Weights::getWordVectors() const {
  return R; 
}

MatrixReal Weights::getFeatureVectors() const {
  return Q; 
}

MatrixReal Weights::getSentenceVectors() const {
  std::cout << P.cols() << std::endl;
  return P; 
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
