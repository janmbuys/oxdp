#include "lbl/parsed_factored_weights.h"

#include <iomanip>

#include <boost/make_shared.hpp>

#include "lbl/operators.h"

namespace oxlm {

ParsedFactoredWeights::ParsedFactoredWeights()
    : data(NULL) {}

ParsedFactoredWeights::ParsedFactoredWeights(
    const boost::shared_ptr<ModelConfig>& config,
    const boost::shared_ptr<FactoredMetadata>& metadata,
    bool init)
    : FactoredWeights(config, metadata, init), 
 //   metadata(metadata), index(metadata->getIndex()),  //for now, don't have own metadata and index
      data(NULL) {
  allocate();

  if (init) {
    // Parameter Initialization
  } else {

  }
}

ParsedFactoredWeights::ParsedFactoredWeights(const ParsedFactoredWeights& other)
    : FactoredWeights(other), data(NULL) {
  allocate();
  if (size > 0)
    memcpy(data, other.data, size * sizeof(Real));  
}

size_t ParsedFactoredWeights::numParameters() const {
  return FactoredWeights::numParameters() + size;
}

void ParsedFactoredWeights::allocate() {
  //set vector sizes
  
  size = 0;
  if (size > 0)
    data = new Real[size]; 

  for (int i = 0; i < config->threads; ++i) {
    mutexes.push_back(boost::make_shared<mutex>());
  }

  setModelParameters();
}

void ParsedFactoredWeights::setModelParameters() {
  //new the model parameters
}

Real ParsedFactoredWeights::predictWord(int word, Words context) const {
  std::cout << "context: " << std::endl;
  std::cout << context.size() << std::endl;
  return FactoredWeights::predict(word, context);
}

Real ParsedFactoredWeights::predictTag(int tag, Words context) const {
  return 0.0;
}
  
Real ParsedFactoredWeights::predictAction(int action, Words context) const {
  //TODO
  return -std::log(1.0/numActions());
}

int ParsedFactoredWeights::numWords() const {
  return FactoredWeights::vocabSize();
}

int ParsedFactoredWeights::numTags() const {
  return 1;
}

int ParsedFactoredWeights::numActions() const {
  return 3;
}
 
void ParsedFactoredWeights::getGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<FactoredWeights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  FactoredWeights::getGradient(examples->word_examples(), gradient, objective, words);
}

Real ParsedFactoredWeights::getObjective(
    const boost::shared_ptr<ParseDataSet>& examples) const {
  FactoredWeights::getObjective(examples->word_examples());
}

bool ParsedFactoredWeights::checkGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<FactoredWeights>& gradient,
    double eps) {
  FactoredWeights::checkGradient(examples->word_examples(), gradient, eps);
}

void ParsedFactoredWeights::estimateGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const boost::shared_ptr<FactoredWeights>& gradient,
    Real& objective,
    MinibatchWords& words) const {
  FactoredWeights::estimateGradient(examples->word_examples(), gradient, objective, words);
}

Real ParsedFactoredWeights::getObjective(
    const boost::shared_ptr<ParseDataSet>& examples,
    vector<vector<int>>& contexts,
    vector<MatrixReal>& context_vectors,
    MatrixReal& prediction_vectors,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs) const {
  FactoredWeights::getObjective(examples->word_examples(), contexts, context_vectors, prediction_vectors, class_probs, word_probs);
}

void ParsedFactoredWeights::getProbabilities(
    const boost::shared_ptr<ParseDataSet>& examples,
    const vector<vector<int>>& contexts,
    const MatrixReal& prediction_vectors,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs) const {
  FactoredWeights::getProbabilities(examples->word_examples(), contexts, prediction_vectors, class_probs, word_probs);
}

MatrixReal ParsedFactoredWeights::getWeightedRepresentations(
    const boost::shared_ptr<ParseDataSet>& examples,
    const MatrixReal& prediction_vectors,
    const MatrixReal& class_probs,
    const vector<VectorReal>& word_probs) const {
  FactoredWeights::getWeightedRepresentations(examples->word_examples(), prediction_vectors, class_probs, word_probs);
}

void ParsedFactoredWeights::getFullGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& prediction_vectors,
    const MatrixReal& weighted_representations,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs,
    const boost::shared_ptr<FactoredWeights>& gradient,
    MinibatchWords& words) const {
  FactoredWeights::getFullGradient(examples->word_examples(), contexts, context_vectors, prediction_vectors, weighted_representations, class_probs, word_probs, gradient, words);
}

std::vector<Words> ParsedFactoredWeights::getNoiseWords(
    const boost::shared_ptr<ParseDataSet>& examples) const {
  FactoredWeights::getNoiseWords(examples->word_examples());
}

void ParsedFactoredWeights::estimateProjectionGradient(
    const boost::shared_ptr<ParseDataSet>& examples,
    const MatrixReal& prediction_vectors,
    const boost::shared_ptr<FactoredWeights>& gradient,
    MatrixReal& weighted_representations,
    Real& objective,
    MinibatchWords& words) const {
  FactoredWeights::estimateProjectionGradient(examples->word_examples(), prediction_vectors, gradient, weighted_representations, objective, words);
}

} //namespace oxlm
