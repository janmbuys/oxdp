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
  return FactoredWeights::predict(word, context);
}

Real ParsedFactoredWeights::predictTag(int tag, Words context) const {
  return 1.0;
}
  
Real ParsedFactoredWeights::predictAction(int action, Words context) const {
  //TODO
  return 1.0/numActions();
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
 
} //namespace oxlm
