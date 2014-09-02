#ifndef _CORPUS_PWEIGHTS_I_H_
#define _CORPUS_PWEIGHTS_I_H_

#include "corpus/weights_interface.h"

namespace oxlm {

class ParsedWeightsInterface: public WeightsInterface {
  public:
  virtual double predictWord(int word, std::vector<int> context) const = 0;

  virtual double predictTag(int tag, std::vector<int> context) const = 0;
  
  virtual double predictAction(int action, std::vector<int> context) const = 0;

  virtual size_t numWords() const = 0;
  
  virtual size_t numTags() const = 0;

  virtual size_t numActions() const = 0;

  virtual ~ParsedWeightsInterface() {}
};


}

#endif
