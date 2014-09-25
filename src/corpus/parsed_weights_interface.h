#ifndef _CORPUS_PWEIGHTS_I_H_
#define _CORPUS_PWEIGHTS_I_H_

#include "corpus/weights_interface.h"

namespace oxlm {

class ParsedWeightsInterface: public WeightsInterface {
  public:
  virtual Real predictWord(int word, std::vector<int> context) const = 0;

  virtual Real predictTag(int tag, std::vector<int> context) const = 0;
  
  virtual Real predictAction(int action, std::vector<int> context) const = 0;

  virtual int numWords() const = 0;
  
  virtual int numTags() const = 0;

  virtual int numActions() const = 0;

  virtual ~ParsedWeightsInterface() {}
};


}

#endif
