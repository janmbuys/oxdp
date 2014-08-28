#ifndef _CORPUS_PWEIGHTS_I_H_
#define _CORPUS_PWEIGHTS_I_H_

#include "weights_interface.h"

namespace oxlm {

class ParsedWeightsInterface: public WeightsInterface {

  virtual double predictWord(int word, vector<int> context) const = 0;

  virtual double predictTag(int tag, vector<int> context) const = 0;
  
  virtual double predictAction(int action, vector<int> context) const = 0;
};


}

#endif
