#ifndef _CORPUS_PWEIGHTS_I_H_
#define _CORPUS_PWEIGHTS_I_H_

#include "weights_interface.h"

namespace oxlm {

class ParsedWeightsInterface: public WeightsInterface {

  virtual double predict_word(int word, vector<int> context) const = 0;

  virtual double predict_tag(int tag, vector<int> context) const = 0;
  
  virtual double predict_action(int action, vector<int> context) const = 0;
};


}

#endif
