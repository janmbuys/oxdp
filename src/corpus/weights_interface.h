#ifndef _CORPUS_WEIGHTS_I_H_
#define _CORPUS_WEIGHTS_I_H_

namespace oxlm {

class WeightsInterface {

  //use negative log probabilities for all predict functions
  virtual double predict(int word, vector<int> context) const = 0;

};


}

#endif
