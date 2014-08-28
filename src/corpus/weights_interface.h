#ifndef _CORPUS_WEIGHTS_I_H_
#define _CORPUS_WEIGHTS_I_H_

namespace oxlm {

class WeightsInterface {

  virtual double predict(int word, vector<int> context) const = 0;

};


}

#endif
