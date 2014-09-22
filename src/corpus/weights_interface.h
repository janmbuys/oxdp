 #ifndef _CORPUS_WEIGHTS_I_H_
 #define _CORPUS_WEIGHTS_I_H_

#include <vector>
#include <cstdlib>

namespace oxlm {

class WeightsInterface {

  public:

  //use negative log probabilities for all predict functions

  virtual Real predict(int word, std::vector<int> context) const = 0;

  virtual int vocabSize() const = 0;

  virtual ~WeightsInterface() {}

};

}

#endif

