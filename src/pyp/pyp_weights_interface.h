#ifndef _PYP_WEIGHTS_I_H_ 
#define _PYP_WEIGHTS_I_H_ 

#include "corpus/weights_interface.h"
#include "utils/random.h"

namespace oxlm {

class PypWeightsInterface: public WeightsInterface {
  public:
  virtual double likelihood() const = 0;

  virtual void resampleHyperparameters(MT19937& eng) = 0;

  virtual ~PypWeightsInterface() {}
};

}

#endif
