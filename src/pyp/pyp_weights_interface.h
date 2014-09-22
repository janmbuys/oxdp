#ifndef _PYP_WEIGHTS_I_H_ 
#define _PYP_WEIGHTS_I_H_ 

#include "utils/random.h"
#include "corpus/weights_interface.h"
#include "corpus/data_set.h"

namespace oxlm {

class PypWeightsInterface: public WeightsInterface {
  public:
  virtual Real likelihood() const = 0;

  virtual void resampleHyperparameters(MT19937& eng) = 0;

  virtual void updateInsert(const DataSet& examples, MT19937& eng) = 0;

  virtual void updateRemove(const DataSet& examples, MT19937& eng) = 0;

  virtual ~PypWeightsInterface() {}
};

}

#endif
