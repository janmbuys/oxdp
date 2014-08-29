#ifndef _PYP_WEIGHTS_H_
#define _PYP_WEIGHTS_H_

#include "pyp/pyp_weights_interface.h"
#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "pyp/pyplm.h"

namespace oxlm {

const LMOrder = 4;

template<kOrder>
class PypWeights: public PypWeightsInterface {

  public:
  PypWeights(size_t vocab_size);

  double predict(WordId word, Words context) const override;

  double likelihood() const override {
    return -lm.log_likelihood();
  }

  void resampleHyperparameters() override;

  void updateInsert(const DataSet& examples);
  
  void updateInsert(const DataPoint& example);

  void updateRemove(const DataSet& examples);
  
  void updateRemove(const DataPoint& example);

  private:
  PYPLM<kOrder> lm;
  MT19937 eng;
};

}

#endif
