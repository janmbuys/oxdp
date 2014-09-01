#ifndef _PYP_WEIGHTS_H_
#define _PYP_WEIGHTS_H_

#include "pyp/pyp_weights_interface.h"
#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "pyp/pyplm.h"

namespace oxlm {

#define LMOrder 4

template<unsigned kOrder>
class PypWeights: public PypWeightsInterface {

  public:
  PypWeights(size_t vocab_size);

  double predict(WordId word, Words context) const override;

  double likelihood() const override;

  void resampleHyperparameters(MT19937& eng) override;

  void updateInsert(const DataSet& examples, MT19937& eng);
  
  void updateInsert(const DataPoint& example, MT19937& eng);

  void updateRemove(const DataSet& examples, MT19937& eng);
  
  void updateRemove(const DataPoint& example, MT19937& eng);

  private:
  PYPLM<kOrder> lm;
};

}

#endif
