#ifndef _PYP_WEIGHTS_H_
#define _PYP_WEIGHTS_H_

#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "corpus/data_set.h"
#include "pyp/pyplm.h"
#include "pyp/constants.h"

namespace oxlm {

template<unsigned kOrder>
class PypWeights {

  public:
  PypWeights(size_t vocab_size);

  virtual Real predict(WordId word, Words context) const;

  virtual Real likelihood() const;

  virtual void resampleHyperparameters(MT19937& eng);

  void updateInsert(const DataSet& examples, MT19937& eng);
  
  void updateRemove(const DataSet& examples, MT19937& eng);
  
  void updateInsert(const DataPoint& example, MT19937& eng);

  void updateRemove(const DataPoint& example, MT19937& eng);

  virtual int vocabSize() const;

  private:
  PYPLM<kOrder> lm_;
  int vocab_size_;
};

}

#endif
