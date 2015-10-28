#ifndef _PYP_WEIGHTS_H_
#define _PYP_WEIGHTS_H_

#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "corpus/data_set.h"
#include "pyp/pyplm.h"
#include "pyp/constants.h"

namespace oxlm {

// Implements a PYP language model.  
template <unsigned kOrder>
class PypWeights {
 public:
  PypWeights(size_t vocab_size);

  // Returns the negative log probability of the prediction.
  virtual Real predict(WordId word, Context context) const;

  virtual Reals predict(Context context) const;

  virtual Real likelihood() const;

  virtual void resampleHyperparameters(MT19937& eng);

  // Updates the PYP model to insert new training examples.
  void updateInsert(const boost::shared_ptr<DataSet>& examples, MT19937& eng);

  // Updates the PYP model to remove old training examples.
  void updateRemove(const boost::shared_ptr<DataSet>& examples, MT19937& eng);

  // Updates the PYP model to insert one training example.
  void updateInsert(const DataPoint& example, MT19937& eng);

  // Updates the PYP model to remove one training example.
  void updateRemove(const DataPoint& example, MT19937& eng);

  virtual int vocabSize() const;

 private:
  PYPLM<kOrder> lm_;
  int vocab_size_;
};

}  // namespace oxlm

#endif
