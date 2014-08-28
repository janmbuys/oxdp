#ifndef _HPYP_WEIGHTS_H_
#define _HPYP_WEIGHTS_H_

#include "corpus/weights_interface.h"
#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "pyp/pyplm.h"

namespace oxlm {

const wordLMOrder = 4;

template<wOrder>
class PypWeights: public WeightsInterface {

  public:
  PypWeights(size_t vocab_size);

  virtual double predict(WordId word, Words context);

  virtual double log_likelihood() {
    return word_lm.log_likelihood();
  }

  virtual void resample_hyperparameters() {
    word_lm.resample_hyperparameters(eng);
    std::cerr << "  [Word LLH=" << log_likelihood() << "]\n\n";    
  }

  virtual void PypWeights::updateInsert(const DataSet& examples);

  virtual void PypWeights::updateRemove(const DataSet& examples);

  private:
  PYPLM<wOrder> word_lm;
  MT19937 eng;
}

}

#endif
