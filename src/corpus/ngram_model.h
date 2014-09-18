#ifndef _PYP_NGRAM_MODEL_H_
#define _PYP_NGRAM_MODEL_H_

#include "corpus/utils.h"
#include "corpus/sentence.h"
#include "corpus/data_set.h"
#include "corpus/model_interface.h"
#include "corpus/weights_interface.h"

namespace oxlm {

class NGramModel: public ModelInterface {
  public:
  NgramModel(unsigned order, WordId eos);

  void extractSentence(const Sentence& sent, 
          const boost::shared_ptr<DataSet>& examples) override;

  //return likelihood
  double evaluateSentence(const Sentence& sent, 
          const boost::shared_ptr<WeightsInterface>& weights) override;

  private:
  unsigned order_;
  WordId eos_;
          
};

}

#endif
