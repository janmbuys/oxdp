#ifndef _CORPUS_MODEL_I_H_
#define _CORPUS_MODEL_I_H_

#include "corpus/utils.h"
#include "corpus/sentence.h"
#include "corpus/weights_interface.h"

namespace oxlm {

class ModelInterface {
  public:

  virtual void extractSentence(const Sentence& sent, 
          const boost::shared_ptr<DataSet>& examples) = 0;

  //return likelihood
  virtual double evaluateSentence(const Sentence& sent, 
          const boost::shared_ptr<WeightsInterface>& weights) = 0; 

  virtual ~ModelInterface() {}
          
};

}

#endif
