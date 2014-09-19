#ifndef _CORPUS_NGRAM_MODEL_H_
#define _CORPUS_NGRAM_MODEL_H_

#include "corpus/utils.h"
#include "corpus/sentence.h"
#include "corpus/data_set.h"
#include "corpus/corpus.h"
#include "corpus/weights_interface.h"

namespace oxlm {

class NGramModel {
  public:
  NGramModel(unsigned order, WordId sos, WordId eos);

  Words extractContext(const boost::shared_ptr<Corpus> corpus, int position);

  void extract(const boost::shared_ptr<Corpus> corpus, int position,     
          const boost::shared_ptr<DataSet>& examples);

  double evaluate(const boost::shared_ptr<Corpus> corpus, int position, 
          const boost::shared_ptr<WeightsInterface>& weights); 

  void extractSentence(const Sentence& sent, 
          const boost::shared_ptr<DataSet>& examples);

  //return likelihood
  double evaluateSentence(const Sentence& sent, 
          const boost::shared_ptr<WeightsInterface>& weights);

  private:
  unsigned order_;
  WordId sos_;
  WordId eos_;
          
};

}

#endif
