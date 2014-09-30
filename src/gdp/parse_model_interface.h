#ifndef _GDP_PARSE_MODEL_I_H_
#define _GDP_PARSE_MODEL_I_H_

#include "corpus/utils.h"
#include "corpus/parsed_sentence.h"
#include "corpus/parsed_weights_interface.h"
#include "gdp/accuracy_counts.h"

namespace oxlm {

class ParseModelInterface {
  public:

  virtual void extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParseDataSet>& examples) = 0;

  virtual void extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) = 0;

  virtual void extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          MT19937& eng, const boost::shared_ptr<ParseDataSet>& examples) = 0;

  //return likelihood
  virtual Real evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) = 0; 

  virtual ~ParseModelInterface() {}
          
};

}

#endif
