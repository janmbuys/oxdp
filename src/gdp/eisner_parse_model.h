#ifndef _GDP_ESNR_PARSE_MODEL_H_
#define _GDP_ESNR_PARSE_MODEL_H_

#include "utils/random.h"
#include "corpus/utils.h"
#include "corpus/parsed_weights_interface.h"

#include "gdp/parse_model_interface.h"
#include "gdp/eisner_parser.h"

namespace oxlm {

class EisnerParseModel: public ParseModelInterface {
  public:

  EisnerParser parseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeightsInterface>& weights);

  void scoreSentence(EisnerParser* parser, const boost::shared_ptr<ParsedWeightsInterface>& weights);
  
  void extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParseDataSet>& examples) override;

  void extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) override;

  void extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          MT19937& eng, const boost::shared_ptr<ParseDataSet>& examples) override;

  Real evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) override; 

  Real evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          MT19937& eng, const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) override; 
};

}

#endif
