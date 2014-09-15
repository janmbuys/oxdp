#ifndef _GDP_ESNR_PARSE_MODEL_H_
#define _GDP_ESNR_PARSE_MODEL_H_

#include "utils/random.h"
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

  void extractSentence(ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) override;

  double evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) override; 

};

}

#endif
