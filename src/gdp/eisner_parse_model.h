#ifndef _GDP_ESNR_PARSE_MODEL_H_
#define _GDP_ESNR_PARSE_MODEL_H_

#include "gdp/eisner_parser.h"
#include "gdp/parse_model.h"

namespace oxlm {

class EisnerParseModel: public ParseModel {
  public:
  EisnerParseModel();

  EisnerParser parseSentence(const ParsedSentence& sent, const ParsedWeightsInterface& weights);

  EisnerParser scoreSentence(EisnerParser* parser, const ParsedWeightsInterface& weights);

  
};

}

#endif
