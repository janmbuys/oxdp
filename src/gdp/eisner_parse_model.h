#ifndef _GDP_ESNR_PARSE_MODEL_H_
#define _GDP_ESNR_PARSE_MODEL_H_

#include "gdp/eisner_parser.h"
#include "corpus/parsed_weights_interface.h"
#include "utils/random.h"

namespace oxlm {

class EisnerParseModel {
  public:

  EisnerParser parseSentence(const ParsedSentence& sent, const ParsedWeightsInterface& weights);

  void scoreSentence(EisnerParser* parser, const ParsedWeightsInterface& weights);
  
};

}

#endif
