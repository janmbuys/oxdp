#ifndef _GDP_AS_PARSE_MODEL_H_
#define _GDP_AS_PARSE_MODEL_H_

#include "gdp/arc_standard_parser.h"
#include "gdp/parse_model.h"

namespace oxlm {

typedef std::vector<boost::shared_ptr<ArcStandardParser>> AsParserList;

class ArcStandardParseModel: public ParseModel {
  public:
  ArcStandardParseModel(unsigned size);

  ArcStandardParser beamParseSentence(const ParsedSentence& sent, const ParsedWeightsInterface& weights);

  ArcStandardParser staticGoldParseSentence(const ParsedSentence& sent, 
                                             const ParsedWeightsInterface& weights);

  ArcStandardParser staticGoldParseSentence(const ParsedSentence& sent);
};

}
#endif

