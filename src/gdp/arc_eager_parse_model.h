#ifndef _GDP_AE_PARSE_MODEL_H_
#define _GDP_AE_PARSE_MODEL_H_

#include "gdp/arc_eager_parser.h"
#include "gdp/parse_model.h"

namespace oxlm {

typedef std::vector<boost::shared_ptr<ArcEagerParser>> AeParserList;

class ArcEagerParseModel: public ParseModel {
  public:
  ArcEagerParseModel(unsigned size);

  ArcEagerParser beamParseSentence(const ParsedSentence& sent, const ParsedWeightsInterface& weights);

  ArcEagerParser staticGoldParseSentence(const ParsedSentence& sent, 
                                    const ParsedWeightsInterface& weights);

  ArcEagerParser staticGoldParseSentence(const ParsedSentence& sent);
};

}
#endif

