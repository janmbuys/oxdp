#ifndef _GDP_AS_PARSE_MODEL_H_
#define _GDP_AS_PARSE_MODEL_H_

#include "gdp/arc_standard_parser.h"
#include "gdp/transition_parse_model_interface.h"
#include "corpus/parsed_weights_interface.h"

namespace oxlm {

typedef std::vector<boost::shared_ptr<ArcStandardParser>> AsParserList;

class ArcStandardParseModel: public TransitionParseModelInterface {
  public:

  void resampleParticles(AsParserList* beam_stack, MT19937& eng, unsigned num_particles);

  //TODO downcast return type to TransitionParser
  ArcStandardParser beamParseSentence(const ParsedSentence& sent, const ParsedWeightsInterface& weights,
                unsigned beam_size) override;

  ArcStandardParser particleParseSentence(const ParsedSentence& sent, 
        const ParsedWeightsInterface& weights, MT19937& eng, unsigned num_particles,
        bool resample) override;

  ArcStandardParser particleGoldParseSentence(const ParsedSentence& sent, 
          const ParsedWeightsInterface& weights, MT19937& eng, 
          unsigned num_particles, bool resample) override;

  ArcStandardParser staticGoldParseSentence(const ParsedSentence& sent, 
                                             const ParsedWeightsInterface& weights) override;

  ArcStandardParser staticGoldParseSentence(const ParsedSentence& sent) override;

  ArcStandardParser generateSentence(const ParsedWeightsInterface& weights, MT19937& eng) override;
};

}
#endif

