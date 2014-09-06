#ifndef _GDP_AE_PARSE_MODEL_H_
#define _GDP_AE_PARSE_MODEL_H_

#include "gdp/arc_eager_parser.h"
#include "gdp/transition_parse_model_interface.h"
#include "corpus/parsed_weights_interface.h"

namespace oxlm {

typedef std::vector<boost::shared_ptr<ArcEagerParser>> AeParserList;

class ArcEagerParseModel: public TransitionParseModelInterface {
  public:

  void resampleParticles(AeParserList* beam_stack, MT19937& eng, unsigned num_particles);

  TransitionParser beamParseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeightsInterface>& weights,
          unsigned beam_size) override;

  TransitionParser particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng, unsigned num_particles,
        bool resample) override;

  TransitionParser particleGoldParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng, 
          unsigned num_particles, bool resample) override;

  TransitionParser staticGoldParseSentence(const ParsedSentence& sent, 
                                    const boost::shared_ptr<ParsedWeightsInterface>& weights) override;

  TransitionParser staticGoldParseSentence(const ParsedSentence& sent) override;

  TransitionParser generateSentence(const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng) override;
};

}
#endif

