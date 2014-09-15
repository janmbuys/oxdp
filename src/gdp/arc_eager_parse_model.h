#ifndef _GDP_AE_PARSE_MODEL_H_
#define _GDP_AE_PARSE_MODEL_H_

#include "gdp/arc_eager_parser.h"
#include "gdp/transition_parse_model_interface.h"
#include "corpus/parsed_weights_interface.h"

namespace oxlm {

typedef std::vector<boost::shared_ptr<ArcEagerParser>> AeParserList;

class ArcEagerParseModel: public TransitionParseModelInterface<ArcEagerParser> {
  public:

  void resampleParticles(AeParserList* beam_stack, MT19937& eng, unsigned num_particles);

  ArcEagerParser beamParseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeightsInterface>& weights,
          unsigned beam_size) override;

  ArcEagerParser particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng, unsigned num_particles,
        bool resample) override;

  ArcEagerParser particleGoldParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng, 
          unsigned num_particles, bool resample) override;

  ArcEagerParser staticGoldParseSentence(const ParsedSentence& sent, 
                                    const boost::shared_ptr<ParsedWeightsInterface>& weights) override;

  ArcEagerParser staticGoldParseSentence(const ParsedSentence& sent) override;

  ArcEagerParser generateSentence(const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng) override;

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

