#ifndef _GDP_AS_PARSE_MODEL_H_
#define _GDP_AS_PARSE_MODEL_H_

#include "gdp/arc_standard_parser.h"
#include "gdp/transition_parse_model_interface.h"
#include "corpus/parsed_weights_interface.h"

namespace oxlm {

typedef std::vector<boost::shared_ptr<ArcStandardParser>> AsParserList;

class ArcStandardParseModel: public TransitionParseModelInterface<ArcStandardParser> {
  public:

  void resampleParticles(AsParserList* beam_stack, MT19937& eng, unsigned num_particles);

  ArcStandardParser beamParseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeightsInterface>& weights,
                unsigned beam_size) override;

  ArcStandardParser particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng, unsigned num_particles,
        bool resample) override;

  ArcStandardParser particleGoldParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng, 
          unsigned num_particles, bool resample) override;

  ArcStandardParser staticGoldParseSentence(const ParsedSentence& sent, 
                                             const boost::shared_ptr<ParsedWeightsInterface>& weights) override;

  ArcStandardParser staticGoldParseSentence(const ParsedSentence& sent) override;

  ArcStandardParser generateSentence(const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng) override;

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

