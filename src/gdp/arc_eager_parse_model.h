#ifndef _GDP_AE_PARSE_MODEL_H_
#define _GDP_AE_PARSE_MODEL_H_

#include "gdp/accuracy_counts.h"
#include "gdp/arc_eager_parser.h"
#include "gdp/transition_parse_model_interface.h"

#include "pyp/parsed_pyp_weights.h"
#include "pyp/parsed_lex_pyp_weights.h"
#include "lbl/parsed_factored_weights.h"

namespace oxlm {

typedef std::vector<boost::shared_ptr<ArcEagerParser>> AeParserList;

template<class ParsedWeights>
class ArcEagerParseModel: public TransitionParseModelInterface<ArcEagerParser, ParsedWeights> {
  public:

  ArcEagerParseModel(boost::shared_ptr<ModelConfig> config);

  void resampleParticles(AeParserList* beam_stack, MT19937& eng, unsigned num_particles);

  ArcEagerParser beamParseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
          unsigned beam_size) override;

  ArcEagerParser particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles,
        bool resample) override;

  ArcEagerParser particleGoldParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, 
          unsigned num_particles, bool resample) override;

  ArcEagerParser staticGoldParseSentence(const ParsedSentence& sent, 
                                    const boost::shared_ptr<ParsedWeights>& weights) override;

  ArcEagerParser staticGoldParseSentence(const ParsedSentence& sent) override;

  ArcEagerParser generateSentence(const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng) override;

  void extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParseDataSet>& examples);

  void extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples);

  void extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<ParseDataSet>& examples);

  void extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<ParseDataSet>& examples);

  void extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples);

  Real evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size); 

  Real evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size); 
  
  private:
  boost::shared_ptr<ModelConfig> config_;
};

}
#endif

