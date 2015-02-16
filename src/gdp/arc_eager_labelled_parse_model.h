#ifndef _GDP_AE_LAB_PARSE_MODEL_H_
#define _GDP_AE_LAB_PARSE_MODEL_H_

#include "gdp/accuracy_counts.h"
#include "gdp/arc_eager_labelled_parser.h"
#include "gdp/transition_parse_model_interface.h"

#include "pyp/parsed_pyp_weights.h"
#include "pyp/parsed_lex_pyp_weights.h"
#include "pyp/parsed_chlex_pyp_weights.h"
#include "lbl/parsed_factored_weights.h"
#include "lbl/parsed_weights.h"

namespace oxlm {

typedef std::vector<boost::shared_ptr<ArcEagerLabelledParser>> AelParserList;

template<class ParsedWeights>
class ArcEagerLabelledParseModel: public TransitionParseModelInterface<ArcEagerLabelledParser, ParsedWeights> {
  public:

  ArcEagerLabelledParseModel(boost::shared_ptr<ModelConfig> config);

  void resampleParticles(AelParserList* beam_stack, MT19937& eng, unsigned num_particles);
  
  void reallocateParticles(AelParserList* beam_stack, unsigned num_particles);
  
  void resampleParticleParticles(AelParserList* beam_stack, MT19937& eng, unsigned num_particles);

  ArcEagerLabelledParser greedyParseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights) override;

  ArcEagerLabelledParser beamDiscriminativeParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, unsigned beam_size);

  ArcEagerLabelledParser beamParseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
          unsigned beam_size) override;

  ArcEagerLabelledParser beamParticleParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, unsigned num_particles);

/*  ArcEagerLabelledParser particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles,
        bool resample) override;

  ArcEagerLabelledParser particleGoldParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, 
          unsigned num_particles, bool resample) override;  */

  ArcEagerLabelledParser staticGoldParseSentence(const ParsedSentence& sent, 
                                    const boost::shared_ptr<ParsedWeights>& weights) override;

  ArcEagerLabelledParser staticGoldParseSentence(const ParsedSentence& sent) override;

  ArcEagerLabelledParser generateSentence(const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng) override;

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

  Parser evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size); 

  Parser evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size); 
  
  private:
  boost::shared_ptr<ModelConfig> config_;
};

}
#endif

