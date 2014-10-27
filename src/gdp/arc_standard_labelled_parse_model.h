#ifndef _GDP_AS_LAB_PARSE_MODEL_H_
#define _GDP_AS_LAB_PARSE_MODEL_H_

#include "gdp/accuracy_counts.h"
#include "gdp/arc_standard_labelled_parser.h"
#include "gdp/transition_parse_model_interface.h"

#include "pyp/parsed_pyp_weights.h"
#include "pyp/parsed_lex_pyp_weights.h"
#include "lbl/parsed_factored_weights.h"

namespace oxlm {

typedef std::vector<boost::shared_ptr<ArcStandardLabelledParser>> AsParserList;

template<class ParsedWeights>
class ArcStandardLabelledParseModel: public TransitionParseModelInterface<ArcStandardLabelledParser, ParsedWeights> {
  public:

  ArcStandardLabelledParseModel(boost::shared_ptr<ModelConfig> config);

  void resampleParticles(AsParserList* beam_stack, MT19937& eng, unsigned num_particles);

  ArcStandardLabelledParser beamParseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
                unsigned beam_size) override;

  ArcStandardLabelledParser particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles,
        bool resample) override;

  ArcStandardLabelledParser particleGoldParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, 
          unsigned num_particles, bool resample) override;

  ArcStandardLabelledParser staticGoldParseSentence(const ParsedSentence& sent, 
                                             const boost::shared_ptr<ParsedWeights>& weights) override;

  ArcStandardLabelledParser staticGoldParseSentence(const ParsedSentence& sent) override;

  ArcStandardLabelledParser generateSentence(const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng) override;

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

