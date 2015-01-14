#ifndef _GDP_AS_PARSE_MODEL_H_
#define _GDP_AS_PARSE_MODEL_H_

#include "gdp/accuracy_counts.h"
#include "gdp/arc_standard_parser.h"
#include "gdp/transition_parse_model_interface.h"

#include "pyp/parsed_pyp_weights.h"
#include "pyp/parsed_lex_pyp_weights.h"
#include "pyp/parsed_chlex_pyp_weights.h"
#include "lbl/parsed_factored_weights.h"
#include "lbl/parsed_weights.h"

namespace oxlm {

typedef std::vector<boost::shared_ptr<ArcStandardParser>> AsParserList;

template<class ParsedWeights>
class ArcStandardParseModel: public TransitionParseModelInterface<ArcStandardParser, ParsedWeights> {
  public:

  ArcStandardParseModel(boost::shared_ptr<ModelConfig> config);

  void resampleParticles(AsParserList* beam_stack, MT19937& eng, unsigned num_particles);

  ArcStandardParser greedyParseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights) override;

  ArcStandardParser beamParseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
                unsigned beam_size) override;

  ArcStandardParser particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles,
        bool resample) override;

  ArcStandardParser particleGoldParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, 
          unsigned num_particles, bool resample) override;

  ArcStandardParser staticGoldParseSentence(const ParsedSentence& sent, 
                                             const boost::shared_ptr<ParsedWeights>& weights) override;

  ArcStandardParser staticGoldParseSentence(const ParsedSentence& sent) override;

  ArcStandardParser generateSentence(const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng) override;

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

