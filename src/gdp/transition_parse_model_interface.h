#ifndef GDP_TRAN_PARSE_MODEL_H_
#define GDP_TRAN_PARSE_MODEL_H_

#include "corpus/parsed_sentence.h"
#include "gdp/transition_parser.h"

namespace oxlm {

//TODO comment methods!  
// Interface for generative parsing decoding algorithms.
template <class TParser, class ParsedWeights>
class TransitionParseModelInterface {
 public:
  virtual TParser greedyParseSentence(
      const ParsedSentence& sent,
      const boost::shared_ptr<ParsedWeights>& weights) = 0;

  virtual TParser beamParseSentence(
      const ParsedSentence& sent,
      const boost::shared_ptr<ParsedWeights>& weights, unsigned beam_size) = 0;

  virtual TParser particleParseSentence(
      const ParsedSentence& sent,
      const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng,
      unsigned num_particles,
      const boost::shared_ptr<ParseDataSet>& examples) = 0;

  virtual TParser beamParticleParseSentence(
      const ParsedSentence& sent,
      const boost::shared_ptr<ParsedWeights>& weights, unsigned num_particles,
      const boost::shared_ptr<ParseDataSet>& examples) = 0;

  virtual TParser particleGoldParseSentence(
      const ParsedSentence& sent,
      const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng,
      unsigned num_particles) = 0;

  virtual ArcStandardLabelledParser beamParticleGoldParseSentence(
      const ParsedSentence& sent,
      const boost::shared_ptr<ParsedWeights>& weights,
      unsigned num_particles) = 0;

  virtual TParser staticGoldParseSentence(
      const ParsedSentence& sent,
      const boost::shared_ptr<ParsedWeights>& weights) = 0;

  virtual TParser staticGoldParseSentence(const ParsedSentence& sent) = 0;

  virtual TParser generateSentence(
      const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng,
      int sentence_id = 0) = 0;

  virtual ~TransitionParseModelInterface() {}
};

} // namespace oxlm

#endif
