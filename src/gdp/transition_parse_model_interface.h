#ifndef GDP_TRAN_PARSE_MODEL_H_
#define GDP_TRAN_PARSE_MODEL_H_

#include "corpus/parsed_sentence.h"
#include "corpus/parsed_weights_interface.h"

namespace oxlm {

//try without a template

//template <class TParser>
class TransitionParseModelInterface {
  public:
  virtual TransitionParser beamParseSentence(const ParsedSentence& sent, 
        const ParsedWeightsInterface& weights, unsigned beam_size) = 0;

  virtual TransitionParser particleParseSentence(const ParsedSentence& sent, 
        const ParsedWeightsInterface& weights, MT19937& eng, unsigned num_particles,
        bool resample) = 0;

  //sample a derivation for the gold parse, given the current model
  virtual TransitionParser particleGoldParseSentence(const ParsedSentence& sent, 
          const ParsedWeightsInterface& weights, MT19937& eng, 
          unsigned num_particles, bool resample) = 0;

  virtual TransitionParser staticGoldParseSentence(const ParsedSentence& sent, 
        const ParsedWeightsInterface& weights) = 0;

  virtual TransitionParser staticGoldParseSentence(const ParsedSentence& sent) = 0;

  virtual TransitionParser generateSentence(const ParsedWeightsInterface& weights, MT19937& eng) = 0;

  virtual ~TransitionParseModelInterface() {}

};

}

#endif
