#ifndef GDP_TRAN_PARSE_MODEL_H_
#define GDP_TRAN_PARSE_MODEL_H_

#include "corpus/parsed_sentence.h"
#include "corpus/parsed_weights_interface.h"

namespace oxlm {

template <class TParser>
class TransitionParseModelInterface {

  public:

  //this is a bit of a duplication, but otherwise it would just make things more messy
  virtual void resampleParticles(std::vector<boost::shared_ptr<TParser>>* beam_stack, MT19937& eng, 
          unsigned num_particles) = 0;
      
  virtual TParser beamParseSentence(const ParsedSentence& sent, 
        const ParsedWeightsInterface& weights, unsigned beam_size) = 0;

  virtual TParser particleParseSentence(const ParsedSentence& sent, 
        const ParsedWeightsInterface& weights, MT19937& eng, unsigned num_particles,
        bool resample) = 0;

  //sample a derivation for the gold parse, given the current model
  virtual TParser particleGoldParseSentence(const ParsedSentence& sent, 
          const ParsedWeightsInterface& weights, MT19937& eng, 
          unsigned num_particles, bool resample) = 0;

  virtual TParser staticGoldParseSentence(const ParsedSentence& sent, 
        const ParsedWeightsInterface& weights) = 0;

  virtual TParser staticGoldParseSentence(const ParsedSentence& sent) = 0;

  virtual TParser generateSentence(const ParsedWeightsInterface& weights, MT19937& eng) = 0;

  virtual ~TransitionParseModelInterface() {}

};

}

#endif
