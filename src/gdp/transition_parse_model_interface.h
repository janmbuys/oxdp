#ifndef GDP_TRAN_PARSE_MODEL_H_
#define GDP_TRAN_PARSE_MODEL_H_

#include "corpus/parsed_sentence.h"
#include "corpus/parsed_weights_interface.h"
#include "gdp/parse_model_interface.h"
#include "gdp/transition_parser.h"

namespace oxlm {

template <class TParser>
class TransitionParseModelInterface: public ParseModelInterface {
  public:
  virtual TParser beamParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeightsInterface>& weights, unsigned beam_size) = 0;

  virtual TParser particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng, unsigned num_particles,
        bool resample) = 0;

  //sample a derivation for the gold parse, given the current model
  virtual TParser particleGoldParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng, 
          unsigned num_particles, bool resample) = 0;

  virtual TParser staticGoldParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeightsInterface>& weights) = 0;

  virtual TParser staticGoldParseSentence(const ParsedSentence& sent) = 0;

  virtual TParser generateSentence(const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng) = 0;

  virtual ~TransitionParseModelInterface() {}

};

}

#endif
