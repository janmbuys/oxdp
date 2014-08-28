#ifndef _PYP_PARSE_WEIGHTS_H_
#define _PYP_PARSE_WEIGHTS_H_

#include "pyp/pyp_weights.h"
#include "corpus/parsed_weights_interface.h"
#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "pyp/pyplm.h"

namespace oxlm {

const tagLMOrder = 6;
const actionLMOrder = 6;

//NB this is the unlexicalized model, with only tags
template<wOrder, aOrder>
class ParsedPypWeights: public PypWeights<wOrder>, public ParsedWeightsInterface {

  public:
  ParsedWeights(size_t vocab_size, size_t num_actions);

  virtual double predictWord(WordId word, Words context);
  
  virtual double predictTag(WordId tag, Words context);

  virtual double predictAction(WordId action, Words context);

  virtual double word_likelihood() {
    return 0;
  }

  virtual double tag_likelihood() {
    return PypWeights::log_likelihood();
  }

  virtual double action_likelihood() {
    return -action_lm.log_likelihood();
  }

  virtual double likelihood() {
    return tag_likelihood() + action_likelihood();
  }

  virtual void resample_hyperparameters() {
    PypWeights<wOrder>::resample_hyperparameters();
    action_lm.resample_hyperparameters(eng);
    std::cerr << "  [Action LLH=" << action_likelihood() << "]\n\n";    
  }

  virtual void PypWeights::updateInsert(const DataSet& tag_examples, 
          const DataSet& action_examples);

  virtual void PypWeights::updateRemove(const DataSet& tag_examples,
          const DataSet& action_examples);

  private:
  PYPLM<aOrder> action_lm;
}

}

#endif
