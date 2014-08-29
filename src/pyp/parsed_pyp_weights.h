#ifndef _PYP_PARSE_WEIGHTS_H_
#define _PYP_PARSE_WEIGHTS_H_

#include "pyp/pyp_weights.h"
#include "pyp/pyp_parsed_weights_interface.h"
#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "pyp/pyplm.h"

namespace oxlm {

const tagLMOrder = 6;
const actionLMOrder = 6;

//NB this is the unlexicalized model, with only tags
template<tOrder, aOrder>
class ParsedPypWeights<tOrder, aOrder>: public PypWeights<tOrder>, public PypParsedWeightsInterface {

  public:
  ParsedPypWeights(size_t num_tags, size_t num_actions);

  predictWord(WordId word, Words context) const;
  
  double predictTag(WordId tag, Words context) const override;

  double predictAction(WordId action, Words context) const override;

  double wordLikelihood() const override;

  double tagLikelihood() const override;

  double action_likelihood() const override;

  double likelihood() const override;

  void resampleHyperparameters() override;

  void updateInsert(const DataSet& tag_examples, 
          const DataSet& action_examples);

  void updateRemove(const DataSet& tag_examples,
          const DataSet& action_examples);

  private:
  PYPLM<aOrder> action_lm;
};

}

#endif
