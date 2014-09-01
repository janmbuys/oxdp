#ifndef _PYP_PARSE_WEIGHTS_H_
#define _PYP_PARSE_WEIGHTS_H_

#include "pyp/pyp_weights.h"
#include "pyp/pyp_parsed_weights_interface.h"
#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "pyp/pyplm.h"

namespace oxlm {

#define tagLMOrder 6
#define actionLMOrder 6

//NB this is the unlexicalized model, with only tags
template<unsigned tOrder, unsigned aOrder>
class ParsedPypWeights: public PypWeights<tOrder>, public PypParsedWeightsInterface {

  public:
  ParsedPypWeights(size_t num_tags, size_t num_actions);

  double predictWord(WordId word, Words context) const override;
  
  double predictTag(WordId tag, Words context) const override;

  double predictAction(WordId action, Words context) const override;

  double wordLikelihood() const override;

  double tagLikelihood() const override;

  double actionLikelihood() const override;

  double likelihood() const override;

  void resampleHyperparameters(MT19937& eng) override;

  void updateInsert(const DataSet& tag_examples, 
          const DataSet& action_examples, MT19937& eng);

  void updateRemove(const DataSet& tag_examples,
          const DataSet& action_examples, MT19937& eng);

  private:
  PYPLM<aOrder> action_lm;
};

}

#endif