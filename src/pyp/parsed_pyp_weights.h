#ifndef _PYP_PARSE_WEIGHTS_H_
#define _PYP_PARSE_WEIGHTS_H_

#include "pyp/pyp_weights.h"
#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "corpus/parse_data_set.h"
#include "pyp/pyplm.h"

namespace oxlm {

// Implements an unlexicalised PYP generative parsing model, with tag and action
// distribution models.
template <unsigned tOrder, unsigned aOrder>
class ParsedPypWeights : public PypWeights<tOrder> {
 public:
  // TODO Implement separate action and label models with a parameter to set
  // joint or seperate scoring.
  ParsedPypWeights(boost::shared_ptr<Dict> dict, size_t num_actions);

  virtual Real predictWord(WordId word, Context context) const;

  virtual Reals predictWordOverTags(WordId word, Context context) const;

  virtual Reals predictWord(Context context) const;

  Real predictTag(WordId tag, Context context) const;

  Reals predictTag(Context context) const;

  Real predictAction(WordId action, Context context) const;

  Reals predictAction(Context context) const;

  virtual Real wordLikelihood() const;

  Real tagLikelihood() const;

  Real actionLikelihood() const;

  Real likelihood() const override;

  void resampleHyperparameters(MT19937& eng) override;

  virtual void updateInsert(const boost::shared_ptr<ParseDataSet>& examples,
                            MT19937& eng);

  virtual void updateRemove(const boost::shared_ptr<ParseDataSet>& examples,
                            MT19937& eng);

  virtual int numWords() const;

  int numTags() const;

  int numActions() const;

 private:
  PYPLM<aOrder> shre_lm_;
  PYPLM<aOrder> action_lm_;
  int num_actions_;
};

}  // namespace oxlm

#endif
