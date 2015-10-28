#ifndef _PYP_PARSE_LEX_WEIGHTS_H_
#define _PYP_PARSE_LEX_WEIGHTS_H_

#include "pyp/parsed_pyp_weights.h"

namespace oxlm {

// Implements a lexicalised PYP generative parsing model. Predicts words, tags
// and actions.
template <unsigned wOrder, unsigned tOrder, unsigned aOrder>
class ParsedLexPypWeights : public ParsedPypWeights<tOrder, aOrder> {
 public:
  ParsedLexPypWeights(boost::shared_ptr<Dict> dict, size_t num_actions);

  Real predict(WordId word, Context context) const override;

  Reals predict(Context context) const override;

  Real predictWord(WordId word, Context context) const override;

  Reals predictWordOverTags(WordId word, Context context) const override;

  Reals predictWord(Context context) const override;

  Real likelihood() const override;

  Real wordLikelihood() const override;

  void resampleHyperparameters(MT19937& eng) override;

  void updateInsert(const boost::shared_ptr<ParseDataSet>& examples,
                    MT19937& eng) override;

  void updateRemove(const boost::shared_ptr<ParseDataSet>& examples,
                    MT19937& eng) override;

  int numWords() const override;

  int vocabSize() const override;

 private:
  PYPLM<wOrder> lex_lm_;
  int lex_vocab_size_;
};

}  // namespace oxlm

#endif
