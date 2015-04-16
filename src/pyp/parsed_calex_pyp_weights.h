#ifndef _PYP_PARSE_CALEX_WEIGHTS_H_
#define _PYP_PARSE_CALEX_WEIGHTS_H_

#include "pyp/parsed_pyp_weights.h"
#include "pyp/capyplm.h"

namespace oxlm {

//Lexicalized model, predicting tags and words
template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
class ParsedCALexPypWeights: public ParsedPypWeights<tOrder, aOrder> {
 public:
  ParsedCALexPypWeights(boost::shared_ptr<Dict> dict, boost::shared_ptr<Dict> ch_dict,
                      size_t num_actions);

  Real predict(WordId word, Context context) const override;

  Reals predict(Context context) const override;

  Real predictWord(WordId word, Context context) const override;
  
  Reals predictWordOverTags(WordId word, Context context) const override;

  Reals predictWord(Context context) const override;
  
  Real likelihood() const override;

  Real wordLikelihood() const override;

  void resampleHyperparameters(MT19937& eng) override;

  void updateInsert(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) override;

  void updateRemove(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) override;
  
  int numWords() const override;

  int vocabSize() const override;

 private:
  PYPLM<waOrder> unlex_lm_;
  CAPYPLM<wOrder,waOrder> lex_lm_;
  int lex_vocab_size_;
};

}

#endif
