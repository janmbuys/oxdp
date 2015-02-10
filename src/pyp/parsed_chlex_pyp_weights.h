#ifndef _PYP_PARSE_CHLEX_WEIGHTS_H_
#define _PYP_PARSE_CHLEX_WEIGHTS_H_

#include "pyp/chpyplm.h"
#include "pyp/parsed_pyp_weights.h"

namespace oxlm {

//lexicalized model, character-based word prediction model
template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
class ParsedChLexPypWeights: public ParsedPypWeights<tOrder, aOrder> {

  public:     
  ParsedChLexPypWeights(boost::shared_ptr<Dict> dict, boost::shared_ptr<Dict> ch_dict, 
          size_t num_actions);

  Real predict(WordId word, Words context) const override;

  Reals predict(Words context) const override;

  Real predictWord(WordId word, Words context) const override;
  
  Reals predictWord(Words context) const override;
  
  Real likelihood() const override;

  Real wordLikelihood() const override;

  void resampleHyperparameters(MT19937& eng) override;

  void updateInsert(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) override;

  void updateRemove(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) override;
  
  int numWords() const override;

  int vocabSize() const override;

  private:
  CHPYPLM<wOrder, cOrder> lex_lm_;
  boost::shared_ptr<Dict> dict_;
};

}

#endif
