#ifndef _PYP_PARSE_LEX_WEIGHTS_H_
#define _PYP_PARSE_LEX_WEIGHTS_H_

#include "pyp/parsed_pyp_weights.h"
#include "pyp/pyp_parsed_weights_interface.h"

namespace oxlm {

//this is the lexicalized model, with tags and words
template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
class ParsedLexPypWeights: public ParsedPypWeights<tOrder, aOrder> {

  public:
  ParsedLexPypWeights(size_t vocab_size, size_t num_tags, size_t num_actions);

  double predict(WordId word, Words context) const override;

  double predictWord(WordId word, Words context) const override;
  
  double likelihood() const override;

  double wordLikelihood() const override;

  void resampleHyperparameters(MT19937& eng) override;

  void updateInsert(const ParseDataSet& examples, MT19937& eng) override;

  void updateRemove(const ParseDataSet& examples, MT19937& eng) override;

  void updateInsert(const DataSet& examples, MT19937& eng) override;

  void updateRemove(const DataSet& examples, MT19937& eng) override;
  
  size_t numWords() const override;

  size_t vocabSize() const override;

  private:
  PYPLM<wOrder> lex_lm_;
  size_t lex_vocab_size_;
};

}

#endif
