#ifndef _PYP_PARSE_LEX_WEIGHTS_H_
#define _PYP_PARSE_LEX_WEIGHTS_H_

#include "pyp/parsed_pyp_weights.h"

namespace oxlm {

#define wordLMOrder 6

//NB this is the unlexicalized model, with only tags
template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
class ParsedLexPypWeights: public ParsedPypWeights<tOrder, aOrder> {

  public:
  ParsedLexPypWeights(size_t vocab_size, size_t num_tags, size_t num_actions);

  double predict(WordId word, Words context) const override;

  double predictWord(WordId word, Words context) const override;
  
  double likelihood() const override;

  double wordLikelihood() const override;

  void resampleHyperparameters(MT19937& eng) override;

  void updateInsert(const DataSet& word_examples,
                const DataSet& tag_examples, 
                const DataSet& action_examples, MT19937& eng);

  void updateRemove(const DataSet& word_examples,
          const DataSet& tag_examples,
          const DataSet& action_examples, MT19937& eng);

  private:
  PYPLM<wOrder> lex_lm;
};

}

#endif
