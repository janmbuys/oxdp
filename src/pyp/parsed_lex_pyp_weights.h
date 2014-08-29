#ifndef _PYP_PARSE_LEX_WEIGHTS_H_
#define _PYP_PARSE_LEX_WEIGHTS_H_

#include "pyp/parsed_pyp_weights.h"

namespace oxlm {

const wordLMOrder = 6;

//NB this is the unlexicalized model, with only tags
template<wOrder, tOrder, aOrder>
class ParsedLexPypWeights<wOrder, tOrder, aOrder>: public ParsedPypWeights<tOrder, aOrder> {

  public:
  ParsedLexPypWeights(size_t vocab_size, size_t num_tags, size_t num_actions);

  double predict(WordId word, Words context) const override;

  double predictWord(WordId word, Words context) const override;
  
  double likelihood() const override {
    return wordLikelihood + tagLikelihood() + actionLikelihood();
  }

  double wordLikelihood() const override {
    return -lex_lm.log_likelihood();
  }

  void resampleHyperparameters() override {
    ParsedPypWeights<tOrder, aOrder>::resampleHyperparameters();
    lex_lm.resample_hyperparameters(eng);
    std::cerr << "  [Word LLH=" << word_likelihood() << "]\n\n";    
  }

  void updateInsert(const DataSet& word_examples,
                const DataSet& tag_examples, 
                const DataSet& action_examples);

  void updateRemove(const DataSet& word_examples,
          const DataSet& tag_examples,
          const DataSet& action_examples);

  private:
  PYPLM<wOrder> lex_lm;
};

}

#endif
