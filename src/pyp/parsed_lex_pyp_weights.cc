#include "pyp/parsed_lex_pyp_weights.h"

namespace oxlm {

template<wOrder, tOrder, aOrder>
ParsedLexPypWeights<wOrder, tOrder, aOrder>::ParsedLexPypWeights(size_t vocab_size, 
        size_t num_tags, size_t num_actions);
  ParsedPypWeights<tOrder, aOrder>(num_tags, num_actions),    
  lex_lm(vocab_size, 1, 1, 1, 1) {}

  template<wOrder, tOrder, aOrder>
ParsedLexPypWeights<wOrder, tOrder, aOrder>::predict(WordId word, Words context) {
  return predictWord(word, context);
}

template<wOrder, tOrder, aOrder>
ParsedLexPypWeights<wOrder, tOrder, aOrder>::predictWord(WordId word, Words context) {
  return -std::log(lex_lm.prob(word, context));
}

template<wOrder, tOrder, aOrder>
double ParsedLexPypWeights<wOrder, tOrder, aOrder>::likelihood() {
  return wordLikelihood + tagLikelihood() + actionLikelihood();
}

template<wOrder, tOrder, aOrder>
double ParsedLexPypWeights<wOrder, tOrder, aOrder>::wordLikelihood() {
  return -lex_lm.log_likelihood();
}

template<wOrder, tOrder, aOrder>
void ParsedLexPypWeights<wOrder, tOrder, aOrder>::resampleHyperparameters() {
  ParsedPypWeights<tOrder, aOrder>::resampleHyperparameters();
  lex_lm.resample_hyperparameters(eng);
  std::cerr << "  [Word LLH=" << wordLikelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<wOrder, tOrder, aOrder>
ParsedLexPypWeights<wOrder, tOrder, aOrder>::updateInsert(const DataSet& word_examples, const DataSet& tag_examples, const DataSet& action_examples) {
  updateInsert(tag_examples, action_examples);
  for (const auto& ex: word_examples) 
    lex_lm->increment(ex.prediction, ex.context, eng);
}

//update PYP model to remove old training examples
template<wOrder, tOrder, aOrder>
ParsedLexPypWeights<wOrder, tOrder, aOrder>::updateRemove(const DataSet& word_examples, const DataSet& tag_examples, const DataSet& action_examples) {
  updateInsert(tag_examples, action_examples);
  for (const auto& ex: word_examples) 
    lex_lm->decrement(ex.prediction, ex.context, eng);
}

template class ParsedLexPypWeights<wordLMOrder, tagLMOrder, actionLMOrder>;

}

