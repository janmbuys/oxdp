#include "pyp/parsed_lex_pyp_weights.h"

namespace oxlm {

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
ParsedLexPypWeights<wOrder, tOrder, aOrder>::ParsedLexPypWeights(size_t vocab_size, size_t num_tags, size_t num_actions):
  ParsedPypWeights<tOrder, aOrder>(num_tags, num_actions),    
  lex_lm_(vocab_size, 1, 1, 1, 1),
  lex_vocab_size_(vocab_size) {}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
double ParsedLexPypWeights<wOrder, tOrder, aOrder>::predict(WordId word, Words context) const {
  return predictWord(word, context);
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
double ParsedLexPypWeights<wOrder, tOrder, aOrder>::predictWord(WordId word, Words context) const {
  return -std::log(lex_lm_.prob(word, context));
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
double ParsedLexPypWeights<wOrder, tOrder, aOrder>::likelihood() const {
  return wordLikelihood() + ParsedPypWeights<tOrder, aOrder>::tagLikelihood() + ParsedPypWeights<tOrder, aOrder>::actionLikelihood();
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
double ParsedLexPypWeights<wOrder, tOrder, aOrder>::wordLikelihood() const {
  return -lex_lm_.log_likelihood();
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
void ParsedLexPypWeights<wOrder, tOrder, aOrder>::resampleHyperparameters(MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::resampleHyperparameters(eng);
  lex_lm_.resample_hyperparameters(eng);
  std::cerr << "  [Word LLH=" << wordLikelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
void ParsedLexPypWeights<wOrder, tOrder, aOrder>::updateInsert(const DataSet& word_examples, const DataSet& tag_examples, const DataSet& action_examples, MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::updateInsert(tag_examples, action_examples, eng);
  for (const auto& ex: word_examples) 
    lex_lm_.increment(ex.word, ex.context, eng);
}

//update PYP model to remove old training examples
template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
void ParsedLexPypWeights<wOrder, tOrder, aOrder>::updateRemove(const DataSet& word_examples, const DataSet& tag_examples, const DataSet& action_examples, MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::updateRemove(tag_examples, action_examples, eng);
  for (const auto& ex: word_examples) 
    lex_lm_.decrement(ex.word, ex.context, eng);
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
size_t ParsedLexPypWeights<wOrder, tOrder, aOrder>::numWords() const {
  return lex_vocab_size_;
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
size_t ParsedLexPypWeights<wOrder, tOrder, aOrder>::vocabSize() const {
  return numWords();
}

template class ParsedLexPypWeights<wordLMOrder, tagLMOrder, actionLMOrder>;

}

