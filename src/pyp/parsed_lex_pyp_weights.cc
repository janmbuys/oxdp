#include "pyp/parsed_lex_pyp_weights.h"

namespace oxlm {

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
ParsedLexPypWeights<wOrder, tOrder, aOrder>::ParsedLexPypWeights(
        boost::shared_ptr<Dict> dict, size_t num_actions):
  ParsedPypWeights<tOrder, aOrder>(dict, num_actions),    
  lex_lm_(dict->size(), 1, 1, 1, 1),
  lex_vocab_size_(dict->size()) {}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
Real ParsedLexPypWeights<wOrder, tOrder, aOrder>::predict(WordId word, Context context) const {
  return predictWord(word, context);
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
Reals ParsedLexPypWeights<wOrder, tOrder, aOrder>::predict(Context context) const {
  return predictWord(context);
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
Reals ParsedLexPypWeights<wOrder, tOrder, aOrder>::predictWord(Context context) const {
  Reals weights(numWords(), 0);
  for (int i = 0; i < numWords(); ++i)
    weights[i] = predictWord(i, context);
  return weights;
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
Real ParsedLexPypWeights<wOrder, tOrder, aOrder>::predictWord(WordId word, Context context) const {
  return -std::log(lex_lm_.prob(word, context.words));
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
Reals ParsedLexPypWeights<wOrder, tOrder, aOrder>::predictWordOverTags(WordId word, Context context) const {
  Reals weights(ParsedPypWeights<tOrder,aOrder>::numTags(), 0);
  for (int i = 0; i < ParsedPypWeights<tOrder,aOrder>::numTags(); ++i) {
    context.words.back() = i;
    weights[i] = -std::log(lex_lm_.prob(word, context.words));
  }
  return weights;
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
Real ParsedLexPypWeights<wOrder, tOrder, aOrder>::likelihood() const {
  return wordLikelihood() + ParsedPypWeights<tOrder, aOrder>::tagLikelihood() + ParsedPypWeights<tOrder, aOrder>::actionLikelihood();
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
Real ParsedLexPypWeights<wOrder, tOrder, aOrder>::wordLikelihood() const {
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
void ParsedLexPypWeights<wOrder, tOrder, aOrder>::updateInsert(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::updateInsert(examples, eng);
  for (unsigned i = 0; i < examples->word_example_size(); ++i)
    lex_lm_.increment(examples->word_at(i), examples->word_context_at(i).words, eng);
}

//update PYP model to remove old training examples
template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
void ParsedLexPypWeights<wOrder, tOrder, aOrder>::updateRemove(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::updateRemove(examples, eng);
  for (unsigned i = 0; i < examples->word_example_size(); ++i)
    lex_lm_.decrement(examples->word_at(i), examples->word_context_at(i).words, eng);
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
int ParsedLexPypWeights<wOrder, tOrder, aOrder>::numWords() const {
  return lex_vocab_size_;
}

template<unsigned wOrder, unsigned tOrder, unsigned aOrder>
int ParsedLexPypWeights<wOrder, tOrder, aOrder>::vocabSize() const {
  return numWords();
}

template class ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>;

#if ((wordLMOrderAE != wordLMOrderAS) || (tagLMOrderAS != tagLMOrderAE) || (actionLMOrderAS != actionLMOrderAE))
template class ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>;
#endif

template class ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>;

}

