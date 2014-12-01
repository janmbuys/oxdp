#include "pyp/parsed_chlex_pyp_weights.h"

namespace oxlm {

template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::ParsedChLexPypWeights(
        boost::shared_ptr<Dict> dict, boost::shared_ptr<Dict> ch_dict, size_t num_actions):
  ParsedPypWeights<tOrder, aOrder>(dict, ch_dict, num_actions),    
  lex_lm_(ch_dict, 1, 1, 1, 1),
  dict_(dict) {}

template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
Real ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::predict(WordId word, Words context) const {
  return predictWord(word, context);
}

template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
Reals ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::predict(Words context) const {
  return predictWord(context);
}

template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
Reals ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::predictWord(Words context) const {
  Reals weights(numWords(), 0);
  for (int i = 0; i < numWords(); ++i)
    weights[i] = predictWord(i, context);
  return weights;
}

template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
Real ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::predictWord(WordId word, Words context) const {
  return -std::log(lex_lm_.prob(dict_->lookup(word), context));
}

template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
Real ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::likelihood() const {
  return wordLikelihood() + ParsedPypWeights<tOrder, aOrder>::tagLikelihood() + ParsedPypWeights<tOrder, aOrder>::actionLikelihood();
}

template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
Real ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::wordLikelihood() const {
  return -lex_lm_.log_likelihood();
}

template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
void ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::resampleHyperparameters(MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::resampleHyperparameters(eng);
  lex_lm_.resample_hyperparameters(eng);
  std::cerr << "  [Word LLH=" << wordLikelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
void ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::updateInsert(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::updateInsert(examples, eng);
  for (unsigned i = 0; i < examples->word_example_size(); ++i)
    lex_lm_.increment(dict_->lookup(examples->word_at(i)), examples->word_context_at(i), eng);
}

//update PYP model to remove old training examples
template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
void ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::updateRemove(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::updateRemove(examples, eng);
  for (unsigned i = 0; i < examples->word_example_size(); ++i)
    lex_lm_.decrement(dict_->lookup(examples->word_at(i)), examples->word_context_at(i), eng);
}

template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
int ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::numWords() const {
  return dict_->size();
}

template<unsigned wOrder, unsigned cOrder, unsigned tOrder, unsigned aOrder>
int ParsedChLexPypWeights<wOrder, cOrder, tOrder, aOrder>::vocabSize() const {
  return numWords();
}

template class ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>;

#if ((wordLMOrderAE != wordLMOrderAS) || (tagLMOrderAS != tagLMOrderAE) || (actionLMOrderAS != actionLMOrderAE))
template class ParsedChLexPypWeights<wordLMOrderAE, charLMOrder, tagLMOrderAE, actionLMOrderAE>;
#endif

template class ParsedChLexPypWeights<wordLMOrderE, charLMOrder, tagLMOrderE, 1>;

}

