#include "pyp/parsed_calex_pyp_weights.h"

namespace oxlm {

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::ParsedCALexPypWeights(
        boost::shared_ptr<Dict> dict, boost::shared_ptr<Dict> ch_dict, size_t num_actions):
  ParsedPypWeights<tOrder, aOrder>(dict, ch_dict, num_actions),    
  unlex_lm_(dict->size(), 1, 1, 1, 1),
  lex_lm_(unlex_lm_),
  lex_vocab_size_(dict->size()) {}

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
Real ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::predict(WordId word, Context context) const {
  return predictWord(word, context);
}

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
Reals ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::predict(Context context) const {
  return predictWord(context);
}

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
Reals ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::predictWord(Context context) const {
  Reals weights(numWords(), 0);
  for (int i = 0; i < numWords(); ++i)
    weights[i] = predictWord(i, context);
  return weights;
}

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
Real ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::predictWord(WordId word, Context context) const {
  return -std::log(lex_lm_.prob(word, context.words, context.tags));
}

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
Reals ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::predictWordOverTags(WordId word, Context context) const {
  Reals weights(ParsedPypWeights<tOrder,aOrder>::numTags(), 0);
  for (int i = 0; i < ParsedPypWeights<tOrder,aOrder>::numTags(); ++i) {
    context.words.back() = i;
    weights[i] = -std::log(lex_lm_.prob(word, context.words, context.tags));
  }
  return weights;
}

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
Real ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::likelihood() const {
  return wordLikelihood() + ParsedPypWeights<tOrder, aOrder>::tagLikelihood() + ParsedPypWeights<tOrder, aOrder>::actionLikelihood();
}

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
Real ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::wordLikelihood() const {
  return -lex_lm_.log_likelihood();
}

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
void ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::resampleHyperparameters(MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::resampleHyperparameters(eng);
  lex_lm_.resample_hyperparameters(eng);
  unlex_lm_.resample_hyperparameters(eng);
  std::cerr << "  [Word LLH=" << wordLikelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
void ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::updateInsert(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::updateInsert(examples, eng);
  for (unsigned i = 0; i < examples->word_example_size(); ++i)
    lex_lm_.increment(examples->word_at(i), examples->word_context_at(i).words, 
                                            examples->word_context_at(i).tags, eng);
}

//update PYP model to remove old training examples
template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
void ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::updateRemove(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) {
  ParsedPypWeights<tOrder, aOrder>::updateRemove(examples, eng);
  for (unsigned i = 0; i < examples->word_example_size(); ++i)
    lex_lm_.decrement(examples->word_at(i), examples->word_context_at(i).words, 
                                            examples->word_context_at(i).tags, eng);
}

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
int ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::numWords() const {
  return lex_vocab_size_;
}

template<unsigned wOrder, unsigned waOrder, unsigned tOrder, unsigned aOrder>
int ParsedCALexPypWeights<wOrder, waOrder, tOrder, aOrder>::vocabSize() const {
  return numWords();
}

template class ParsedCALexPypWeights<wordLMOrderAS, wordTagLMOrderAS, tagLMOrderAS, actionLMOrderAS>;

//#if ((wordLMOrderAE != wordLMOrderAS) || (tagLMOrderAS != tagLMOrderAE) || (actionLMOrderAS != actionLMOrderAE))
//template class ParsedCALexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>;
//#endif

//template class ParsedCALexPypWeights<wordLMOrderE, tagLMOrderE, 1>;

}

