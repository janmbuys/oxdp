#include "pyp/parsed_pyp_weights.h"

namespace oxlm {

template<unsigned tOrder, unsigned aOrder>
ParsedPypWeights<tOrder, aOrder>::ParsedPypWeights(size_t num_tags, size_t num_actions):
  PypWeights<tOrder>(num_tags),    
  action_lm_(num_actions, 1, 1, 1, 1),
  num_actions_(num_actions) {}

template<unsigned tOrder, unsigned aOrder> 
double ParsedPypWeights<tOrder, aOrder>::predict(WordId word, Words context) const {
  return predictTag(word, context);
}

template<unsigned tOrder, unsigned aOrder>
double ParsedPypWeights<tOrder, aOrder>::predictWord(WordId word, Words context) const {
  return 0;
}

template<unsigned tOrder, unsigned aOrder>
double ParsedPypWeights<tOrder, aOrder>::predictTag(WordId tag, Words context) const {
  return PypWeights<tOrder>::predict(tag, context);
}

template<unsigned tOrder, unsigned aOrder>
double ParsedPypWeights<tOrder, aOrder>::predictAction(WordId action, Words context) const {
  return -std::log(action_lm_.prob(action, context));
}

template<unsigned tOrder, unsigned aOrder>
double ParsedPypWeights<tOrder, aOrder>::wordLikelihood() const {
  return 0;
}

template<unsigned tOrder, unsigned aOrder>
double ParsedPypWeights<tOrder, aOrder>::tagLikelihood() const {
  return PypWeights<tOrder>::likelihood();
}

template<unsigned tOrder, unsigned aOrder>
double ParsedPypWeights<tOrder, aOrder>::actionLikelihood() const {
  return -action_lm_.log_likelihood();
}

template<unsigned tOrder, unsigned aOrder>
double ParsedPypWeights<tOrder, aOrder>::likelihood() const {
  return tagLikelihood() + actionLikelihood();
}

template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::resampleHyperparameters(MT19937& eng) {
  PypWeights<tOrder>::resampleHyperparameters(eng);
  action_lm_.resample_hyperparameters(eng);
  std::cerr << "  [Action LLH=" << actionLikelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::updateInsert(const ParseDataSet& examples, MT19937& eng) {
  PypWeights<tOrder>::updateInsert(examples.tag_examples(), eng);
  for (unsigned i = 0; i < examples.size(); ++i)
    action_lm_.increment(examples.action_at(i), examples.action_context_at(i), eng);
}

//update PYP model to remove old training examples
template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::updateRemove(const ParseDataSet& examples, MT19937& eng) {
  PypWeights<tOrder>::updateRemove(examples.tag_examples(), eng);
  for (unsigned i = 0; i < examples.size(); ++i)
    action_lm_.decrement(examples.action_at(i), examples.action_context_at(i), eng);
}

//update PYP model to insert new training examples 
template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::updateInsert(const DataSet& examples, MT19937& eng) {
  PypWeights<tOrder>::updateInsert(examples, eng);
}

//update PYP model to remove old training examples
template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::updateRemove(const DataSet& examples, MT19937& eng) {
  PypWeights<tOrder>::updateRemove(examples, eng);
}

template<unsigned tOrder, unsigned aOrder>
size_t ParsedPypWeights<tOrder, aOrder>::numWords() const {
  return 1;
}

template<unsigned tOrder, unsigned aOrder>
size_t ParsedPypWeights<tOrder, aOrder>::numTags() const {
  return PypWeights<tOrder>::vocabSize();
}

template<unsigned tOrder, unsigned aOrder>
size_t ParsedPypWeights<tOrder, aOrder>::numActions() const {
  return num_actions_;
}

template<unsigned tOrder, unsigned aOrder>
size_t ParsedPypWeights<tOrder, aOrder>::vocabSize() const {
  return numTags();
}

template class ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>;

#if ((tagLMOrderAS != tagLMOrderAE) || (actionLMOrderAS != actionLMOrderAE))
template class ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>;
#endif

template class ParsedPypWeights<tagLMOrderE, 1>;

}

