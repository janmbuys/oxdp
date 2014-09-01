#include "pyp/parsed_pyp_weights.h"

namespace oxlm {

template<unsigned tOrder, unsigned aOrder>
ParsedPypWeights<tOrder, aOrder>::ParsedPypWeights(size_t num_tags, size_t num_actions):
  PypWeights<tOrder>(num_tags),    
  action_lm(num_actions, 1, 1, 1, 1) {}

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
  return -std::log(action_lm.prob(action, context));
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
  return -action_lm.log_likelihood();
}

template<unsigned tOrder, unsigned aOrder>
double ParsedPypWeights<tOrder, aOrder>::likelihood() const {
  return tagLikelihood() + actionLikelihood();
}

template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::resampleHyperparameters(MT19937& eng) {
  PypWeights<tOrder>::resampleHyperparameters(eng);
  action_lm.resample_hyperparameters(eng);
  std::cerr << "  [Action LLH=" << actionLikelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::updateInsert(const DataSet& tag_examples, 
        const DataSet& action_examples, MT19937& eng) {
  PypWeights<tOrder>::updateInsert(tag_examples, eng);
  for (const auto& ex: action_examples) 
    action_lm.increment(ex.word, ex.context, eng);
}

//update PYP model to remove old training examples
template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::updateRemove(const DataSet& tag_examples, 
        const DataSet& action_examples, MT19937& eng) {
  PypWeights<tOrder>::updateRemove(tag_examples, eng);
  for (const auto& ex: action_examples) 
    action_lm.decrement(ex.word, ex.context, eng);
}

template class ParsedPypWeights<tagLMOrder, actionLMOrder>;

}

