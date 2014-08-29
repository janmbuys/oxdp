#include "pyp/parsed_pyp_weights.h"

namespace oxlm {

template<tOrder, aOrder>
ParsedPypWeights<tOrder, aOrder>::ParsedPypWeights(size_t num_tags, size_t num_actions):
  PypWeights<tOrder>(num_tags),    
  action_lm(num_actions, 1, 1, 1, 1) {}

template<tOrder, aOrder>
ParsedPypWeights<tOrder, aOrder>::predictWord(WordId word, Words context) {
  return 0;
}

template<tOrder, aOrder>
ParsedPypWeights<tOrder, aOrder>::predictTag(WordId tag, Words context) {
  return PypWeights<tOrder>::predict(tag, context);
}

template<tOrder, aOrder>
ParsedPypWeights<tOrder, aOrder>::predictAction(WordId action, Words context) {
  return -std::log(action_lm.prob(action, context));
}

template<tOrder, aOrder>
double ParsedPypWeights<tOrder, aOrder>::wordLikelihood() {
  return 0;
}

template<tOrder, aOrder>
double ParsedPypWeights<tOrder, aOrder>::tagLikelihood() {
  return PypWeights::likelihood();
}

template<tOrder, aOrder>
double ParsedPypWeights<tOrder, aOrder>::actionLikelihood() {
  return -action_lm.log_likelihood();
}

template<tOrder, aOrder>
double ParsedPypWeights<tOrder, aOrder>::likelihood() {
  return tagLikelihood() + actionLikelihood();
}

template<tOrder, aOrder>
void ParsedPypWeights<tOrder, aOrder>::resampleHyperparameters() {
  PypWeights<tOrder>::resampleHyperparameters();
  action_lm.resample_hyperparameters(eng);
  std::cerr << "  [Action LLH=" << actionLikelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<tOrder, aOrder>
ParsedPypWeights<tOrder, aOrder>::updateInsert(const DataSet& tag_examples, 
        const DataSet& action_examples) {
  updateInsert(tag_examples);
  for (const auto& ex: action_examples) 
    action_lm->increment(ex.prediction, ex.context, eng);
}

//update PYP model to remove old training examples
template<tOrder>
ParsedPypWeights<tOrder, aOrder>::updateRemove(const DataSet& tag_examples, 
        const DataSet& action_examples) {
  updateRemove(tag_examples);
  for (const auto& ex: action_examples) 
    action_lm->decrement(ex.prediction, ex.context, eng);
}

template class ParsedPypWeights<tagLMOrder, actionLMOrder>;

}

