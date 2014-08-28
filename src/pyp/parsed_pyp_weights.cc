#include "pyp/parsed_pyp_weights.h"

namespace oxlm {

template<wOrder, aOrder>
ParsedPypWeights<wOrder, aOrder>::ParsedPypWeights(size_t vocab_size, size_t num_actions):
  PypWeights<wOrder>(vocab_size),    
  action_lm(num_actions, 1, 1, 1, 1) {}

template<wOrder, aOrder>
ParsedPypWeights<wOrder, aOrder>::predictWord(WordId word, Words context) {
  return 0;
}

template<wOrder, aOrder>
ParsedPypWeights<wOrder, aOrder>::predictTag(WordId tag, Words context) {
  return PypWeights<wOrder>::predict(tag, context);
}

template<wOrder, aOrder>
ParsedPypWeights<wOrder, aOrder>::predictAction(WordId action, Words context) {
  return -std::log(action_lm.prob(action, context));
}

//update PYP model to insert new training examples 
template<wOrder, aOrder>
ParsedPypWeights<wOrder, aOrder>::updateInsert(const DataSet& tag_examples, const DataSet& action_examples) {
  PypWeights<wOrder>::updateInsert(tag_examples);
  for (const auto& ex: action_examples) 
    action_lm->increment(ex.prediction, ex.context, eng);
}

//update PYP model to remove old training examples
template<wOrder>
ParsedPypWeights<wOrder, aOrder>::updateRemove(const DataSet& tag_examples, const DataSeet& action_examples) {
  PypWeights<wOrder>::updateInsert(tag_examples);
  for (const auto& ex: action_examples) 
    action_lm->decrement(ex.prediction, ex.context, eng);
}

template class ParsedPypWeights<tagLMOrder, actionLMOrder>;

}

