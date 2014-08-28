#include "pyp/pyp_weights.h"

namespace oxlm {

template<wOrder>
PypWeights<wOrder>::PypWeights(size_t vocab_size):
      word_lm(vocab_size, 1, 1, 1, 1),
      eng() {}

//return negative log probability
template<wOrder>
PypWeights<wOrder>::predict(WordId word, Words context) {
  return -std::log(word_lm.prob(word, context));
}

//update PYP model to insert new training examples 
template<wOrder>
void PypWeights<wOrder>::updateInsert(const DataSet& examples) {
  for (const auto& ex: examples) 
    word_lm->increment(ex.prediction, ex.context, eng);
}

//update PYP model to remove old training examples
template<wOrder>
void PypWeights<wOrder>::updateRemove(const DataSet& examples) {
  for (const auto& ex: examples) 
    word_lm->decrement(ex.word, ex.context, eng);
}

template class PypWeights<wordLMOrder>;

}

