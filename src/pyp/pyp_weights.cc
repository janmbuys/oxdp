#include "pyp/pyp_weights.h"

namespace oxlm {

template<kOrder>
PypWeights<kOrder>::PypWeights(size_t vocab_size):
      lm(vocab_size, 1, 1, 1, 1),
      eng() {}

//return negative log probability
template<kOrder>
PypWeights<kOrder>::predict(WordId word, Words context) {
  return -std::log(lm.prob(word, context));
}

template<kOrder>
double PypWeights<kOrder>::likelihood() {
  return -lm.log_likelihood();
}

template<kOrder>
void PypWeights<kOrder>::resampleHyperparameters() {
  lm.resample_hyperparameters(eng);
  std::cerr << "  [LLH=" << likelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<kOrder>
void PypWeights<kOrder>::updateInsert(const DataSet& examples) {
  for (const auto& ex: examples) 
    lm->increment(ex.prediction, ex.context, eng);
}

//update PYP model to remove old training examples
template<kOrder>
void PypWeights<kOrder>::updateRemove(const DataSet& examples) {
  for (const auto& ex: examples) 
    lm->decrement(ex.word, ex.context, eng);
}

//update PYP model to insert one training example
template<kOrder>
void PypWeights<kOrder>::updateInsert(const DataPoint& example) {
  lm->increment(example.prediction, example.context, eng);
}

//update PYP model to remove one training example
template<kOrder>
void PypWeights<kOrder>::updateRemove(const DataPoint& example) {
  lm->decrement(example.word, example.context, eng);
}

template class PypWeights<LMOrder>;

}

