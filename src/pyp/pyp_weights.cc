#include "pyp/pyp_weights.h"

namespace oxlm {

template<unsigned kOrder>
PypWeights<kOrder>::PypWeights(size_t vocab_size):
      lm(vocab_size, 1, 1, 1, 1) {}

//return negative log probability
template<unsigned kOrder>
double PypWeights<kOrder>::predict(WordId word, Words context) const {
  return -std::log(lm.prob(word, context));
}

template<unsigned kOrder>
double PypWeights<kOrder>::likelihood() const {
  return -lm.log_likelihood();
}

template<unsigned kOrder>
void PypWeights<kOrder>::resampleHyperparameters(MT19937& eng) {
  lm.resample_hyperparameters(eng);
  std::cerr << "  [LLH=" << likelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<unsigned kOrder>
void PypWeights<kOrder>::updateInsert(const DataSet& examples, MT19937& eng) {
  for (const auto& ex: examples) 
    lm.increment(ex.word, ex.context, eng);
}

//update PYP model to remove old training examples
template<unsigned kOrder>
void PypWeights<kOrder>::updateRemove(const DataSet& examples, MT19937& eng) {
  for (const auto& ex: examples) 
    lm.decrement(ex.word, ex.context, eng);
}

//update PYP model to insert one training example
template<unsigned kOrder>
void PypWeights<kOrder>::updateInsert(const DataPoint& example, MT19937& eng) {
  lm.increment(example.word, example.context, eng);
}

//update PYP model to remove one training example
template<unsigned kOrder>
void PypWeights<kOrder>::updateRemove(const DataPoint& example, MT19937& eng) {
  lm.decrement(example.word, example.context, eng);
}

template class PypWeights<LMOrder>;

}

