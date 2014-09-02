#include "pyp/pyp_weights.h"

namespace oxlm {

template<unsigned kOrder>
PypWeights<kOrder>::PypWeights(size_t vocab_size):
      lm_(vocab_size, 1, 1, 1, 1),
      vocab_size_(vocab_size) {}

//return negative log probability
template<unsigned kOrder>
double PypWeights<kOrder>::predict(WordId word, Words context) const {
  return -std::log(lm_.prob(word, context));
}

template<unsigned kOrder>
double PypWeights<kOrder>::likelihood() const {
  return -lm_.log_likelihood();
}

template<unsigned kOrder>
void PypWeights<kOrder>::resampleHyperparameters(MT19937& eng) {
  lm_.resample_hyperparameters(eng);
  std::cerr << "  [LLH=" << likelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<unsigned kOrder>
void PypWeights<kOrder>::updateInsert(const DataSet& examples, MT19937& eng) {
  for (const auto& ex: examples) 
    lm_.increment(ex.word, ex.context, eng);
}

//update PYP model to remove old training examples
template<unsigned kOrder>
void PypWeights<kOrder>::updateRemove(const DataSet& examples, MT19937& eng) {
  for (const auto& ex: examples) 
    lm_.decrement(ex.word, ex.context, eng);
}

//update PYP model to insert one training example
template<unsigned kOrder>
void PypWeights<kOrder>::updateInsert(const DataPoint& example, MT19937& eng) {
  lm_.increment(example.word, example.context, eng);
}

//update PYP model to remove one training example
template<unsigned kOrder>
void PypWeights<kOrder>::updateRemove(const DataPoint& example, MT19937& eng) {
  lm_.decrement(example.word, example.context, eng);
}

template<unsigned kOrder>
size_t PypWeights<kOrder>::vocabSize() const {
  return vocab_size_;
}

template class PypWeights<LMOrder>;

}

