#include "pyp/pyp_weights.h"

namespace oxlm {

template <unsigned kOrder>
PypWeights<kOrder>::PypWeights(size_t vocab_size)
    : lm_(vocab_size, 1, 1, 1, 1), vocab_size_(vocab_size) {}

template <unsigned kOrder>
Real PypWeights<kOrder>::predict(WordId word, Context context) const {
  Real prob = lm_.prob(word, context.words);
  return -std::log(prob);
}

template <unsigned kOrder>
Reals PypWeights<kOrder>::predict(Context context) const {
  Reals weights(vocab_size_, 0);
  for (int i = 0; i < vocab_size_; ++i) {
    weights[i] = predict(i, context.words);
  }
  return weights;
}

template <unsigned kOrder>
Real PypWeights<kOrder>::likelihood() const {
  return -lm_.log_likelihood();
}

template <unsigned kOrder>
void PypWeights<kOrder>::resampleHyperparameters(MT19937& eng) {
  lm_.resample_hyperparameters(eng);
  std::cerr << "  [LLH=" << likelihood() << "]\n\n";
}

template <unsigned kOrder>
void PypWeights<kOrder>::updateInsert(
    const boost::shared_ptr<DataSet>& examples, MT19937& eng) {
  for (unsigned i = 0; i < examples->size(); ++i) {
    lm_.increment(examples->wordAt(i), examples->contextAt(i).words, eng);
  }
}

template <unsigned kOrder>
void PypWeights<kOrder>::updateRemove(
    const boost::shared_ptr<DataSet>& examples, MT19937& eng) {
  for (unsigned i = 0; i < examples->size(); ++i) {
    lm_.decrement(examples->wordAt(i), examples->contextAt(i).words, eng);
  }
}

template <unsigned kOrder>
void PypWeights<kOrder>::updateInsert(const DataPoint& example, MT19937& eng) {
  lm_.increment(example.word, example.context.words, eng);
}

template <unsigned kOrder>
void PypWeights<kOrder>::updateRemove(const DataPoint& example, MT19937& eng) {
  lm_.decrement(example.word, example.context.words, eng);
}

template <unsigned kOrder>
int PypWeights<kOrder>::vocabSize() const {
  return vocab_size_;
}

template class PypWeights<tagLMOrderAS>;

#if (wordLMOrderAS != tagLMOrderAS)
template class PypWeights<wordLMOrderAS>;
#endif

#if ((wordLMOrder != tagLMOrderAS) && (wordLMOrder != wordLMOrderAS))
template class PypWeights<wordLMOrder>;
#endif

}  // namespace oxlm

