#include "pyp/pyp_weights.h"

namespace oxlm {

template<unsigned kOrder>
PypWeights<kOrder>::PypWeights(size_t vocab_size):
      lm_(vocab_size, 1, 1, 1, 1),
      vocab_size_(vocab_size) {}

//return negative log probability
template<unsigned kOrder>
Real PypWeights<kOrder>::predict(WordId word, Words context) const {
  Real prob = lm_.prob(word, context);
  //if (prob <= 0 || prob >= 1)
  //  std::cout << prob << " ";
  return -std::log(prob);
}

template<unsigned kOrder>
Real PypWeights<kOrder>::likelihood() const {
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
  for (unsigned i = 0; i < examples.size(); ++i)
    lm_.increment(examples.wordAt(i), examples.contextAt(i), eng);
}

//update PYP model to remove old training examples
template<unsigned kOrder>
void PypWeights<kOrder>::updateRemove(const DataSet& examples, MT19937& eng) {
  for (unsigned i = 0; i < examples.size(); ++i)
    lm_.decrement(examples.wordAt(i), examples.contextAt(i), eng);
}

//update PYP model to insert new training examples 
template<unsigned kOrder>
void PypWeights<kOrder>::updateInsert(const DataPoints& examples, MT19937& eng) {
  for (unsigned i = 0; i < examples.size(); ++i)
    lm_.increment(examples.at(i).word, examples.at(i).context, eng);
}

//update PYP model to remove old training examples
template<unsigned kOrder>
void PypWeights<kOrder>::updateRemove(const DataPoints& examples, MT19937& eng) {
  for (unsigned i = 0; i < examples.size(); ++i)
    lm_.decrement(examples.at(i).word, examples.at(i).context, eng);
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
int PypWeights<kOrder>::vocabSize() const {
  return vocab_size_;
}

template class PypWeights<tagLMOrderAS>;

#if (wordLMOrderAS != tagLMOrderAS)
template class PypWeights<wordLMOrderAS>;
#endif

#if ((tagLMOrderAE != tagLMOrderAS) && (tagLMOrderAE != wordLMOrderAS))
template class PypWeights<tagLMOrderAE>;
#endif

#if ((wordLMOrderAE != tagLMOrderAS) && (wordLMOrderAE != wordLMOrderAS) && (wordLMOrderAE != tagLMOrderAE))
template class PypWeights<wordLMOrderAE>;
#endif

#if ((tagLMOrderE != wordLMOrderAE) && (tagLMOrderE != tagLMOrderAS) && (tagLMOrderE != wordLMOrderAS) && (tagLMOrderE != tagLMOrderAE))
template class PypWeights<tagLMOrderE>;
#endif

#if ((wordLMOrderE != tagLMOrderE) && (wordLMOrderE != wordLMOrderAE) && (wordLMOrderE != tagLMOrderAS) && (wordLMOrderE != wordLMOrderAS) && (wordLMOrderE != tagLMOrderAE))
template class PypWeights<wordLMOrderE>;
#endif

#if ((wordLMOrder != wordLMOrderE) && (wordLMOrder != tagLMOrderE) && (wordLMOrder != wordLMOrderAE) && (wordLMOrder != tagLMOrderAS) && (wordLMOrder != wordLMOrderAS) && (wordLMOrder != tagLMOrderAE))
template class PypWeights<wordLMOrder>;
#endif

}

