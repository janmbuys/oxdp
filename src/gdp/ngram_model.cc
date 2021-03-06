#include "gdp/ngram_model.h"

namespace oxlm {

template <class Weights>
NGramModel<Weights>::NGramModel(unsigned order, WordId sos, WordId eos)
    : order_(order), sos_(sos), eos_(eos) {
  std::cout << "constructed" << std::endl;
}

template <class Weights>
Context NGramModel<Weights>::extractContext(const Sentence& sent,
                                            int position) {
  Words context;
  WordsList features;

  int context_start = position - order_ + 1;
  for (int i = 0; i < order_ - 1; ++i) {
    int index = context_start + i;
    bool sentence_start = (index < 0);
    int word_id = sentence_start ? sos_ : sent.word_at(index);
    context.push_back(word_id);
    features.push_back(Words(1, word_id));
  }

  return Context(context, features);
}

template <class Weights>
Context NGramModel<Weights>::extractContext(
    const boost::shared_ptr<Corpus> corpus, int position) {
  Words context;
  WordsList features;

  int context_start = position - order_ + 1;
  for (int i = 0; i < order_ - 1; ++i) {
    int index = context_start + i;
    bool sentence_start = (index < 0);
    int word_id = sentence_start ? sos_ : corpus->at(index);
    context.push_back(word_id);
    features.push_back(Words(1, word_id));
  }

  return Context(context, features);
}

template <class Weights>
void NGramModel<Weights>::extract(const boost::shared_ptr<Corpus> corpus,
                                  int position,
                                  const boost::shared_ptr<DataSet>& examples) {
  WordId word = corpus->at(position);
  Context context = extractContext(corpus, position);
  examples->addExample(DataPoint(word, context));
}

template <class Weights>
Real NGramModel<Weights>::evaluate(const boost::shared_ptr<Corpus> corpus,
                                   int position,
                                   const boost::shared_ptr<Weights>& weights) {
  WordId word = corpus->at(position);
  Context context = extractContext(corpus, position);
  return weights->predict(word, context);
}

template <class Weights>
void NGramModel<Weights>::extractSentence(
    const Sentence& sent, const boost::shared_ptr<DataSet>& examples) {
  for (int i = 0; i < sent.size(); ++i) {
    WordId word = sent.word_at(i);
    Context ctx = extractContext(sent, i);
    DataPoint example(word, ctx);
    examples->addExample(example);
  }
}

template <class Weights>
Real NGramModel<Weights>::evaluateSentence(
    const Sentence& sent, const boost::shared_ptr<Weights>& weights) {
  Real weight = 0;
  for (int i = 0; i < static_cast<int>(sent.size()); ++i) {
    WordId word = sent.word_at(i);
    Context ctx = extractContext(sent, i);
    Real predict_weight = weights->predict(word, ctx);
    weight += predict_weight;
  }

  return weight;
}

template <class Weights>
Parser NGramModel<Weights>::generateSentence(
    const boost::shared_ptr<Weights>& weights, MT19937& eng) {
  unsigned sent_limit = 100;
  Words sent(order_ - 1, sos_);
  bool terminate = false;
  Real weight = 0;
  while (sent.back() != eos_) {
    Reals word_distr = weights->predict(sent);

    multinomial_distribution_log<Real> w_mult(word_distr);
    WordId word = w_mult(eng);
    weight += word_distr[word];

    sent.push_back(word);
  }

  Parser parse = Parser(sent);
  parse.set_weight(weight);
  return parse;
}

template class NGramModel<PypWeights<wordLMOrder>>;
template class NGramModel<Weights>;
template class NGramModel<FactoredWeights>;

}  // namespace oxlm
