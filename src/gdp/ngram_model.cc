#include "gdp/ngram_model.h"

namespace oxlm {

template<class Weights>
NGramModel<Weights>::NGramModel(unsigned order, WordId sos, WordId eos):
    order_(order),
    sos_(sos), 
    eos_(eos) {
        std::cout << "constructed" << std::endl;
    }

template<class Weights>
Words NGramModel<Weights>::extractContext(const boost::shared_ptr<Corpus> corpus, int position) {
  Words context;

  // The context is constructed starting from the most recent word:
  // context = [w_{n-1}, w_{n-2}, ...]
  // different order - but will it matter if we change it around??
  int context_start = position - order_ + 1;
  bool sentence_start = (position == 0); 
  for (int i = order_ - 2; i >= 0; --i) {
    int index = context_start + i;
    sentence_start |= (index < 0 || corpus->at(index) == eos_);
    int word_id = sentence_start ? sos_: corpus->at(index);
    //std::cout << word_id << " ";
    context.push_back(word_id);
  }

  //std::cout << std::endl;
  return context;
}

template<class Weights>
void NGramModel<Weights>::extract(const boost::shared_ptr<Corpus> corpus, int position,
    const boost::shared_ptr<DataSet>& examples) {
  WordId word = corpus->at(position);
  Words context = extractContext(corpus, position);
  examples->addExample(DataPoint(word, context));  
}

template<class Weights>
Real NGramModel<Weights>::evaluate(const boost::shared_ptr<Corpus> corpus, int position, 
          const boost::shared_ptr<Weights>& weights) {
  WordId word = corpus->at(position);
  Words context = extractContext(corpus, position);
  return weights->predict(word, context);
}

template<class Weights>
void NGramModel<Weights>::extractSentence(const Sentence& sent, 
          const boost::shared_ptr<DataSet>& examples) {
  Words context(order_ - 1, sos_);
  //eos is already at end of sentence
  for (int i = 0; i < sent.size(); ++i) {    
    WordId word = sent.word_at(i);
    Words ctx = Words(context.begin() + i, context.begin() + i + order_ - 1);
    //std::reverse(ctx.begin(), ctx.end());
    DataPoint example(word, ctx); 
    examples->addExample(example);
    context.push_back(word);
  }    
}

template<class Weights>
Real NGramModel<Weights>::evaluateSentence(const Sentence& sent, 
          const boost::shared_ptr<Weights>& weights) {
  Real weight = 0;
  Words context(order_ -1, sos_);
  //eos is already at end of sentence
  for (int i = 0; i < static_cast<int>(sent.size()); ++i) {    
    WordId word = sent.word_at(i);
    Words ctx = Words(context.begin() + i, context.begin() + i + order_ - 1);
    //std::reverse(ctx.begin(), ctx.end());
    Real predict_weight = weights->predict(word, ctx); 

    //if (word == -1)
    //  std::cout << predict_weight << " ";
    weight += predict_weight;
    context.push_back(word);
  }  

  return weight;
}

template<class Weights>
Sentence NGramModel<Weights>::generateSentence(const boost::shared_ptr<Weights>& weights, MT19937& eng) {
  unsigned sent_limit = 100;
  Words sent(order_ - 1, sos_);
  bool terminate = false;
  Real weight = 0;
  while (sent.back() != eos_) {
    Reals word_distr = weights->predict(sent);
    //word_distr[0] = L_MAX;  

    multinomial_distribution_log<Real> w_mult(word_distr);
    WordId word = w_mult(eng);
    weight += word_distr[word];

    sent.push_back(word);
   
  }
  std::cout << weight << "  ";
  return Sentence(sent);
}

template class NGramModel<PypWeights<wordLMOrder>>;
template class NGramModel<Weights>;
template class NGramModel<FactoredWeights>;

}
