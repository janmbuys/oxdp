#include "corpus/ngram_model.h"

namespace oxlm {

NGramModel::NGramModel(unsigned order, WordId eos):
    order_(order),
    eos_(eos) {}

void NGramModel::extractSentence(const Sentence& sent, 
          const boost::shared_ptr<DataSet>& examples) {
  Words ctx(ngram_order_ - 1, eos_);
  //eos is already at end of sentence
  for (int i = 0; i < sent.size(); ++i) {    
    //ctx(i, i+ngram_order-1)
    DataPoint example(sent.at(i), Words(ctx.begin() + i, ctx.begin() + i + ngram_order_ - 1));
    examples->addExample(example);
    ctx.push_back(sent.at(i));
  }    
  
}

double NGramModel::evaluateSentence(const Sentence& sent, 
          const boost::shared_ptr<WeightsInterface>& weights) {
  double weight = 0;
  Words ctx(ngram_order_ -1, eos_);
  //eos is already at end of sentence
  for (int i = 0; i < sent.size(); ++i) {    
    //ctx(i, i+ngram_order-1)
    weight += weights->predict(sent.at(i), Words(ctx.begin() + i, ctx.begin() + i + ngram_order_ - 1));
    ctx.push_back(sent.at(i));
  }  

  return weight;
}

}
