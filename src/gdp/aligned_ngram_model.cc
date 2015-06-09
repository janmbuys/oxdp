#include "gdp/aligned_ngram_model.h"

namespace oxlm {

template<class Weights>
AlignedNGramModel<Weights>::AlignedNGramModel(unsigned out_ctx_size, unsigned in_window_size, WordId sos, WordId eos):
    out_ctx_size_(out_ctx_size),
    in_window_size_(in_window_size),
    sos_(sos), 
    eos_(eos) {
        std::cout << "constructed" << std::endl;
    }

/*
template<class Weights>
void AlignedNGramModel<Weights>::extract(const boost::shared_ptr<Corpus> corpus, int position,
    const boost::shared_ptr<DataSet>& examples) {
  WordId word = corpus->at(position);
  Context context = extractContext(corpus, position);
  examples->addExample(DataPoint(word, context));  
}

template<class Weights>
Real AlignedNGramModel<Weights>::evaluate(const boost::shared_ptr<Corpus> corpus, int position, 
          const boost::shared_ptr<Weights>& weights) {
  WordId word = corpus->at(position);
  Context context = extractContext(corpus, position);
  return weights->predict(word, context);
}


template<class Weights>
Context AlignedNGramModel<Weights>::extractContext(const ParallelSentence& sent, int position) {
  WordsList features;

  //output features
  for (int i = position - out_ctx_size_; i < position; ++i) {
    if (i >= 0) 
      features.push_back(Words(1, sent.out_word_at(i)));
    else 
      features.push_back(Words(1, sos_));
  }

  //input features
  int in_position = sent.alignment_at(position);
  for (int i = in_position - in_window_size_; i < in_position; ++i) {
    if (i >= 0) 
      features.push_back(Words(1, sent.in_word_at(i)));
    else 
      features.push_back(Words(1, sos_));
  }
  
  for (int i = in_position; i <= in_position + in_window_size_; ++i) {
    if (i < sent.in_size()) 
      features.push_back(Words(1, sent.in_word_at(i)));
    else 
      features.push_back(Words(1, eos_));
  }

  return Context(Words(), features);
} */

template<class Weights>
void AlignedNGramModel<Weights>::extractSentence(const ParallelSentence& sent, 
          const boost::shared_ptr<DataSet>& examples) {
  //not optimally efficient, but ok for now
  for (int i = 0; i < sent.out_size(); ++i) {    
    Context context = sent.extractContext(i, out_ctx_size_, in_window_size_);
    WordId word = sent.out_word_at(i);
    DataPoint example(word, context); 
    examples->addExample(example);
  }    
}

template<class Weights>
Real AlignedNGramModel<Weights>::evaluateSentence(const ParallelSentence& sent, 
          const boost::shared_ptr<Weights>& weights) {
  Real weight = 0;
  for (int i = 0; i < static_cast<int>(sent.out_size()); ++i) {    
    WordId word = sent.out_word_at(i);
    Context context = sent.extractContext(i, out_ctx_size_, in_window_size_);
    Real predict_weight = weights->predict(word, context); 
    weight += predict_weight;
  }  

  return weight;
}

template<class Weights>
ParallelSentence AlignedNGramModel<Weights>::generateSentence(const Words& in_sent, const boost::shared_ptr<Weights>& weights, int beam_size, int max_beam_increment) {
  std::vector<boost::shared_ptr<ParallelSentence>> beam_stack; 
  beam_stack.push_back(boost::make_shared<ParallelSentence>(in_sent)); 
  //ParallelSentence sent(in_sent);
  
  //assume strictly monotonic alignment for now
  int i = 0;
  int init_position = 0; //predict
  beam_stack[0]->push_alignment(init_position);
  
  while (beam_stack[0]->aligned_word_at(i) != eos_) {
    //prune stack
    if (beam_stack.size() > beam_size) {
      std::sort(beam_stack.begin(), beam_stack.end(), ParallelSentence::cmp_weights); 
      for (int j = beam_stack.size()- 1; (j >= beam_size); --j)
        beam_stack.pop_back();
    }
    
    unsigned stack_size = beam_stack.size();
    for (unsigned j = 0; (j < stack_size); ++j) {
      Context context = beam_stack[j]->extractContext(i, out_ctx_size_, in_window_size_);
      Reals word_distr = weights->predictViterbi(context); //approx best, 
      WordId best_word = arg_min(word_distr, 0); 
      Real best_weight = word_distr[best_word];
      
      beam_stack.push_back(boost::make_shared<ParallelSentence>(*beam_stack[j]));
      for (int k = 1; k < max_beam_increment; ++k) {
        word_distr[arg_min(word_distr, 0)] = std::numeric_limits<Real>::max();
        WordId word = arg_min(word_distr, 0); 
        //std::cout << word_distr[0] <<  "->" << word << std::endl;
        beam_stack.back()->push_out_word(word);
        beam_stack.back()->add_weight(word_distr[word]);

        int in_position = i + 1; //predict
        beam_stack.back()->push_alignment(in_position);
      }

      beam_stack[j]->push_out_word(best_word);
      beam_stack[j]->add_weight(best_weight);

      int in_position = i + 1; //predict
      beam_stack[j]->push_alignment(in_position);

    }
    ++i;
  }

  for (int j = 0; j < beam_stack.size(); ++j)
    beam_stack[j]->push_out_word(eos_);


  std::sort(beam_stack.begin(), beam_stack.end(), ParallelSentence::cmp_weights); 
  return ParallelSentence(*beam_stack[0]);
}

//template class AlignedNGramModel<PypWeights<wordLMOrder>>;
template class AlignedNGramModel<Weights>;
template class AlignedNGramModel<FactoredWeights>;

}
