#include "corpus/parallel_sentence.h"

namespace oxlm {

ParallelSentence::ParallelSentence():
  in_sentence_(),
  out_sentence_(),
  id_(0)
  {
  }

ParallelSentence::ParallelSentence(Words in_sent):
  in_sentence_(in_sent),
  out_sentence_(),
  id_(0)
  {
  }

ParallelSentence::ParallelSentence(Words in_sent, int id):
  in_sentence_(in_sent),
  out_sentence_(),
  id_(id)
  {
  }

ParallelSentence::ParallelSentence(Words in_sent, Words out_sent, Indices alignment):
  in_sentence_(in_sent),
  out_sentence_(out_sent),
  alignment_(alignment),
  id_(0)
  {
  }

Context ParallelSentence::extractContext(int position, int out_ctx_size, int in_window_size) const {
  WordsList features;
  int sos = 0;
  int eos = 1;

  //output features
  for (int i = position - out_ctx_size; i < position; ++i) {
    if (i >= 0) 
      features.push_back(Words(1,out_word_at(i)));
    else 
      features.push_back(Words(1, sos));
  }

  //input features
  int in_position = alignment_at(position);
  for (int i = in_position - in_window_size; i < in_position; ++i) {
    if (i >= 0) 
      features.push_back(Words(1, in_word_at(i)));
    else 
      features.push_back(Words(1, sos));
  }
  
  for (int i = in_position; i <= in_position + in_window_size; ++i) {
    if (i < in_size()) 
      features.push_back(Words(1, in_word_at(i)));
    else 
      features.push_back(Words(1, eos));
  }

  return Context(Words(), features);
}

}

