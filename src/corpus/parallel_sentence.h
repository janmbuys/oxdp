#ifndef _CORPUS_LL_SENT_H_
#define _CORPUS_LL_SENT_H_

#include "corpus/dict.h"
#include "corpus/context.h"

namespace oxlm {

class ParallelSentence {
  public:

  ParallelSentence();

  ParallelSentence(Words in_sent); //only incorrect
  
  ParallelSentence(Words in_sent, int id);
  
  ParallelSentence(Words in_sent, Words out_sent, Indices alignment); 

  Context extractContext(int position, int out_ctx_size, int in_window_size) const;

  void push_in_word(WordId w) {
   in_sentence_.push_back(w);
  }

  void push_out_word(WordId w) {
   out_sentence_.push_back(w);
  }

  void push_alignment(WordIndex al) {
   alignment_.push_back(al);
  }

  void print_in_sentence(const boost::shared_ptr<Dict>& dict) const {
    for (auto word: in_sentence_)
      std::cout << dict->lookup(word) << " ";
    std::cout << std::endl;
  }

  void print_out_sentence(const boost::shared_ptr<Dict>& dict) const {
    for (auto word: out_sentence_)
      std::cout << dict->lookup(word) << " ";
    std::cout << std::endl;
  }

  std::string out_sentence_string(const boost::shared_ptr<Dict>& dict) const {
    std::string sent = "";
    for (unsigned i = 0; i < out_sentence_.size() - 1; ++i)  //exclude eos
      sent += dict->lookup(out_sentence_.at(i)) + " ";
    return sent;   
  }

  void set_id(int id) {
    id_ = id;
  }

  void set_alignment_at(WordIndex i, WordIndex j) {
    alignment_.at(i) = j;
  }

  void set_weight(Real w) {
    weight_ = w;
  }

  void add_weight(Real w) {
    weight_ += w;
  }

  int id() const {
    return id_;
  }

  Real weight() const {
    return weight_;
  }

  size_t in_size() const {
    return in_sentence_.size();
  }

  size_t out_size() const {
    return out_sentence_.size();
  }

  WordId in_word_at(WordIndex i) const {
    return in_sentence_.at(i);
  }

  WordId out_word_at(WordIndex i) const {
    return out_sentence_.at(i);
  }

  WordIndex alignment_at(WordIndex i) const {
    return alignment_.at(i);
  }

  WordId aligned_word_at(WordIndex i) const {
    return out_sentence_.at(alignment_.at(i));
  }

  Words in_sentence() const {
    return in_sentence_;
  }

  Words out_sentence() const {
    return in_sentence_;
  }

  static bool cmp_weights(const boost::shared_ptr<ParallelSentence>& p1, 
                          const boost::shared_ptr<ParallelSentence>& p2) {
    return (p1->weight() < p2->weight());
  }

  private:
  Words in_sentence_;
  Words out_sentence_;
  Indices alignment_; //same length as output
  int id_;
  Real weight_;
};


}

#endif
