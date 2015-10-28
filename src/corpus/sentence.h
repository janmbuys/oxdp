#ifndef _CORPUS_SENT_H_
#define _CORPUS_SENT_H_

#include "corpus/dict.h"

namespace oxlm {

// Class that represent a sentence (a sequence of words).  
class Sentence {
 public:
  Sentence();

  Sentence(Words sent);

  Sentence(Words sent, int id);

  void push_word(WordId w) { sentence_.push_back(w); }

  void print_sentence(const boost::shared_ptr<Dict>& dict) const {
    for (auto word : sentence_) {
      std::cout << dict->lookup(word) << " ";
    }
    std::cout << std::endl;
  }

  std::string sentence_string(const boost::shared_ptr<Dict>& dict) const {
    std::string sent = "";
    for (auto word : sentence_) {
      sent += dict->lookup(word) + " ";
    }
    return sent;
  }

  void set_id(int id) { id_ = id; }

  int id() const { return id_; }

  virtual size_t size() const { return sentence_.size(); }

  WordId word_at(WordIndex i) const { return sentence_.at(i); }

 private:
  Words sentence_;
  int id_;
};

}  // namespace oxlm

#endif
