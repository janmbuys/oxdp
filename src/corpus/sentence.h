#ifndef _CORPUS_SENT_H_
#define _CORPUS_SENT_H_

#include "corpus/dict.h"

namespace oxlm {

class Sentence {
  public:

  Sentence();

  Sentence(Words sent);

  void push_word(WordId w) {
   sentence_.push_back(w);
  }

  void print_sentence(const boost::shared_ptr<Dict>& dict) const {
    for (auto word: sentence_)
      std::cout << dict->lookup(word) << " ";
    std::cout << std::endl;
  }

  virtual size_t size() const {
    return sentence_.size();
  }

  WordId word_at(WordIndex i) const {
    return sentence_.at(i);
  }

  private:
  Words sentence_;

};


}

#endif
