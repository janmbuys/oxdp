#ifndef _CORPUS_TAGSENT_H_
#define _CORPUS_TAGSENT_H_

#include "corpus/dict.h"
#include "corpus/sentence.h"

namespace oxlm {

class TaggedSentence: public Sentence {
  public:

  TaggedSentence();
  
  TaggedSentence(Words tags);

  TaggedSentence(Words sent, Words tags);

  void print_tags(Dict& dict) const {
    for (auto tag: tags_)
      std::cout << dict.lookup_tag(tag) << " ";
    std::cout << std::endl;
  }
  
  void push_tag(WordId t) {
   tags_.push_back(t);
  }

  //overriding base class method
  size_t size() const override {
    return tags_.size();
  }

  /*
   Words tags() const {
    return tags_;
  } */
  
  size_t tags_length() const {
    return tags_.size();
  }

  WordId tag_at(WordIndex i) const {
    return tags_.at(i);
  }

  private:
  Words tags_;

};


}

#endif
