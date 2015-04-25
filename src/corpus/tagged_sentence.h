#ifndef _CORPUS_TAGSENT_H_
#define _CORPUS_TAGSENT_H_

#include "corpus/dict.h"
#include "corpus/sentence.h"

namespace oxlm {

class TaggedSentence: public Sentence {
 public:
  TaggedSentence();
  
  TaggedSentence(WordsList features);

  TaggedSentence(Words sent, Words tags, WordsList features);

  void print_tags(const boost::shared_ptr<Dict>& dict) const {
    for (auto tag: tags_)
      std::cout << dict->lookupTag(tag) << " ";
    std::cout << std::endl;
  }
  
  void push_tag(WordId t) {
   tags_.push_back(t);
  }

  void set_tag_at(WordIndex i, WordId t) {
    tags_.at(i) = t;
  }

  size_t size() const override {
    return features_.size();
  }
 
  size_t tags_length() const {
    return tags_.size();
  }

  size_t features_length() const {
    return features_.size();
  }

  WordId tag_at(WordIndex i) const {
    return tags_.at(i);
  }

  Words features_at(WordIndex i) const {
    return features_.at(i);
  }

 private:
  Words tags_;
  WordsList features_;

};

}

#endif
