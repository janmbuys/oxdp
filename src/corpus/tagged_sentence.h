#ifndef _CORPUS_TAGSENT_H_
#define _CORPUS_TAGSENT_H_

#include "corpus/dict.h"
#include "corpus/sentence.h"

namespace oxlm {

class TaggedSentence: public Sentence {
 public:
  TaggedSentence();
  
  TaggedSentence(WordsList features);

  TaggedSentence(Words sent, WordsList features);

  void print_tags(const boost::shared_ptr<Dict>& dict) const {
    for (auto item: features_)
      std::cout << dict->lookupTag(item[0]) << " ";
    std::cout << std::endl;
  }
  
  void push_tag(WordId t) {
   features_.push_back(Words(1, t));
  }

  size_t size() const override {
    return features_.size();
  }
 
  size_t tags_length() const {
    return features_.size();
  }

  WordId tag_at(WordIndex i) const {
    return features_.at(i)[0];
  }

  Words features_at(WordIndex i) const {
    return features_.at(i);
  }

 private:
  WordsList features_;

};

}

#endif
