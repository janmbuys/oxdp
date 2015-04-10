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

  void set_tag_at(WordIndex i, WordId t) {
    features_.at(i)[0] = t;
  }

  size_t size() const override {
    return features_.size();
  }
 
  size_t tags_length() const {
    return features_.size();
  }

  WordId tag_at(WordIndex i) const {
    if (features_.at(i).size() > 0)
      return features_.at(i)[0];
    else
      return 0;
  }

  Words features_at(WordIndex i) const {
    return features_.at(i);
  }

 private:
  WordsList features_;

};

}

#endif
