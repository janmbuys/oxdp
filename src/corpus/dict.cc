#include "corpus/dict.h"

namespace oxlm {

Dict::Dict(): 
  b0_("<unk>"), 
  sos_("<s>"), 
  eos_("</s>"), 
  bad0_id_(-1) 
{
  words_.reserve(1000);
  convert(sos_, false);
  convert(eos_, false);
}

Dict::Dict(Word sos, Word eos): 
  b0_("<unk>"), 
  sos_(sos), 
  eos_(eos), 
  bad0_id_(-1)
{
  words_.reserve(1000);
  convert(sos_, false);
  convertTag(sos_, false);
  if (eos!="") {
    convert(eos_, false);
    convertTag(eos_, false);
  }
}
 
//for parsing 
Dict::Dict(bool sos, bool eos): 
  b0_("<unk>"), 
  null_("<null>"), 
  sos_(""), 
  eos_(""), 
  bad0_id_(-1)
{
  words_.reserve(1000);
  //let null be 0
  convert(null_, false);
  convertTag(null_, false);
  convertLabel(null_, false);
  if (sos) {
    sos_ = "<root>";
    convert(sos_, false);
    convertTag(sos_, false);
  }
  if (eos) {
    eos_ = "<stop>";
    convert(eos_, false);
    convertTag(eos_, false);
  }
} 

WordId Dict::convert(const Word& word, bool frozen) {
  if (word == "")
    return bad0_id_;

  auto i = d_.find(word);
  if (i == d_.end()) {
    if (frozen) 
      return bad0_id_;
    words_.push_back(word);
    d_[word] = words_.size()-1;
    return words_.size()-1;
  } else {
    return i->second;
  }
}

WordId Dict::convertTag(const Word& tag, bool frozen) {
  auto i = tag_d_.find(tag);
  if (i == tag_d_.end()) {
    if (frozen) 
      return bad0_id_;
    tags_.push_back(tag);
    tag_d_[tag] = tags_.size()-1;
    return tags_.size()-1;
  } else {
    return i->second;
  }
}

WordId Dict::convertLabel(const Word& label, bool frozen) {
  auto i = label_d_.find(label);
  if (i == label_d_.end()) {
    if (frozen) 
      return bad0_id_;
    labels_.push_back(label);
    label_d_[label] = labels_.size()-1;
    return labels_.size()-1;
  } else {
    return i->second;
  }
}

bool Dict::punctTag(WordId id) const {
  std::vector<Word> punct = {".", ",", "?", "!", ":", "''", "``", 
                                   "(", ")", "-LRB-", "-RRB-", "#", "$"};
  Word word = lookupTag(id);
  for (auto punc: punct) {
    if (word == punc)
      return true;
  }
 
  return false;
}

Word Dict::lookup(WordId id) const {
  if (!valid(id)) 
    return b0_;
  return words_[id];
}

Word Dict::lookupTag(WordId id) const {
  if (!valid(id)) 
    return b0_;
  return tags_[id];
}
 
Word Dict::lookupLabel(WordId id) const {
  if (!valid(id)) 
    return b0_;
  return labels_[id];
}
 
}

