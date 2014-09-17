#include "corpus/dict.h"

namespace oxlm {

Dict::Dict(): 
  b0_("<bad0>"), 
  sos_("<s>"), 
  eos_("</s>"), 
  bad0_id_(-1) 
{
  words_.reserve(1000);
  convert(sos_, false);
  convert(eos_, false);
}

Dict::Dict(Word sos, Word eos): 
  b0_("<bad0>"), 
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
  b0_("<bad0>"), 
  sos_(""), 
  eos_(""), 
  bad0_id_(-1)
{
  words_.reserve(1000);
  if (sos) {
    sos_ = "ROOT";
    convert(sos_, false);
    convertTag(sos_, false);
  }
  if (eos) {
    eos_ = "STOP";
    convert(eos_, false);
    convertTag(eos_, false);
  }
} 

WordId Dict::convert(const Word& word, bool frozen) {
  if (word == "")
    return bad0_id_;

  //special treatment for sos_ (root)
  if (words_.size()==0 && !frozen) {
    words_.push_back(word);
    d_[word] = 0;
    return 0;
  } else if (word == sos_) {
    return 0;
  } else if (word == eos_ && !frozen) {
    words_.push_back(word);
    d_[word] = 1;
    return 1;
  } else if (word == eos_) {
    return 1;
  }

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
 
}

