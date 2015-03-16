#include "corpus/dict.h"

namespace oxlm {

//for language modeling
Dict::Dict(): 
  b0_("<unk>"), 
  sos_("<s>"), 
  eos_("</s>"), 
  root_first_(false),
  bad0_id_(-1) 
{
  words_.reserve(1000);
  convert(sos_, false);
  convert(eos_, false);
}

//used eg for char lm
Dict::Dict(Word sos, Word eos): 
  b0_("<unk>"), 
  sos_(sos), 
  eos_(eos), 
  root_first_(true),
  bad0_id_(-1)
{
  words_.reserve(1000);
  convert(sos_, false);
  convertTag(sos_, false);
  if (eos!="") {
    root_first_ = false;
    convert(eos_, false);
    convertTag(eos_, false);
  }
}  
 
//for parsing 
Dict::Dict(bool root_first): 
  b0_("<unk>"), //not in the vocabulary
  null_("<null>"), 
  sos_(""), 
  eos_(""), 
  root_first_(root_first),
  bad0_id_(-1)
{
  words_.reserve(1000);
  convert(null_, false); //let null map to 0
  convertTag(null_, false);
  convertLabel(null_, false);

  //allways put root token in front
  sos_= "<s>";
  convert(sos_, false);
  convertTag(sos_, false);
  
  //  eos_= "</s>";
  //  convert(eos_, false);
  //  convertTag(eos_, false);
} 

WordId Dict::convert(const Word& word, bool frozen) {
  if (word == "")
    return bad0_id_;

  //assume unknown words have been removed by preprocessing
  //map remaining exceptions to null
  auto i = d_.find(word);
  if (i == d_.end()) {
    if (frozen) {
      //std::cerr << "OOV:" << word << " ";
      //return bad0_id_;
      return 0;
    }
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
    if (frozen) {
      //std::cerr << "OOV:" << tag << " ";
      //return bad0_id_;
      return 0;
    }
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
    if (frozen) {
      //std::cerr << "OOV:" << label << " ";
      //return bad0_id_;
      return 0;
    }
    labels_.push_back(label);
    label_d_[label] = labels_.size()-1;
    return labels_.size()-1;
  } else {
    return i->second;
  }
}

bool Dict::punctTag(WordId id) const {
  std::vector<Word> punct = {".", ",", "?", "!", ":", "''", "``", 
                                   "(", ")", "-LRB-", "-RRB-", "#", "$", "PU"};
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

