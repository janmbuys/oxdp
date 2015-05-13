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
  convertFeature(null_, false);
  convertLabel(null_, false);

  //allways put root token in front
  sos_= "<s>";
  convert(sos_, false);
  convertFeature(sos_, false);
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
    word_to_features_.push_back(std::vector<WordId>()); 
    return words_.size()-1;
  } else {
    return i->second;
  }
}

WordId Dict::convertTag(const Word& tag, bool frozen) {
  auto i = tag_d_.find(tag);
  if (i == tag_d_.end()) {
    if (frozen) {
      //std::cerr << "OOV-tag:" << tag << " ";
      //return bad0_id_;
      return 0;
    }
    //std::cout << tag << " ";
    tags_.push_back(tag);
    tag_d_[tag] = tags_.size()-1;
    tag_to_feature_.push_back(convertFeature(tag, frozen));
    return tags_.size()-1;
  } else {
    return i->second;
  }
}

WordId Dict::convertFeature(const Word& feat, bool frozen) {
  auto i = feature_d_.find(feat);
  if (i == feature_d_.end()) {
    if (frozen) {
      //std::cerr << "OOV-feature:" << feat << " ";
      //return bad0_id_;
      return 0;
    }
    features_.push_back(feat);
    feature_d_[feat] = features_.size()-1;
    return features_.size()-1;
  } else {
    return i->second;
  }
}

WordId Dict::convertLabel(const Word& label, bool frozen) {
  auto i = label_d_.find(label);
  if (i == label_d_.end()) {
    if (frozen) {
      //std::cerr << "OOV-label:" << label << " ";
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

void Dict::setWordFeatures(WordId id, const std::vector<WordId>& features) {
  if (valid(id) && (word_to_features_[id].empty()))
    for (auto feat: features)
      word_to_features_[id].push_back(feat); 
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

std::vector<WordId> Dict::getWordFeatures(WordId id) const {
  if (!valid(id))
    return std::vector<WordId>();
  return word_to_features_[id];
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
 
Word Dict::lookupFeature(WordId id) const {
  if (!valid(id)) 
    return b0_;
  return features_[id];
}
 
Word Dict::lookupLabel(WordId id) const {
  if (!valid(id)) 
    return b0_;
  return labels_[id];
}
 
WordId Dict::tagToFeature(WordId tag_id) const {
  if (!valid(tag_id)) 
    return 0;
  return tag_to_feature_[tag_id];
}

}

