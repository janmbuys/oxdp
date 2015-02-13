#ifndef _CORPUSDICT_H_
#define _CORPUSDICT_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>

#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/functional/hash.hpp>

#include "corpus/utils.h"

namespace oxlm {

class Dict {
public:
  Dict();

  Dict(Word sos, Word eos);

  Dict(bool root_first); 

  WordId convert(const Word& word, bool frozen);
 
  WordId convertTag(const Word& tag, bool frozen);

  WordId convertLabel(const Word& label, bool frozen);

  Word lookup(WordId id) const;

  Word lookupTag(WordId id) const;

  Word lookupLabel(WordId id) const;
 
  bool punctTag(WordId id) const;

  WordId min() const {
    return 0;
  }

  WordId max() const {
    return words_.size()-1;
  }

  WordId null() {
    return convert(null_, true);
  }

  WordId sos() {
    return convert(sos_, true);
  }

  WordId eos() {
    return convert(eos_, true);
  }

  size_t size() const {
    return words_.size();
  }

  size_t tag_size() const {
    return tags_.size();
  }

  size_t label_size() const {
    return labels_.size();
  }

  std::vector<Word> get_vocab() const {
    return words_;
  }

  bool valid(const WordId id) const {
    return (id >= 0);
  }

  bool root_first() const {
    return root_first_;
  }

  static bool is_ws(char x) {
    return (x == ' ' || x == '\t');
  }

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & b0_;
    ar & null_;
    ar & sos_;
    ar & eos_;
    ar & bad0_id_;
    ar & words_;
    ar & d_;
    ar & tags_;
    ar & tag_d_;
    ar & labels_;
    ar & label_d_;
  }

private:
  Word b0_;
  Word null_;
  Word sos_;
  Word eos_;
  bool root_first_;
  WordId bad0_id_;
  std::vector<Word> words_;
  std::map<std::string, WordId> d_;
  std::vector<Word> tags_;
  std::map<std::string, WordId> tag_d_;
  std::vector<Word> labels_;
  std::map<std::string, WordId> label_d_;
};

}

#endif // CORPUSDICT_H_

