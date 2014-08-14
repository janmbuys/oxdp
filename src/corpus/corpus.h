#ifndef _PYPDICT_H_
#define _PYPDICT_H_

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

namespace oxlm {

typedef std::string Word;
typedef int WordId;
typedef int WordIndex;
typedef std::vector<WordId> Words;

class Dict {
// typedef std::unordered_map<std::string, WordId, std::hash<std::string> > Map;
// typedef std::map<std::string, WordId> Map;
public:
  //for language modeling
  Dict(): 
    b0_("<bad0>"), 
    sos_("<s>"), 
    eos_("</s>"), 
    bad0_id_(-1) 
  {
    words_.reserve(1000);
    convert(sos_, false);
    convert(eos_, false);
  }

  Dict(Word sos, Word eos): 
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
  
  //alternative initializer for parsing
  Dict(bool sos, bool eos): 
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

  void convertWhitespaceDelimitedLine(const std::string& line, std::vector<WordId>* out, bool frozen);

  void convertWhitespaceDelimitedConllLine(const std::string& line, std::vector<WordId>* sent_out, std::vector<WordId>* tag_out, std::vector<WordIndex>* dep_out, bool frozen);

  void readFromFile(const std::string& filename, std::vector<std::vector<WordId> >* src, std::set<WordId>* src_vocab, bool frozen);

  void readFromConllFile(const std::string& filename, std::vector<std::vector<WordId> >* sents, std::vector<std::vector<WordId> >* ptags, std::vector<std::vector<WordIndex> >* deps, bool frozen);

  WordId convert(const Word& word, bool frozen);

  WordId convertTag(const Word& tag, bool frozen);

  Word lookup(WordId id) const;

  Word lookupTag(WordId id) const;

  WordId min() const {
    return 0;
  }

  WordId max() const {
    return words_.size()-1;
  }

  size_t size() const {
    return words_.size();
  }

  size_t tag_size() const {
    return tags_.size();
  }

  static bool is_ws(char x) {
    return (x == ' ' || x == '\t');
  }

  std::vector<Word> get_vocab() const {
    return words_;
  }

  bool valid(const WordId id) const {
    return (id >= 0);
  }

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & b0_;
    ar & sos_;
    ar & eos_;
    ar & bad0_id_;
    ar & words_;
    ar & d_;
    ar & single_d_;
    ar & tags_;
    ar & tag_d_;
  }

private:
  Word b0_, sos_, eos_;
  WordId bad0_id_;
  std::vector<Word> words_;
  std::map<std::string, WordId> d_;
  std::map<std::string, WordId> single_d_;
  std::vector<Word> tags_;
  std::map<std::string, WordId> tag_d_;
};

template <typename Container>
struct container_hash {
  std::size_t operator()(Container const& c) const {
    return boost::hash_range(c.begin(), c.end());
  }
};

}

namespace std {
template<typename S, typename T> struct hash<pair<S, T>> {
  inline size_t operator()(const pair<S, T> & v) const {
    size_t seed = 0;
    boost::hash_combine(seed, v.first);
    boost::hash_combine(seed, v.second);
    return seed;
  }
};
}


#endif // PYPDICT_H_
