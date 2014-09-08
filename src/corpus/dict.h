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

#include <boost/shared_ptr.hpp>

#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/functional/hash.hpp>

#include "corpus/utils.h"

namespace oxlm {

//add a cc file later, but not yet
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
    //sos required
    convert(sos_, false);
    convert_tag(sos_, false);
    if (eos!="") {
      convert(eos_, false);
      convert_tag(eos_, false);
    }
  }
 
  //for parsing 
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
      convert_tag(sos_, false);
    }
    if (eos) {
      eos_ = "STOP";
      convert(eos_, false);
      convert_tag(eos_, false);
    }
  } 

  //TODO remove these methods

  void convert_whitespace_delimited_line(const std::string& line, std::vector<WordId>* out, bool frozen) {
    size_t cur = 0;
    size_t last = 0;
    int state = 0;
    out->clear();
    while (cur < line.size()) {
      if (is_ws(line[cur++])) {
        if (state == 0) 
          continue;
        out->push_back(convert(line.substr(last, cur - last - 1), frozen));
        state = 0;
      } else {
        if (state == 1) 
          continue;
        last = cur - 1;
        state = 1;
      }
    }

    if (state == 1)
      out->push_back(convert(line.substr(last, cur - last), frozen));
  }

  void convert_whitespace_delimited_conll_line(const std::string& line, std::vector<WordId>* sent_out, std::vector<WordId>* tag_out, std::vector<WordIndex>* dep_out, bool frozen) {
    size_t cur = 0;
    size_t last = 0;
    int state = 0;
    int col_num = 0;
   
    while(cur < line.size()) {
      if (is_ws(line[cur++])) {
        if (state == 0) 
          continue;
        if (col_num == 1) //1 - word
          sent_out->push_back(convert(line.substr(last, cur - last - 1), frozen));
        else if (col_num == 4) //4 - postag (3 - course postag)
          tag_out->push_back(convert_tag(line.substr(last, cur - last - 1), frozen));
        else if (col_num == 6)
          dep_out->push_back(static_cast<WordIndex>(stoi(line.substr(last, cur - last - 1))));
        ++col_num;
        state = 0;
      } else {
        if (state == 1) 
          continue;
        last = cur - 1;
        state = 1;
      }
    }

    if ((state == 1) && (col_num == 1)) //use relavent if we need last column
      sent_out->push_back(convert(line.substr(last, cur - last), frozen));
  }

  void read_from_file(const std::string& filename, boost::shared_ptr<std::vector<std::vector<WordId> > >& sents, bool frozen) {
    sents->clear();
    std::cerr << "Reading from " << filename << std::endl;
    std::ifstream in(filename);
    assert(in);
    std::string line;
    int lc = 0;
    while(getline(in, line)) {
      //sentence level
      ++lc;
      sents->push_back(std::vector<WordId>());
      //add start of sentence symbol
      sents->back().push_back(0); 

      convert_whitespace_delimited_line(line, &sents->back(), frozen);

      //add end of sentence symbol if defined 
      if (eos_!="") 
        sents->back().push_back(1); 
    }
  }

  void read_from_conll_file(const std::string& filename, boost::shared_ptr<std::vector<std::vector<WordId> > >& sents, boost::shared_ptr<std::vector<std::vector<WordId> > >& tags, boost::shared_ptr<std::vector<std::vector<WordIndex> > >& deps, bool frozen) {
    sents->clear();
    tags->clear();
    deps->clear();
    std::cerr << "Reading from " << filename << std::endl;
    std::ifstream in(filename);
    assert(in);
    std::string line;
    int lc = 0;
    int state = 1; //have to add new vector

    while(getline(in, line)) {
      //token level
      ++lc;

      if (line=="") { 
        state = 1;
        //add end of sentence symbol if defined (==1)
        if (eos_!="") {
          sents->back().push_back(1); 
          tags->back().push_back(1);
          deps->back().push_back(-1);
        }
      } else {
        if (state==1) {
          //add start of sentence symbol
          sents->push_back(std::vector<WordId>());
          sents->back().push_back(0); 
          tags->push_back(std::vector<WordId>());
          tags->back().push_back(0);
          deps->push_back(std::vector<WordIndex>());
          deps->back().push_back(-1);
          state = 0;
        }

        convert_whitespace_delimited_conll_line(line, &sents->back(), &tags->back(), &deps->back(), frozen); 
      }
    } 
  }

  WordId convert(const Word& word, bool frozen) {
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

  WordId convert_tag(const Word& tag, bool frozen) {
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

  Word lookup(WordId id) const {
    if (!valid(id)) 
      return b0_;
    return words_[id];
  }

  Word lookup_tag(WordId id) const {
    if (!valid(id)) 
      return b0_;
    return tags_[id];
  }
  
  WordId min() const {
    return 0;
  }

  WordId max() const {
    return words_.size()-1;
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

  std::vector<Word> get_vocab() const {
    return words_;
  }

  bool valid(const WordId id) const {
    return (id >= 0);
  }

  static bool is_ws(char x) {
    return (x == ' ' || x == '\t');
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
    ar & tags_;
    ar & tag_d_;
  }

private:
  Word b0_;
  Word sos_;
  Word eos_;
  WordId bad0_id_;
  std::vector<Word> words_;
  std::map<std::string, WordId> d_;
  std::vector<Word> tags_;
  std::map<std::string, WordId> tag_d_;
};

//TODO figure out if I really need this
/*
template <typename Container>
struct container_hash {
  std::size_t operator()(Container const& c) const {
    return boost::hash_range(c.begin(), c.end());
  }
}; */

}

/*
namespace std {
template<typename S, typename T> struct hash<pair<S, T>> {
  inline size_t operator()(const pair<S, T> & v) const {
    size_t seed = 0;
    boost::hash_combine(seed, v.first);
    boost::hash_combine(seed, v.second);
    return seed;
  }
};
} */

#endif // CORPUSDICT_H_

