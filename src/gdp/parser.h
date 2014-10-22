
#ifndef _GDP_PARSE_H_
#define _GDP_PARSE_H_

#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "corpus/parsed_sentence.h"


namespace oxlm {

class Parser: public ParsedSentence {
  public:
  Parser();
  
  Parser(Words tags);

  Parser(Words sent, Words tags);
  
  Parser(Words sent, Words tags, Indices arcs);

  Parser(Words sent, Words tags, Indices arcs, Words labels);

  //Parse(const Parse& parse);
  Parser(const TaggedSentence& parse);

  void push_arc() override {
    ParsedSentence::push_arc();
    left_children_.push_back(Indices()); 
    right_children_.push_back(Indices()); 
  }

  //may need a using statement
  void set_arc(WordIndex i, WordIndex j) override {
    ParsedSentence::set_arc(i, j);
      
    if ((j < 0) || (j >= size()))
      return;
    
    //TODO make sure insertions are at correct positions
    if (i < j) {
      bool inserted = false;
      //   WordIndex k = 0; (k < left_children_.at(j).size()) && !inserted; ++k) {
      for (auto p = left_children_.at(j).begin(); (p < left_children_.at(j).end()) && !inserted; ++p) {
        if (*p > i) {
          left_children_.at(j).insert(p, i);
          inserted = true;
        }
      } 
      if (!inserted)
        left_children_.at(j).push_back(i); 
    } else if (i > j) {
      bool inserted = false;
     // for (WordIndex k = 0; (k < right_children_.at(j).size()) && !inserted; ++k) {
     for (auto p = right_children_.at(j).begin(); (p < right_children_.at(j).end()) && !inserted; ++p) {
        if (*p > i) {
          right_children_.at(j).insert(p, i);
          inserted = true;
        }
      }
      if (!inserted)
       right_children_.at(j).push_back(i); 
    }
  }

  void set_weight(Real w) {
    weight_ = w;
  }

  void reset_weight() {
    weight_ = 0;
  }

  void add_weight(Real w) {
    weight_ += w;
  }

  //child i < head j
  //find child right of i
  WordIndex prev_left_child_at(WordIndex i, WordIndex j) const {
    for (unsigned k = 0; k < (left_children_.at(j).size() - 1); ++k) {
      if (left_children_[j][k] == i) {
        return left_children_[j][k+1];
      }
    }

    return 0;
  }

  //child i > head j
  //find child left of i
  WordIndex prev_right_child_at(WordIndex i, WordIndex j) const {
   for (unsigned k = (right_children_[j].size() - 1); k > 0; --k) {
      if (right_children_[j][k] == i) {
        return right_children_[j][k-1];
      }
    }

    return 0;
  }
  
  WordIndex leftmost_child_at(WordIndex j) const {
    if ((j >= size()) || (left_children_.at(j).empty())) 
      return -1;
    else
      return left_children_.at(j).front();
  }

  WordIndex rightmost_child_at(WordIndex j) const {
    if ((j >= size()) || (right_children_.at(j).empty())) 
      return -1;
    return right_children_.at(j).back();
  }

  WordIndex second_leftmost_child_at(WordIndex j) const {
    if ((j >= size()) || (left_children_.at(j).size() < 2)) 
      return -1;
    else
      return left_children_.at(j).at(1);
  }

  WordIndex second_rightmost_child_at(WordIndex j) const {
    if ((j >= size()) || (right_children_.at(j).size() < 2)) 
      return -1;
    return right_children_.at(j).rbegin()[1];
  }
  
  bool has_parent_at(WordIndex i) const {
    return (arc_at(i) >= 0);
  }
  
  bool have_children_at(WordIndex j) const {
    return (child_count_at(j) > 0);
  }

  size_t child_count_at(WordIndex j) const {
    return (left_children_.at(j).size() + right_children_.at(j).size());
  }

  bool has_equal_arcs(const ParsedSentence& parse) const {
    for (WordIndex j = 1; j < size(); ++j) {
      if (arc_at(j) != parse.arc_at(j))
        return false;
    }

    return true;
  }

  bool complete_parse() const {
    for (WordIndex i = 1; i < size(); ++i) 
      if (!has_parent_at(i) && ((i < (size() - 1)) || (tag_at(i) != 1)))
        return false;
    
    return true;
  }

  Real weight() const {
    return weight_;
  }

  private:
  IndicesList left_children_;
  IndicesList right_children_;
  Real weight_; 
};

}
#endif
