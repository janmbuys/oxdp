
#ifndef _GDP_PARSE_H_
#define _GDP_PARSE_H_

#include "corpus/dict.h"
#include "corpus/parsed_sentence.h"

namespace oxlm {

class Parse: public ParsedSentence {
  public:
  Parse();
  
  ParsedSentence(Words tags);

  ParsedSentence(Words sent, Words tags);
  
  ParsedSentence(Words sent, Words tags, Indices arcs);

  void push_arc() {
    arcs_.push_back(-1);
    left_children_.push_back(Indices()); 
    right_children_.push_back(Indices()); 
  }

  //overwriting parsed_sentence method.
  //may need a using statement
  virtual void set_arc(WordIndex i, WordIndex j) {
    ParsedSentence::set_arc(i, j);
      
    if ((j < 0) || (j >= size()))
      return;
    
    //TODO make sure insertions are at correct positions
    if (i < j) {
      bool inserted = false;
      for (WordIndex k = 0; (k < left_children_.at(j).size()) && !inserted; ++k) {
        if (left_children_[j][k] > i) {
          left_children_.at(j).insert(k, i);
          inserted = true;
        }
      } 
      if (!inserted)
        left_children_.at(j).push_back(i); 
    } else if (i > j) {
      bool inserted = false;
      for (WordIndex k = 0; (k < right_children_.at(j).size()) && !inserted; ++k) {
        if (left_children_[j][k] > i) {
          left_children_.at(j).insert(k, i);
          inserted = true;
        }
      } 
      if (!inserted)
       right_children_.at(j).push_back(i); 
    }
  }

  void set_weight(double w) {
    weight_ = w;
  }

  void reset_weight() {
    weight_ = 0;
  }

  void add_weight(double w) {
    weight_ += w;
  }

  //child i < head j
  //find child right of i
  WordIndex prev_left_child(WordIndex i, WordIndex j) const {
    for (unsigned k = 0; k < (left_children_.at(j).size() - 1); ++k) {
      if (left_children_[j][k] == i) {
        return left_children_[j][k+1];
      }
    }

    return 0;
  }

  //child i > head j
  //find child left of i
  WordIndex prev_right_child(WordIndex i, WordIndex j) const {
   for (unsigned k = (right_children_[j].size() - 1); k > 0; --k) {
      if (right_children_[j][k] == i) {
        return right_children_[j][k-1];
      }
    }

    return 0;
  }
  
  WordIndex leftmost_child(WordIndex j) const {
    if ((j >= size()) || (left_children_.at(j).empty())) 
      return -1;
    else
      return left_children_.at(j).front();
  }

  WordIndex rightmost_child(WordIndex j) const {
    if ((j >= size()) || (right_children_.at(j).empty())) 
      return -1;
    return right_children_.at(j).back();
  }

  WordIndex leftmost_next_child(WordIndex j) const {
    if ((j >= size()) || (left_children_.at(j).size() < 2)) 
      return -1;
    else
      return left_children_.at.at(j).at(1);
  }

  WordIndex rightmost_next_child(WordIndex j) const {
    if ((j >= size()) || (right_children_.at(j).size() < 2)) 
      return -1;
    return right_children_.at(j).rbegin()[1];
  }
  
  bool has_parent(WordIndex i) const {
    return (arc_at(i) >= 0);
  }
  
  bool have_children(WordIndex j) const {
    return (child_count_at(j) > 0);
  }

  size_t child_count_at(WordIndex j) const {
    return (left_children_.at(j).size() + right_children_.at(j).size());
  }

  bool complete_parse() const {
    for (WordIndex i = 1; i < arcs_.size() - 1; ++i) 
      if (!has_parent(i))
        return false;
    
    return true;
  }

  double weight() const {
    return weight_;
  }

  private:
  IndicesList left_children_;
  IndicesList right_children_;
  double weight_; 
};

}
#endif
