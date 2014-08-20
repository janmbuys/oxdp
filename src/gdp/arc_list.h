#ifndef _GDP_AR_LIST_H_
#define _GDP_AR_LIST_H_

#include<string>
#include<functional>
#include<cstdlib>

namespace oxlm {

typedef std::vector<WordIndex> WxList;
typedef std::vector<Words> WordsList;

class ArcList {

public:
  ArcList(): 
    arcs_(),
    child_count_(),
    leftmost_child_(),
    rightmost_child_(),
    left_child_(),
    right_child_(),
    left_children_(),
    right_children_()
  {
  }

  ArcList(unsigned n): 
    arcs_(n, -1),
    child_count_(n, 0),
    leftmost_child_(n, -1),
    rightmost_child_(n, -1),
    left_child_(n, -1),  //last left child added
    right_child_(n, -1),  //first right child added
    left_children_(n, WxList()),  //all left children
    right_children_(n, WxList())  //all right children
  { 
  }

  void push_back() {
    arcs_.push_back(-1);
    child_count_.push_back(0);
    leftmost_child_.push_back(-1);
    rightmost_child_.push_back(-1);
    left_child_.push_back(-1); 
    right_child_.push_back(-1); 
  }

  void set_arc(WordIndex i, WordIndex j) {
    //node i has parent j
    arcs_[i] = j;
    //TODO change back
    
    if ((j >= 0) && (j < size())) {
      ++child_count_[j];
    } else {
      return;
    }

    /* if (j >= size())
      std::cerr << "add c ";
    else if (j >= 0)
    else
      std::cerr << "not in range"; */

    if (i < j) {
      //i left child of j
      if ((leftmost_child_[j] == -1) || (leftmost_child_[j] > i))  
        leftmost_child_[j] = i;
      else if ((left_child_[j] == -1) || (left_child_[j] > i))
        left_child_[j] = i;
    } else {
      //i right child of j
      if ((rightmost_child_[j] == -1) || (rightmost_child_[j] < i))
        rightmost_child_[j] = i;
      else if ((right_child_[j] == -1) || (right_child_[j] < i))
        right_child_[j] = i;
    } 
  }

  void set_arcs(const WxList& arcs) {
    for (int i = 0; i < size(); ++i) {
      set_arc(i, arcs[i]);
      //children in left to right order for each head
      if (i < arcs[i])
        left_children_[arcs[i]].push_back(i); 
      else if (arcs[i] >= 0)
        right_children_[arcs[i]].push_back(i); 
    }
  }

  void set_children() {
    for (int i = 0; i < size(); ++i) {
      //children in left to right order for each head
      if (i < arcs_[i])
        left_children_[arcs_[i]].push_back(i); 
      else if (arcs_[i] >= 0)
        right_children_[arcs_[i]].push_back(i); 
    }
  }

  void print_arcs() const {
    for (auto d: arcs_)
      std::cout << d << " ";
    std::cout << std::endl;   
  }

  WxList arcs() const {
    return arcs_;
  }

  WordIndex at(WordIndex i) const {
    return arcs_[i];
  }

  //child i < head j
  //child right of i
  WordIndex prev_left_child(WordIndex i, WordIndex j) const {
    for (unsigned k = 0; k < (left_children_[j].size() - 1); ++k) {
      if (left_children_[j][k] == i) {
        return left_children_[j][k+1];
      }
    }

    return 0;
  }

  //child i > head j
  //child left of i
  WordIndex prev_right_child(WordIndex i, WordIndex j) const {
   for (unsigned k = (right_children_[j].size() - 1); k > 0; --k) {
      if (right_children_[j][k] == i) {
        return right_children_[j][k-1];
      }
    }

    return 0;
  }
  
  WordIndex leftmost_child(WordIndex i) const {
    if (i >= size()) 
      return -1;
    return leftmost_child_[i];
  }

  WordIndex rightmost_child(WordIndex i) const {
    if (i >= size()) 
      return -1;
    return rightmost_child_[i];
  }

  WordIndex left_child(WordIndex i) const {
    if (i >= size()) 
      return -1;
    return left_child_[i];
  }

  WordIndex right_child(WordIndex i) const {
    if (i >= size()) 
      return -1;
    return right_child_[i];
  }
  
  bool has_parent(WordIndex i) const {
    return (arcs_[i] >= 0);
  }
  
  bool has_child(WordIndex i) const {
    return (child_count_[i] > 0);
  }

  int child_count_at(WordIndex i) const {
    return child_count_[i];
  }

  int size() const {
    return static_cast<int>(arcs_.size());
  }

  bool has_arc(WordIndex i, WordIndex j) const {
    return (arcs_[i] == j);
  }

  bool is_projective_dependency() const {
    for (int i = 0; i < (size() - 1); ++i)
      for (int j = i + 1; (j < size()); ++j)
        if ((arcs_[i]<i &&
              (arcs_[j]<i && arcs_[j]>arcs_[i])) ||
            ((arcs_[i]>i && arcs_[i]>j) &&
              (arcs_[j]<i || arcs_[j]>arcs_[i])) ||
            ((arcs_[i]>i && arcs_[i]<j) &&
              (arcs_[j]>i && arcs_[j]<arcs_[i])))
          return false;
    return true;
  }

  bool operator==(const ArcList& a) const {
    return (arcs_==a.arcs());
  }

private:
  WxList arcs_;
  std::vector<int> child_count_;
  WxList leftmost_child_;
  WxList rightmost_child_;
  WxList left_child_;
  WxList right_child_;
  std::vector<WxList> left_children_;
  std::vector<WxList> right_children_;
};

}
#endif
