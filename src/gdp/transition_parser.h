#ifndef _GDP_TR_PARSER_H_
#define _GDP_TR_PARSER_H_

#include<string>
#include<functional>
#include<cstdlib>

#include<corpus/corpus.h>

namespace oxlm {

enum class kAction : WordId {sh, la, ra, re, la2, ra2};

typedef std::vector<WordIndex> WxList;
typedef std::vector<Words> WordsList;
typedef std::vector<kAction> ActList;

class ArcList {

public:
  ArcList(): 
    arcs_(),
    child_count_(),
    leftmost_child_(),
    rightmost_child_()
  {
  }

  ArcList(unsigned n): 
    arcs_(n, -1),
    child_count_(n, 0),
    leftmost_child_(n, -1),
    rightmost_child_(n, -1)
  { 
  }

  void push_back() {
    arcs_.push_back(-1);
    child_count_.push_back(0);
  }

  void set_arc(WordIndex i, WordIndex j) {
    //node i has parent j
    arcs_[i] = j;
    if (j >= 0)
      ++child_count_[j];
    if (i < j) {
      //i left child of j
      if ((leftmost_child_[j] == -1) || (leftmost_child_[j] > i))  
        leftmost_child_[j] = i;
    } else {
      //i right child of j
      if ((rightmost_child_[j] == -1) || (rightmost_child_[j] < i))
        rightmost_child_[j] = i;
    }
  }

  void set_arcs(const WxList& arcs) {
    for (int i = 0; i < static_cast<int>(size()); ++i) {
      set_arc(i, arcs[i]);
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

  WordIndex leftmost_child(WordIndex i) const {
    return leftmost_child_[i];
  }

  WordIndex rightmost_child(WordIndex i) const {
    return rightmost_child_[i];
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

  unsigned size() const {
    return arcs_.size();
  }

  bool has_arc(WordIndex i, WordIndex j) const {
    return (arcs_[i] == j);
  }

  bool is_projective_dependency() const {
    for (int i = 0; i < static_cast<int>(arcs_.size() - 1); ++i)
      for (int j = i + 1; j < static_cast<int>(arcs_.size()); ++j)
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
};

class TransitionParser {
  public:

  TransitionParser():  
    stack_(),
    buffer_(),
    arcs_(),
    actions_(),
    sentence_(),
    tags_(),
    liw_{0},
    lpw_{0},
    num_particles_{1}
  {
  }

  TransitionParser(Words sent):  
    stack_(),
    buffer_(sent.size(), 0),
    arcs_(sent.size()),
    actions_(),
    sentence_(sent),
    tags_(),
    liw_{0},
    lpw_{0},
    num_particles_{1}
  {
    //sentence already includes root (0)
    for (int i = 0; i < (int)sentence_.size(); ++i)
      buffer_[i] = sentence_.size() - i - 1; 
  }

  TransitionParser(Words sent, Words tags):  
    stack_(),
    buffer_(sent.size(), 0),
    arcs_(sent.size()),
    actions_(),
    sentence_(sent),
    tags_(tags),
    liw_{0},
    lpw_{0},
    num_particles_{1}
  {
    //sentence already includes root (0)
    for (int i = 0; i < (int)sentence_.size(); ++i)
      buffer_[i] = sentence_.size() - i - 1; 
  }

  //only use when generating 
  bool buffer_tag(WordId t) {
    buffer_.push_back(tags_.size());
    tags_.push_back(t);
    return true;
  }

  //virtual bool leftArc() = 0;
  //virtual bool rightArc() = 0;
  //virtual kAction oracleNext(const ArcList& gold_arcs) const = 0;
  //virtual bool execute_action(kAction a) = 0;
 
  void pop_buffer() {
    buffer_.pop_back();
  }

  void pop_stack() {
    stack_.pop_back();
  }

  void push_stack(WordIndex i) {
    stack_.push_back(i);
  }

  void push_arc() {
    arcs_.push_back();
  }

  void push_word(WordId w) {
   sentence_.push_back(w);
  }

  void append_action(kAction a) {
    actions_.push_back(a);
  }

  void set_arc(WordIndex i, WordIndex j) {
    arcs_.set_arc(i, j);
  }

  void reset_importance_weight() {
    liw_ = 0;
  }

  void set_importance_weight(double w) {
    liw_ = -std::log(w);
  }

  void add_importance_weight(double w) {
    liw_ -= std::log(w);
  }

  void set_log_particle_weight(double w) {
    lpw_ = w;
  }

  void set_particle_weight(double w) {
    lpw_ = -std::log(w);
  }

  void add_particle_weight(double w) {
    lpw_ -= std::log(w);
  }

  void set_num_particles(int n) {
    num_particles_ = n;
  }

  std::string actions_str() const {
    const std::vector<std::string> action_names {"sh", "la", "ra", "re", "la2", "ra2"};
    std::string seq_str = "";
    for (kAction a: actions_)
      seq_str += action_names[static_cast<int>(a)] + " ";
    return seq_str; 
  }
  
  void print_arcs() const {
    for (auto a: arcs_.arcs())
      std::cout << a << " ";
    //std::cout << std::endl;
  }

  void print_sentence(Dict& dict) const {
    for (auto a: sentence_)
      std::cout << dict.lookup(a) << " ";
    std::cout << std::endl;
  }

  void print_tags(Dict& dict) const {
    for (auto a: tags_)
      std::cout << dict.lookupTag(a) << " ";
    std::cout << std::endl;
  }

  unsigned sentence_length() const {
    return sentence_.size();
  }

  Words sentence() const {
    return sentence_;
  }

  Words tags() const {
    return tags_;
  }
  
  WordId tag_at(WordIndex i) const {
    return tags_.at(i);
  }

  ArcList arcs() const {
    return arcs_;
  }

  double importance_weight() const {
    return liw_;
  }

  double particle_weight() const {
    return lpw_;
  }

  int num_particles() const {
    return num_particles_;
  }

  double weighted_importance_weight() const {
    return (liw_ - std::log(num_particles_));
  }

  double weighted_particle_weight() const {
    return (lpw_- std::log(num_particles_));
  }

  ActList actions() const {
    return actions_;
  }
   
  kAction last_action() const {
    return actions_.back();
  }

  unsigned num_actions() const {
    return actions_.size();
  }

  //number of children at sentence position i
  int child_count_at(int i) const {
    return arcs_.child_count_at(i);
  }

  bool has_parent(int i) const {
    return arcs_.has_parent(i);
  }

  int stack_depth() const {
    return stack_.size();
  }

  bool is_stack_empty() const {
    return stack_.empty();
  }

  int buffer_length() const {
    return buffer_.size();
  }

  WordIndex stack_top() const {
    return stack_.back();
  }

  WordIndex stack_top_second() const {
    return stack_.rbegin()[1];
  }

  WordId next_word() const {
    return sentence_[buffer_.back()];
  }

  WordId next_tag() const {
    return tags_[buffer_.back()];
  }

  WordIndex buffer_next() const {
    return buffer_.back();
  }

  bool buffer_next_has_child() const {
    return arcs_.has_child(buffer_.back());
  }

  bool is_buffer_empty() const {
    return buffer_.empty();
  }

  bool is_terminal_configuration() const {
    return (buffer_.empty() && (stack_.size() == 1));
  }

  unsigned directed_accuracy_count(ArcList g_arcs) const {
    unsigned count = 0;
    for (unsigned j = 1; j < arcs_.size(); ++j) {
      if (arcs_.at(j)==g_arcs.at(j))
        ++count;
    }
    return count;
  }

  unsigned undirected_accuracy_count(ArcList g_arcs) const {
    unsigned count = 0;
    for (unsigned j = 1; j < arcs_.size(); ++j) {
      if ((arcs_.at(j)==g_arcs.at(j)) || (arcs_.has_parent(j) && g_arcs.at(arcs_.at(j))==static_cast<int>(j)))
        ++count;
    }
    return count;
  }

  //****functions for context vectors: each function defined for a specific context length

  Words tag_raw_context() const {
    /*Words ctx(4, 0);

    if (stack_.size() >= 1) { 
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (stack_.size() >= 3) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-4));
    } */ 

    Words ctx(3, 0);
    
    if (stack_.size() >= 1) { 
      ctx[2] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (stack_.size() >= 3) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-3));
    } 

    /* Words ctx(2, 0);
    
    if (stack_.size() >= 1) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-2));
    } */

    return ctx;
  }

  Words tag_augmented_plus_context() const {
    Words ctx(6, 0);
    //add word distance feature

    if (stack_.size() >= 1) { 
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-2));
      WordIndex i = stack_.rbegin()[1];
      WordIndex j = stack_.rbegin()[0];
      ctx[4] = j - i;
    }
    if (stack_.size() >= 3) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-4));
    }
    
    ctx[5] = static_cast<int>(stack_.size());
    return ctx;
  }

  Words one_tag_context() const {
    Words ctx(1, 0);
    //add word distance feature

    if (stack_.size() >= 1) { 
      ctx[0] = tags_.at(stack_.at(stack_.size()-1));
    }

    return ctx;
  }

  Words one_tag_distance_context() const {
    Words ctx(2, 0);
    //add word distance feature

    if (stack_.size() >= 1) { 
      //ctx[0] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-2));
      WordIndex i = stack_.rbegin()[1];
      WordIndex j = stack_.rbegin()[0];
      ctx[1] = j - i;
    }

    return ctx;
  }

  Words next_distance_context() const {
    Words ctx(1, 0);
    //add word distance feature

    if (stack_.size() >= 1) { 
      if (!is_buffer_empty()) {
        WordIndex i = stack_top();
        WordIndex j = buffer_next();
        ctx[0] = j - i;
      }
    }
    
    return ctx;
  }

  Words one_tag_next_distance_context() const {
    Words ctx(2, 0);
    //add word distance feature

    if (stack_.size() >= 1) { 
      ctx[0] = tags_.at(stack_.at(stack_.size()-1));
      if (!is_buffer_empty()) {
        WordIndex i = stack_top();
        WordIndex j = buffer_next();
        ctx[1] = j - i;
      }
    }
    
    return ctx;
  }
  
  Words tag_next_distance_context() const {
    Words ctx(3, 0);
    //add word distance feature

    if (stack_.size() >= 1) { 
      ctx[0] = tags_.at(stack_.at(stack_.size()-1));
      if (!is_buffer_empty()) {
        WordIndex i = stack_top();
        WordIndex j = buffer_next();
        ctx[2] = j - i;
      }
    }
    if (stack_.size() >= 2) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-2));
    }

    return ctx;
  } 

  Words tag_more_distance_context() const {
    Words ctx(4, 0);
    //add word distance feature

    if (stack_.size() >= 1) { 
      ctx[2] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-2));
      WordIndex i = stack_.rbegin()[1];
      WordIndex j = stack_.rbegin()[0];
      ctx[3] = j - i;
    }
    if (stack_.size() >= 3) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-3));
    }

    return ctx;
  }

  Words tag_children_distance_context() const {
    Words ctx(9, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[8] = tags_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[4] = tags_.at(l1); //
      if (r1 > 0)
        ctx[6] = tags_.at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[7] = tags_.at(stack_.at(stack_.size()-2));
      if (l2 >= 0)
        ctx[3] = tags_.at(l2);
      if (r2 >= 0)
        ctx[5] = tags_.at(r2); //

      WordIndex i = stack_.rbegin()[1];
      WordIndex j = stack_.rbegin()[0];
      ctx[0] = j - i;
    }
    if (stack_.size() >= 3) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-4));
    }
    return ctx;
  }

  Words tag_children_context() const {
    Words ctx(8, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[7] = tags_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[3] = tags_.at(l1); //
      if (r1 > 0)
        ctx[5] = tags_.at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[6] = tags_.at(stack_.at(stack_.size()-2));
      if (l2 >= 0)
        ctx[2] = tags_.at(l2);
      if (r2 >= 0)
        ctx[4] = tags_.at(r2); //
    }
    if (stack_.size() >= 3) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-4));
    }
    return ctx;
  }

  Words tag_less_children_context() const {
    Words ctx(6, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      //WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[5] = tags_.at(stack_.at(stack_.size()-1));
      //if (l1 > 0)
      //  ctx[1] = tags_.at(l1); //
      if (r1 > 0)
        ctx[3] = tags_.at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      //WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[4] = tags_.at(stack_.at(stack_.size()-2));
      //if (l2 >= 0)
      //  ctx[0] = tags_.at(l2);
      if (r2 >= 0)
        ctx[2] = tags_.at(r2); //
    }
    if (stack_.size() >= 3) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-4));
    }

    return ctx;
  }

  Words tag_some_children_context() const {
    Words ctx(4, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      //WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
      if (r1 > 0)
        ctx[1] = tags_.at(r1);  //[0]
      //if (l1 > 0)
      //  ctx[1] = tags_.at(l1);
    }
    if (stack_.size() >= 2) {
      WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      //WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[2] = tags_.at(stack_.at(stack_.size()-2));
      if (r2 >= 0)
        ctx[0] = tags_.at(r2);
      //if (l2 >= 0)
      //  ctx[0] = tags_.at(l2);  //[1]
    }

    return ctx;
  }

  Words tag_distance_context() const {
    Words ctx(3, 0);
    //add word distance feature

    if (stack_.size() >= 1) { 
      ctx[0] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-2));
      WordIndex i = stack_.rbegin()[1];
      WordIndex j = stack_.rbegin()[0];
      ctx[2] = j - i;
    }

    return ctx;
  }

  Words tag_more_context() const {
    /*Words ctx(5, 0);
    
    if (stack_.size() >= 1) { 
      ctx[4] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[3] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (stack_.size() >= 3) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-4));
    }
    if (stack_.size() >= 5) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-5));
    } */
    
    Words ctx(4, 0);

    if (stack_.size() >= 1) { 
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (stack_.size() >= 3) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-4));
    }
    return ctx;
  }

  Words tag_less_context() const {
    /*Words ctx(3, 0);
    
    if (stack_.size() >= 1) { 
      ctx[2] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (stack_.size() >= 3) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-3));
    } */

    Words ctx(2, 0);
    
    if (stack_.size() >= 1) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-2));
    } 

    return ctx;
  }

  Words word_context() const {
    Words ctx(2, 0);
    
    if (stack_.size() >= 1) { 
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
    }

    return ctx;
  }
  
  Words word_tag_more_context() const {
    Words ctx(6, 0);
    
    //word context 2 and pos context 4
    if (stack_.size() >= 1) { 
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
      ctx[5] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
      ctx[4] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (stack_.size() >= 3) {
      ctx[3] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-4));
    }

    return ctx;
  }
  
  Words word_tag_context() const {
    Words ctx(4, 0);
    
    //word and pos context of 2
    if (stack_.size() >= 1) { 
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
      ctx[2] = tags_.at(stack_.at(stack_.size()-2));
    }

    return ctx;
  }

  Words word_tag_less_context() const {
    Words ctx(3, 0);

    //word context 1 and pos context 2
    if (stack_.size() >= 1) { 
      ctx[0] = sentence_.at(stack_.at(stack_.size()-1));
      ctx[2] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-2));
    }

    return ctx;
  }

  Words word_next_context() const {
    Words ctx(3, 0);
    
    //word context 2, pos context 2 + next token
    if (stack_.size() >= 1) { 
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
    }
    if (buffer_.size() > 0)
      ctx[2] = tags_.at(buffer_.back());

    return ctx;
  }

  Words tag_next_context() const {
    Words ctx(3, 0);
    
    //word context 2, pos context 2 + next token
    if (stack_.size() >= 1) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[0] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (buffer_.size() > 0)
      ctx[2] = tags_.at(buffer_.back());

    return ctx;
  }

  Words word_tag_next_context() const {
    Words ctx(5, 0);
    
    //word context 2, pos context 2 + next token
    if (stack_.size() >= 1) { 
      ctx[2] = sentence_.at(stack_.at(stack_.size()-1));
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
      ctx[1] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (buffer_.size() > 0)
      ctx[4] = tags_.at(buffer_.back());

    return ctx;
  }

  private:
  WxList stack_;
  WxList buffer_;
  ArcList arcs_;
  ActList actions_;
  Words sentence_;
  Words tags_;
  double liw_; //log importance weight
  double lpw_; //log particle weight
  int num_particles_;
};

class ArcStandardParser : public TransitionParser {
  public:

  ArcStandardParser():
    TransitionParser() {
  }

  ArcStandardParser(Words sent):
    TransitionParser(sent) {
  }

  ArcStandardParser(Words sent, Words ptags):
    TransitionParser(sent, ptags) {
  }

  bool shift();

  bool shift(WordId w);

  bool leftArc();

  bool rightArc();
  
  bool left_arc_valid() const {
    if (stack_depth() < 2)
      return false;
    WordIndex i = stack_top_second();
    return (i != 0);
  }

  bool execute_action(kAction a) {
    switch(a) {
    case kAction::sh:
      return shift();
    case kAction::la:
      return leftArc();
    case kAction::ra:
      return rightArc();
    default: 
      std::cerr << "action not implemented" << std::endl;
      return false;
    }
  }

  //**functions that call the context vector functions for a given configuration
  //(ideally would assert length of order)
  Words shift_context() const {
    return word_next_context();
  }

  Words reduce_context() const {
    return tag_children_context();
    //return tag_children_context();
    //return tag_less_context();
  }

  Words arc_context() const {
    return tag_less_context();
  }

  Words tag_context() const {
    return tag_children_context();
    //return tag_less_children_context();
    //return tag_more_context();
  }

  kAction oracleNext(const ArcList& gold_arcs) const;
  
  kAction oracleDynamicNext(const ArcList& gold_arcs) const;
  
  private:
  WxList bufffer_left_children;
  WxList bufffer_right_children;
};

class ArcEagerParser : public TransitionParser {
  public:

  ArcEagerParser(): 
    TransitionParser(),
    buffer_left_most_child_{-1},
    buffer_left_child_{-1} 
  {
  }

  ArcEagerParser(Words sent): 
    TransitionParser(sent),
    buffer_left_most_child_{-1},
    buffer_left_child_{-1}
  {
  }

  ArcEagerParser(Words sent, Words ptags): 
    TransitionParser(sent, ptags),
    buffer_left_most_child_{-1},
    buffer_left_child_{-1}
  {
  }

  bool shift();

  bool leftArc();

  bool rightArc();
  
  bool reduce();

  bool left_arc_valid() const {
    //stack_size 1 -> stack top is root
    if (stack_depth() < 2)
      return false;    
    WordIndex i = stack_top();
    return (!has_parent(i));
  }

  bool reduce_valid() const {
    WordIndex i = stack_top();
    return has_parent(i);
  }

  bool execute_action(kAction a) {
    switch(a) {
    case kAction::sh:
      return shift();
    case kAction::la:
      return leftArc();
    case kAction::ra:
      return rightArc();
    case kAction::re:
      return reduce();
    default: 
      std::cerr << "action not implemented" << std::endl;
      return false;
    }
  }

  //**functions that call the context vector functions for a given configuration
  //(ideally would assert length of order)
  Words shift_context() const {
    return word_next_context();
  }

  Words reduce_context() const {
    return tag_next_distance_context();
  }

  Words arc_context() const {
    return tag_less_context();
  }
  
  Words tag_context(kAction a) const {
    Words ctx = tag_less_context();

    //another back-off level...
    if (buffer_next_has_child()) {
      if (buffer_left_most_child_ > -1)
        ctx.push_back(tag_at(buffer_left_most_child_));
      else 
       ctx.push_back(0);
      ctx.push_back(tag_at(buffer_left_child_));
      ctx.push_back(1);
    }
    else {
      //ctx.push_back(static_cast<WordId>(a));  //no real difference, maybe slightly worse
      ctx.push_back(0); 
      ctx.push_back(0); 
      ctx.push_back(0);
    }

    return ctx;
  }

  kAction oracleNext(const ArcList& gold_arcs) const;
  
  private:
  WordIndex buffer_left_most_child_;
  WordIndex buffer_left_child_;

};

class AccuracyCounts {

public:
  AccuracyCounts(): 
    likelihood_{0},
    gold_likelihood_{0},
    reduce_count_{0},
    reduce_gold_{0},
    shift_count_{0},
    shift_gold_{0},
    final_reduce_error_count_{0},
    total_length_{0},
    directed_count_{0},
    undirected_count_{0}, 
    root_count_{0},
    gold_more_likely_count_{0},
    num_actions_{0},
    complete_sentences_{0},
    num_sentences_{0}
  {
  }

  void inc_reduce_count() {
    ++reduce_count_;
  }

  void inc_reduce_gold() {
    ++reduce_gold_;
  }

  void inc_shift_count() {
    ++shift_count_;
  }

  void inc_shift_gold() {
    ++shift_gold_;
  }

  void inc_final_reduce_error_count() {
    ++final_reduce_error_count_;
  }

  void inc_complete_sentences() {
    ++complete_sentences_;
  }

  void inc_gold_more_likely_count() {
    ++gold_more_likely_count_;
  }

  void inc_root_count() {
    ++root_count_;
  }

  void inc_num_sentences() {
    ++num_sentences_;
  }

  void add_likelihood(double l) {
    likelihood_ += l;
  }

  void add_gold_likelihood(double l) {
    gold_likelihood_ += l;
  }

  void add_total_length(int l) {
    total_length_ += l; 
  }

  void add_num_actions(int l) {
    num_actions_ += l; 
  }

  void add_directed_count(int l) {
    directed_count_ += l; 
  }

  void add_undirected_count(int l) {
    undirected_count_ += l; 
  }
   
  void countAccuracy(const ArcStandardParser& prop_parse, const ArcStandardParser& gold_parse); 
  //void countAccuracy(const ArcStandardParser& prop_parse, const ArcList& gold_arcs); 

  void countAccuracy(const ArcEagerParser& prop_parse, const ArcEagerParser& gold_parse); 
  //void countAccuracy(const ArcEagerParser& prop_parse, const ArcList& gold_arcs); 

  double directed_accuracy() const {
    return (directed_count_ + 0.0)/total_length_;
  }

  double undirected_accuracy() const {
    return (undirected_count_ + 0.0)/total_length_;
  }

  double complete_accuracy() const {
    return (complete_sentences_ + 0.0)/num_sentences_;
  }

  double root_accuracy() const {
    return (root_count_ + 0.0)/num_sentences_;
  }

  double gold_more_likely() const {
    return (gold_more_likely_count_ + 0.0)/num_sentences_;
  }

  double arc_dir_precision() const {
    return (directed_count_ + 0.0)/undirected_count_;
  }

  double reduce_recall() const {
    return (reduce_count_ + 0.0)/reduce_gold_;
  }

  double shift_recall() const {
    return (shift_count_ + 0.0)/shift_gold_;
  }

  double likelihood() const {
    return likelihood_;
  }

  double gold_likelihood() const {
    return gold_likelihood_;
  }

  double final_reduce_error_rate() const {
    return (final_reduce_error_count_ + 0.0)/total_length_;
  }

  double cross_entropy() const {
    return likelihood_/(std::log(2)*num_actions_);
  }

  double gold_cross_entropy() const {
    return gold_likelihood_/(std::log(2)*num_actions_);
  }

  double perplexity() const {
    return std::pow(2, cross_entropy());
  }

  double gold_perplexity() const {
    return std::pow(2, gold_cross_entropy());
  }

private:
    double likelihood_;  
    double gold_likelihood_;  
    int reduce_count_; 
    int reduce_gold_;
    int shift_count_;
    int shift_gold_;
    int final_reduce_error_count_;
    int total_length_;
    int directed_count_;
    int undirected_count_;
    int root_count_;
    int gold_more_likely_count_;
    int num_actions_;
    int complete_sentences_;
    int num_sentences_;
};

inline bool cmp_particle_ptr_weights_as(const std::unique_ptr<ArcStandardParser>& p1, 
                                     const std::unique_ptr<ArcStandardParser>& p2) {
  //null should be the biggest
  if (p1 == nullptr)
    return false;
  else if (p2 == nullptr)
    return true;
  else
    return (p1->particle_weight() < p2->particle_weight());
}

inline bool cmp_importance_ptr_weights_as(const std::unique_ptr<ArcStandardParser>& p1, 
                                     const std::unique_ptr<ArcStandardParser>& p2) {
  //null should be the biggest
  if (p1 == nullptr)
    return false;
  else if (p2 == nullptr)
    return true;
  else
    return (p1->importance_weight() < p2->importance_weight());
}

inline bool cmp_weighted_particle_ptr_weights_as(const std::unique_ptr<ArcStandardParser>& p1, 
                                              const std::unique_ptr<ArcStandardParser>& p2) {
  //null or no particles should be the biggest
  if ((p1 == nullptr) || (p1->num_particles() == 0))
    return false;
  else if ((p2 == nullptr) || (p2->num_particles() == 0))
    return true;
  else
    return (p1->weighted_particle_weight() < p2->weighted_particle_weight());
}

inline bool cmp_weighted_importance_ptr_weights_as(const std::unique_ptr<ArcStandardParser>& p1, 
                                                const std::unique_ptr<ArcStandardParser>& p2) {
  //null or no particles should be the biggest
  if ((p1 == nullptr) || (p1->num_particles() == 0))
    return false;
  else if ((p2 == nullptr) || (p2->num_particles() == 0))
    return true;
  else
    return (p1->weighted_importance_weight() < p2->weighted_importance_weight());
}

inline bool cmp_reduce_particle_ptr_weights_ae(const std::unique_ptr<ArcEagerParser>& p1, 
                                     const std::unique_ptr<ArcEagerParser>& p2) {
  //null should be the biggest
  if (p1 == nullptr)
    return false;
  else if (p2 == nullptr)
    return true;
  //then those that cannot reduce
  else if (!p1->reduce_valid())
    return false;
  else if (!p2->reduce_valid())
    return true;
    return (p1->particle_weight() < p2->particle_weight());
}

inline bool cmp_particle_ptr_weights_ae(const std::unique_ptr<ArcEagerParser>& p1, 
                                     const std::unique_ptr<ArcEagerParser>& p2) {
  //null should be the biggest
  if (p1 == nullptr)
    return false;
  else if (p2 == nullptr)
    return true;
  else
    return (p1->particle_weight() < p2->particle_weight());
}

inline bool cmp_importance_ptr_weights_ae(const std::unique_ptr<ArcEagerParser>& p1, 
                                     const std::unique_ptr<ArcEagerParser>& p2) {
  //null should be the biggest
  if (p1 == nullptr)
    return false;
  else if (p2 == nullptr)
    return true;
  else
    return (p1->importance_weight() < p2->importance_weight());
}

inline bool cmp_weighted_particle_ptr_weights_ae(const std::unique_ptr<ArcEagerParser>& p1, 
                                              const std::unique_ptr<ArcEagerParser>& p2) {
  //null or no particles should be the biggest
  if ((p1 == nullptr) || (p1->num_particles() == 0))
    return false;
  else if ((p2 == nullptr) || (p2->num_particles() == 0))
    return true;
  else
    return (p1->weighted_particle_weight() < p2->weighted_particle_weight());
}

inline bool cmp_weighted_importance_ptr_weights_ae(const std::unique_ptr<ArcEagerParser>& p1, 
                                                const std::unique_ptr<ArcEagerParser>& p2) {
  //null or no particles should be the biggest
  if ((p1 == nullptr) || (p1->num_particles() == 0))
    return false;
  else if ((p2 == nullptr) || (p2->num_particles() == 0))
    return true;
  else
    return (p1->weighted_importance_weight() < p2->weighted_importance_weight());
}

inline kAction convert_to_action(WordId a) {
  std::vector<kAction> actList = {kAction::sh, kAction::la, kAction::ra, kAction::re, kAction::la2, kAction::ra2};
  return actList[a];
}

}
#endif
