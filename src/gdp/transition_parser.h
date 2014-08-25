#ifndef _GDP_TR_PARSER_H_
#define _GDP_TR_PARSER_H_

#include<string>
#include<functional>
#include<cstdlib>

#include "corpus/corpus.h"
#include "utils/random.h"
#include "arc_list.h"
#include "eisner_parser.h"

namespace oxlm {

enum class kAction : WordId {sh, la, ra, re, la2, ra2};

typedef std::vector<kAction> ActList;

class TransitionParser {
  public:

  //use specifically when generating from model
  TransitionParser():  
    stack_(),
    buffer_next_(),
    arcs_(),
    actions_(),
    sentence_(),
    tags_(),
    liw_{0},
    lpw_{0},
    beam_lpw_{0},
    num_particles_{1},
    is_generating_{true}
  {
    arcs_.push_back(); //add arc for next word before it has been generated
  }

  TransitionParser(Words sent):  
    stack_(),
    buffer_next_{0},
    arcs_(sent.size()),
    actions_(),
    sentence_(sent),
    tags_(),
    liw_{0},
    lpw_{0},
    beam_lpw_{0},
    num_particles_{1},
    is_generating_{false}
  {
  }

  TransitionParser(Words sent, Words tags):  
    stack_(),
    buffer_next_{0},
    arcs_(sent.size()),
    actions_(),
    sentence_(sent),
    tags_(tags),
    liw_{0},
    lpw_{0},
    beam_lpw_{0},
    num_particles_{1},
    is_generating_{false}
  {
    //sentence already includes root (0)
    //for (int i = 0; i < (int)sentence_.size(); ++i)
    //  buffer_[i] = sentence_.size() - i - 1; 
  }

  TransitionParser(Words sent, Words tags, int num_particles):  
    stack_(),
    buffer_next_{0},
    arcs_(sent.size()),
    actions_(),
    sentence_(sent),
    tags_(tags),
    liw_{0},
    lpw_{0},
    beam_lpw_{0},
    num_particles_{num_particles},
    is_generating_{false}
  {
  }

  //only use when generating 
  bool buffer_tag(WordId t) {
    if (is_generating_)
      tags_.push_back(t);
    return is_generating_;
  }

  //virtual bool leftArc() = 0;
  //virtual bool rightArc() = 0;
  //virtual kAction oracleNext(const ArcList& gold_arcs) const = 0;
  //virtual bool execute_action(kAction a) = 0;
 
  void pop_buffer() {
    ++buffer_next_;
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

  void add_beam_particle_weight(double lpw) {
    if (beam_lpw_==0)
      beam_lpw_ = lpw;
    else
      beam_lpw_ = neg_log_sum_exp(beam_lpw_, lpw); //add prob in log space
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
      std::cout << dict.lookup_tag(a) << " ";
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

  double beam_particle_weight() const {
    return beam_lpw_;
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

  bool is_generating() const {
    return is_generating_;
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

  bool root_has_child() const {
    return (arcs_.child_count_at(0) > 0);
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

  WordIndex stack_top() const {
    return stack_.back();
  }

  WordIndex stack_top_second() const {
    return stack_.rbegin()[1];
  }

  WordId next_word() const {
    return sentence_[buffer_next_];
  }

  WordId next_tag() const {
    return tags_[buffer_next_];
  }

  WordIndex buffer_next() const {
    return buffer_next_;
  }

  bool buffer_next_has_child() const {
    return arcs_.has_child(buffer_next_);
  }

  bool is_buffer_empty() const {
    if (is_generating_)
      return false;
    else
      return (buffer_next_ >= static_cast<int>(sentence_length())); 
  }

  bool is_complete_parse() const {
    for (WordIndex i = 1; i < arcs_.size() - 1; ++i) {
      if (!arcs_.has_parent(i) && (tags_.at(i)!=1))
        return false;
    }

    return ((buffer_next_ >= 3) && !buffer_next_has_child());
  }

    unsigned directed_accuracy_count(ArcList g_arcs) const {
    unsigned count = 0;
    for (WordIndex j = 1; j < arcs_.size(); ++j) {
      if (arcs_.at(j)==g_arcs.at(j))
        ++count;
    }
    return count;
  }

  unsigned undirected_accuracy_count(ArcList g_arcs) const {
    unsigned count = 0;
    for (WordIndex j = 1; j < arcs_.size(); ++j) {
      if ((arcs_.at(j)==g_arcs.at(j)) || (arcs_.has_parent(j) && g_arcs.at(arcs_.at(j))==static_cast<int>(j)))
        ++count;
    }
    return count;
  }

  //****functions for context vectors: each function defined for a specific context length

  Words tag_raw_context() const {
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

    return ctx;
  }

  Words tag_more_context() const {
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
    Words ctx(2, 0);
    
    if (stack_.size() >= 1) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-2));
    } 

    return ctx;
  }

  Words tag_next_context() const {
    Words ctx(3, 0);
    
    if (stack_.size() >= 1) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[0] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (!is_buffer_empty())
      ctx[2] = tags_.at(buffer_next());

    return ctx;
  }

  Words tag_distance_context() const {
    Words ctx(4, 0);

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

  Words tag_next_distance_context() const {
    Words ctx(3, 0);

    if (stack_.size() >= 1) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-1));
      if (!is_buffer_empty()) {
        WordIndex i = stack_top();
        WordIndex j = buffer_next();
        ctx[2] = j - i;
      }
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
 
  Words word_next_context() const {
    Words ctx(3, 0);
    
    //word context 2 + next token
    if (stack_.size() >= 1) { 
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
    }
    if (!is_buffer_empty())
      ctx[2] = tags_.at(buffer_next_);

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
      if (l2 > 0)
        ctx[3] = tags_.at(l2);
      if (r2 > 0)
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

  Words word_tag_some_children_distance_context() const {
    Words ctx(7, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[6] = tags_.at(stack_.at(stack_.size()-1));
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[2] = tags_.at(l1); //
      if (r1 > 0)
        ctx[3] = tags_.at(r1);
    }
    if (stack_.size() >= 2) {
      ctx[5] = tags_.at(stack_.at(stack_.size()-2));
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));

      WordIndex i = stack_.rbegin()[1];
      WordIndex j = stack_.rbegin()[0];
      ctx[4] = std::min(j - i, 5);
    }
    
    return ctx;
  }

  Words word_tag_children_context() const {
    Words ctx(9, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[8] = tags_.at(stack_.at(stack_.size()-1));
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[4] = tags_.at(l1); //
      if (r1 > 0)
        ctx[6] = tags_.at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[7] = tags_.at(stack_.at(stack_.size()-2));
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
      if (l2 > 0)
        ctx[3] = tags_.at(l2);
      if (r2 > 0)
        ctx[5] = tags_.at(r2); //
    }
     if (stack_.size() >= 3) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-3));
    } /*
    if (stack_.size() >= 4) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-4));
    } */
    return ctx;
  }

  Words tag_children_context() const {
    Words ctx(8, 0);
    if (stack_.size() >= 1) { 
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      
      ctx[7] = tags_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[4] = tags_.at(l1); //
      if (r1 > 0)
        ctx[5] = tags_.at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));
      WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));

      ctx[6] = tags_.at(stack_.at(stack_.size()-2));
      if (l2 > 0)
        ctx[2] = tags_.at(l2);
      if (r2 > 0)
        ctx[3] = tags_.at(r2); //
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
    Words ctx(8, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[7] = tags_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[4] = tags_.at(l1); //
      if (r1 > 0)
        ctx[5] = tags_.at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[6] = tags_.at(stack_.at(stack_.size()-2));
      if (l2 >= 0)
        ctx[1] = tags_.at(l2);
      if (r2 >= 0)
        ctx[3] = tags_.at(r2); //
    }
    if (stack_.size() >= 3) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-4));
    }

    return ctx;
  }

  Words tag_less_children_distance_context() const {
    Words ctx(9, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[8] = tags_.at(stack_.at(stack_.size()-1));
      if (r1 > 0)
        ctx[5] = tags_.at(r1);  //[0]
      if (l1 > 0)
        ctx[6] = tags_.at(l1);
    }
    if (stack_.size() >= 2) {
      WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[7] = tags_.at(stack_.at(stack_.size()-2));
      if (r2 >= 0)
        ctx[3] = tags_.at(r2);
      if (l2 >= 0)
        ctx[2] = tags_.at(l2);  //[1]
      
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

    return ctx;
  }

  Words tag_some_children_distance_context() const {
    Words ctx(5, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[4] = tags_.at(stack_.at(stack_.size()-1));
      if (r1 > 0)
        ctx[1] = tags_.at(r1);  //[0]
      if (l1 > 0)
        ctx[0] = tags_.at(l1);
    }
    if (stack_.size() >= 2) {
      //WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      //WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[3] = tags_.at(stack_.at(stack_.size()-2));
      //if (r2 >= 0)
      //  ctx[0] = tags_.at(r2);
      //if (l2 >= 0)
      //  ctx[0] = tags_.at(l2);  //[1]
      
      WordIndex i = stack_.rbegin()[1];
      WordIndex j = stack_.rbegin()[0];
      ctx[2] = std::min(j - i, 5);
    }

    return ctx;
  }

  Words tag_some_children_context() const {
    Words ctx(4, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
      if (r1 > 0)
        ctx[1] = tags_.at(r1);  //[0]
      if (l1 > 0)
        ctx[0] = tags_.at(l1);
    }
    if (stack_.size() >= 2) {
      //WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      //WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[2] = tags_.at(stack_.at(stack_.size()-2));
      //if (r2 >= 0)
      //  ctx[0] = tags_.at(r2);
      //if (l2 >= 0)
      //  ctx[0] = tags_.at(l2);  //[1]
    }

    //if (stack_.size() >= 3) { 
    //  ctx[0] = tags_.at(stack_.at(stack_.size()-3)); //
    //}
    return ctx;
  }

  Words tag_next_children_more_context() const {
    Words ctx(7, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex r2 = arcs_.right_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      //if (arcs_.has_parent(stack_.at(stack_.size()-1)))
        //ctx[0] = tags_.at(arcs_.at(stack_.at(stack_.size()-1)));
      ctx[6] = tags_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[2] = tags_.at(l1); //
      if (r1 > 0)
        ctx[5] = tags_.at(r1);
      if (r2 > 0)
       ctx[1] = tags_.at(r2);
    }

    if (!is_buffer_empty()) {
      WordIndex bl1 = arcs_.leftmost_child(buffer_next_);
      WordIndex bl2 = arcs_.left_child(buffer_next_);

      if (bl1 > 0)
        ctx[4] = tags_.at(bl1);
        
      if (bl2 > 0)
        ctx[1] = tags_.at(bl2);
    }
    
    if (stack_.size() >= 2) { 
      ctx[3] = tags_.at(stack_.at(stack_.size()-2)); //
    }
    
    return ctx;
  }

  Words tag_next_children_less_context() const {
    Words ctx(4, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      //WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
      //if (l1 > 0)
      //  ctx[0] = tags_.at(l1); //
      if (r1 > 0)
        ctx[1] = tags_.at(r1);

      //if (arcs_.has_parent(stack_.at(stack_.size()-1)))
        //ctx[0] = tags_.at(arcs_.at(stack_.at(stack_.size()-1)));
    }

    if (!is_buffer_empty()) {
      WordIndex bl1 = arcs_.leftmost_child(buffer_next_);
      if (bl1 > 0)
        ctx[2] = tags_.at(bl1);
    }
    
    if (stack_.size() >= 2) { 
      ctx[0] = tags_.at(stack_.at(stack_.size()-2));
    }
    
    return ctx;
  }

  Words tag_next_children_distance_context() const {
    Words ctx(7, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[6] = tags_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[0] = tags_.at(l1); 
      if (r1 > 0)
        ctx[3] = tags_.at(r1); 

      if (!is_buffer_empty()) {
        WordIndex i = stack_top();
        WordIndex j = buffer_next();
        ctx[5] = j - i;
      }
    }

    if (!is_buffer_empty()) {
      WordIndex bl1 = arcs_.leftmost_child(buffer_next());
      if (bl1 > 0)
        ctx[4] = tags_.at(bl1); 
    }
    
    if (stack_.size() >= 2) { 
      ctx[2] = tags_.at(stack_.at(stack_.size()-2)); 
    }
    
    if (stack_.size() >= 3) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-3)); 
    }

    return ctx;
  }
  
  Words tag_next_children_distance_some_context() const {
    Words ctx(6, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      //WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[5] = tags_.at(stack_.at(stack_.size()-1));
      //if (l1 > 0)
      //  ctx[0] = tags_.at(l1); 
      if (r1 > 0)
        ctx[2] = tags_.at(r1); 

      if (!is_buffer_empty()) {
        WordIndex i = stack_top();
        WordIndex j = buffer_next();
        ctx[4] = j - i;
      }
    }

    if (!is_buffer_empty()) {
      WordIndex bl1 = arcs_.leftmost_child(buffer_next());
      if (bl1 > 0)
        ctx[3] = tags_.at(bl1); 
    }
    
    if (stack_.size() >= 2) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-2)); 
    }
    
    if (stack_.size() >= 3) { 
      ctx[0] = tags_.at(stack_.at(stack_.size()-3)); 
    }

    return ctx;
  }

  Words tag_next_children_context() const {
    Words ctx(6, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[5] = tags_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[0] = tags_.at(l1); 
      if (r1 > 0)
        ctx[3] = tags_.at(r1); 

    }

    if (!is_buffer_empty()) {
      WordIndex bl1 = arcs_.leftmost_child(buffer_next());
      if (bl1 > 0)
        ctx[4] = tags_.at(bl1); 
    }
    
    if (stack_.size() >= 2) { 
      ctx[2] = tags_.at(stack_.at(stack_.size()-2)); 
    }
    
    if (stack_.size() >= 3) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-3)); 
    }

    return ctx;
  }

  Words tag_next_children_some_context() const {
    Words ctx(4, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      //WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
      //if (l1 > 0)
      //  ctx[0] = tags_.at(l1); 
      if (r1 > 0)
        ctx[1] = tags_.at(r1); 

    }

    if (!is_buffer_empty()) {
      WordIndex bl1 = arcs_.leftmost_child(buffer_next());
      if (bl1 > 0)
        ctx[2] = tags_.at(bl1); 
    }
    
    if (stack_.size() >= 2) { 
      ctx[0] = tags_.at(stack_.at(stack_.size()-2)); 
    }
    
    //if (stack_.size() >= 3) { 
    //  ctx[1] = tags_.at(stack_.at(stack_.size()-3)); 
   // }

    return ctx;
  }

  Words tag_next_children_word_distance_context() const {
    Words ctx(7, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      //WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[6] = tags_.at(stack_.at(stack_.size()-1));
      ctx[3] = sentence_.at(stack_.at(stack_.size()-1));
      //if (l1 > 0)
      //  ctx[0] = tags_.at(l1); 
      if (r1 > 0)
        ctx[2] = tags_.at(r1); 

      if (!is_buffer_empty()) {
        WordIndex i = stack_top();
        WordIndex j = buffer_next();
        ctx[5] = j - i;
      }
    }

    if (!is_buffer_empty()) {
      WordIndex bl1 = arcs_.leftmost_child(buffer_next());
      if (bl1 > 0)
        ctx[4] = tags_.at(bl1); 
    }
    
    if (stack_.size() >= 2) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-2)); 
    }
    
    if (stack_.size() >= 3) { 
      ctx[0] = tags_.at(stack_.at(stack_.size()-3)); 
    }

    return ctx;
  }
  

  Words tag_next_children_word_context() const {
    Words ctx(7, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[6] = tags_.at(stack_.at(stack_.size()-1));
      ctx[4] = sentence_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[0] = tags_.at(l1); 
      if (r1 > 0)
        ctx[3] = tags_.at(r1); 

    }

    if (!is_buffer_empty()) {
      WordIndex bl1 = arcs_.leftmost_child(buffer_next());
      if (bl1 > 0)
        ctx[5] = tags_.at(bl1); 
    }
    
    if (stack_.size() >= 2) { 
      ctx[2] = tags_.at(stack_.at(stack_.size()-2)); 
    }
    
    if (stack_.size() >= 3) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-3)); 
    }

    return ctx;
  }
  
  Words word_next_tag_children_context() const {
    Words ctx(5, 0);
    
    //word context 2, pos context 2 + next token
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      //WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[3] = sentence_.at(stack_.at(stack_.size()-1));
      //ctx[4] = tags_.at(stack_.at(stack_.size()-1));
      //if (l1 > 0)
      //  ctx[1] = tags_.at(l1); //
      if (r1 > 0)
        ctx[1] = tags_.at(r1);
    }
    if (stack_.size() >= 2) { 
      WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      //WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[3] = sentence_.at(stack_.at(stack_.size()-2));
      //ctx[3] = tags_.at(stack_.at(stack_.size()-2));
      //if (l2 > 0)
      //  ctx[0] = tags_.at(l2);
      if (r2 > 0)
        ctx[0] = tags_.at(r2); //
    }
    if (!is_buffer_empty()) 
      ctx[4] = tags_.at(buffer_next());

    if (stack_.size() >= 3) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-3));
    }
    /* if (stack_.size() >= 4) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-4));
    } */

    return ctx;
  }

  Words word_next_children_context() const {
    Words ctx(7, 0);
    
    //word context 2, pos context 2 + next token
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[5] = sentence_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[1] = tags_.at(l1); //
      if (r1 > 0)
        ctx[3] = tags_.at(r1);
    }
    if (stack_.size() >= 2) { 
      WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[4] = sentence_.at(stack_.at(stack_.size()-2));
      if (l2 > 0)
        ctx[0] = tags_.at(l2);
      if (r2 > 0)
        ctx[2] = tags_.at(r2); //
    }
    if (!is_buffer_empty()) 
      ctx[6] = tags_.at(buffer_next());

    /* if (stack_.size() >= 3) {
      ctx[3] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-4));
    } */

    return ctx;
  }

  Words word_tag_next_children_context() const {
    Words ctx(5, 0);
    
    //word context 2, pos context 2 + next token
    if (stack_.size() >= 1) { 
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
      //ctx[3] = tags_.at(stack_.at(stack_.size()-1));
      if (l1 > 0)
        ctx[2] = tags_.at(l1); //
      if (r1 > 0)
        ctx[3] = tags_.at(r1);
    }
    if (stack_.size() >= 2) { 
      //WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));
      //WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));

      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
      //ctx[0] = tags_.at(stack_.at(stack_.size()-2));
      //if (l2 > 0)
      //  ctx[2] = tags_.at(l2);
      //if (r2 > 0)
      //  ctx[0] = tags_.at(r2); //
    }
    if (!is_buffer_empty())
      ctx[4] = tags_.at(buffer_next());

    /*if (stack_.size() >= 3) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-3));
    }
     if (stack_.size() >= 4) {
      ctx[2] = tags_.at(stack_.at(stack_.size()-4));
    } */

    return ctx;
  }

  Words word_tag_next_more_context() const {
    Words ctx(7, 0);
    
    //word context 2, pos context 2 + next token
    if (stack_.size() >= 1) { 
      ctx[3] = sentence_.at(stack_.at(stack_.size()-1));
      ctx[5] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[2] = sentence_.at(stack_.at(stack_.size()-2));
      ctx[4] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (!is_buffer_empty()) 
      ctx[6] = tags_.at(buffer_next());

    if (stack_.size() >= 3) {
      ctx[1] = tags_.at(stack_.at(stack_.size()-3));
    }
    if (stack_.size() >= 4) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-4));
    } 

    return ctx;
  }

  Words word_tag_next_context() const {
    Words ctx(5, 0);
    
    //word context 2, pos context 2 + next token
    if (stack_.size() >= 1) { 
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
      ctx[2] = tags_.at(stack_.at(stack_.size()-2));
    }
    /* if (stack_.size() >= 3) { 
      ctx[0] = sentence_.at(stack_.at(stack_.size()-3));
      ctx[2] = tags_.at(stack_.at(stack_.size()-3));
    } */
    if (!is_buffer_empty()) 
      ctx[4] = tags_.at(buffer_next());

    return ctx;
  }

  Words linear_word_tag_extended_context() const {
    Words ctx(7, 0);
    
    WordIndex i = buffer_next();
    if (!is_buffer_empty()) 
      ctx[6] = tags_.at(i);
    

    if (i >= 1) { 
      ctx[5] = sentence_.at(i-1); //4
      //ctx[2] = tags_.at(i-1); //5
    }
    if (i >= 2) { 
      ctx[4] = sentence_.at(i-2); //2
      //ctx[1] = tags_.at(i-2); //3
    }
    if (i >= 3) { 
      ctx[3] = sentence_.at(i-3); //0
      //ctx[0] = tags_.at(i-3); //1
    }
    
    if (stack_.size() >= 1) { 
      //WordIndex l1 = arcs_.leftmost_child(stack_.at(stack_.size()-1));
      WordIndex r1 = arcs_.rightmost_child(stack_.at(stack_.size()-1));
      
      ctx[2] = tags_.at(stack_.at(stack_.size()-1));
      //if (l1 > 0)
      //  ctx[3] = tags_.at(l1); //
      if (r1 > 0)
        ctx[0] = tags_.at(r1);
    }
    if (stack_.size() >= 2) { 
      //WordIndex l2 = arcs_.leftmost_child(stack_.at(stack_.size()-2));
      //WordIndex r2 = arcs_.rightmost_child(stack_.at(stack_.size()-2));

      ctx[1] = tags_.at(stack_.at(stack_.size()-2));
      //if (l2 > 0)
      //  ctx[2] = tags_.at(l2);
      //if (r2 > 0)
      //  ctx[0] = tags_.at(r2); //
    }
    
    /* if (stack_.size() >= 1) { 
      WordIndex j = stack_.at(stack_.size()-1);
      if (j < (i-3))
        ctx[1] = sentence_.at(j); 
    }
    if (stack_.size() >= 2) { 
      WordIndex j = stack_.at(stack_.size()-2);
      if (j < (i-3))
        ctx[0] = sentence_.at(j); 
    } */

    return ctx;
  }

  Words linear_word_tag_next_context() const {
    Words ctx(4, 0);
    
    //word context 2, pos context 2 + next token
    WordIndex i = buffer_next();
    if (i >= 1) { 
      ctx[2] = sentence_.at(i-1); //4
      //ctx[2] = tags_.at(i-1); //5
    }
    if (i >= 2) { 
      ctx[1] = sentence_.at(i-2); //2
      //ctx[1] = tags_.at(i-2); //3
    }
    if (i >= 3) { 
      ctx[0] = sentence_.at(i-3); //0
      //ctx[0] = tags_.at(i-3); //1
    }
    /* if (i >= 4) { 
      ctx[0] = sentence_.at(i-4); //0
      //ctx[0] = tags_.at(i-3); //1
    } */
    if (!is_buffer_empty()) {
      ctx[3] = tags_.at(i);
    }

    return ctx;
  }

  Words linear_tag_context() const {
    Words ctx(5, 0);
    
    WordIndex i = buffer_next();
    if (stack_.size() >= 1) { 
      ctx[4] = tags_.at(i-1);
    }
    if (stack_.size() >= 2) { 
      ctx[3] = tags_.at(i-2);
    }
    if (stack_.size() >= 3) { 
      ctx[2] = tags_.at(i-3);
    }
    if (stack_.size() >= 4) { 
      ctx[1] = tags_.at(i-4);
    } 
    if (stack_.size() >= 5) { 
      ctx[0] = tags_.at(i-5);
    } 

    return ctx;
  }

  private:
  WxList stack_;
  WordIndex buffer_next_;
  ArcList arcs_;
  ActList actions_;
  Words sentence_;
  Words tags_;
  double liw_; //log importance weight
  double lpw_; //log particle weight
  double beam_lpw_; //cummulative beam log particle weight
  int num_particles_;
  bool is_generating_;
};

class ArcStandardParser : public TransitionParser {
  public:

  ArcStandardParser():
    TransitionParser() {
  }

  ArcStandardParser(Words sent):
    TransitionParser(sent) {
  }

  ArcStandardParser(Words sent, Words tags):
    TransitionParser(sent, tags) {
  }

  ArcStandardParser(Words sent, Words tags, int num_particles):
    TransitionParser(sent, tags, num_particles) {
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

  bool is_terminal_configuration() const {
    if (is_generating()) 
      return ((buffer_next() >= 3) && (stack_depth() == 1)); //&& !buffer_next_has_child());
    else     
      return (is_buffer_empty() && (stack_depth() == 1));
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
    //return linear_word_tag_next_context(); //best perplexity
    //return word_tag_next_context(); 
    return word_tag_next_children_context(); //best context (order 6)
  }

  Words reduce_context() const {
    return word_tag_children_context(); //best full context, lexicalized (order 10)
    //return word_tag_some_children_distance_context(); //best smaller context, lexicalized (order 8)
    //return tag_children_context(); //best full context (order 9)
    //return tag_some_children_distance_context(); //best smaller context (order 6)
  }

  Words arc_context() const {
    return tag_less_context();
  }

  Words tag_context() const {
    //return linear_tag_context();
    return tag_children_context();  //best full context (order 9)
    //return tag_some_children_context(); //best smaller context (order 5)
  }

  kAction oracleNext(const ArcList& gold_arcs) const;
  
  kAction oracleDynamicNext(const ArcList& gold_arcs) const;
};

class ArcEagerParser : public TransitionParser {
  public:

  ArcEagerParser(): 
    TransitionParser()
  {
  }

  ArcEagerParser(Words sent): 
    TransitionParser(sent)
  {
  }

  ArcEagerParser(Words sent, Words tags): 
    TransitionParser(sent, tags)
  {
  }

  ArcEagerParser(Words sent, Words tags, int num_particles):
    TransitionParser(sent, tags, num_particles) 
  {
  }

  bool shift();
  
  bool shift(WordId w);

  bool leftArc();

  bool rightArc();
  
  bool rightArc(WordId w);
  
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
    //if STOP, should not have parent, else it should
    if (tag_at(i) == 1)
      return !has_parent(i);
    else
      return has_parent(i);
  }

  bool is_terminal_configuration() const {
    //last word generated is STOP
    return (!is_stack_empty() && (tag_at(stack_top()) == 1)); // && !buffer_next_has_child());
    //return (!is_stack_empty() && (stack_top() == static_cast<int>(sentence_length() - 1))); // && !buffer_next_has_child());

    //return ((tag_at(stack_top()) == 1)); // && !buffer_next_has_child());
    //if (is_generating()) 
    //  return ((buffer_next() >= 3) && (stack_depth() == 1)); 
    //else     
    //  return (is_buffer_empty() && (stack_depth() == 1));
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
    return word_tag_next_children_context();  //(order 6)
    //return word_tag_next_context();
  }

  Words reduce_context() const {
    //return tag_next_children_distance_some_context(); //smaller context
    //return tag_next_children_distance_context(); //full
    //return tag_next_children_word_context(); //lexicalized, full context (?)
    return tag_next_children_word_distance_context(); //lexicalized, smaller context (order 8)
  }

  Words arc_context() const {
    return tag_less_context();
  }
  
  Words tag_context(kAction a) const {
    Words ctx = tag_next_children_some_context(); //smaller context (order 6)
    //Words ctx = tag_next_children_context(); //full context
    ctx.push_back(ctx.back());
    if (a == kAction::ra)
      ctx.at(ctx.size()-2) = 1;
    else
      ctx.at(ctx.size()-2) = 0;
    return ctx;
  }

  kAction oracleNext(const ArcList& gold_arcs) const;
};

class AccuracyCounts {

public:
  AccuracyCounts(): 
    likelihood_{0},
    beam_likelihood_{0},
    importance_likelihood_{0},
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

  void add_importance_likelihood(double l) {
    importance_likelihood_ += l;
  }

  void add_beam_likelihood(double l) {
    beam_likelihood_ += l;
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

  void countAccuracy(const ArcEagerParser& prop_parse, const ArcEagerParser& gold_parse); 

  void countAccuracy(const EisnerParser& prop_parse, const EisnerParser& gold_parse); 

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

  double importance_likelihood() const {
    return importance_likelihood_;
  }

  double beam_likelihood() const {
    return beam_likelihood_;
  }

  double gold_likelihood() const {
    return gold_likelihood_;
  }

  int total_length() const {
    return total_length_;
  }

  double final_reduce_error_rate() const {
    return (final_reduce_error_count_ + 0.0)/total_length_;
  }

  double cross_entropy() const {
    return likelihood_/(std::log(2)*total_length_);
    //return likelihood_/(std::log(2)*num_actions_);
  }

  double beam_cross_entropy() const {
    return beam_likelihood_/(std::log(2)*total_length_);
    //return beam_likelihood_/(std::log(2)*num_actions_);
  }
  
  double importance_cross_entropy() const {
    return importance_likelihood_/(std::log(2)*total_length_);
  }
  
  double gold_cross_entropy() const {
    return gold_likelihood_/(std::log(2)*total_length_);
    //return gold_likelihood_/(std::log(2)*num_actions_);
  }

  double perplexity() const {
    return std::pow(2, cross_entropy());
  }

  double beam_perplexity() const {
    return std::pow(2, beam_cross_entropy());
  }

  double importance_perplexity() const {
    return std::pow(2, importance_cross_entropy());
  }

  double gold_perplexity() const {
    return std::pow(2, gold_cross_entropy());
  }

private:
    double likelihood_;  
    double beam_likelihood_;  
    double importance_likelihood_;  
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
