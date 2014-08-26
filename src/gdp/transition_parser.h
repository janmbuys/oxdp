#ifndef _GDP_TR_PARSER_H_
#define _GDP_TR_PARSER_H_

// #include<string>
//  #include<functional>
//  #include<cstdlib>

//#include "utils/random.h"
#include "corpus/dict.h"
#include "parse.h"

namespace oxlm {

enum class kAction: WordId {sh, la, ra, re, la2, ra2};

typedef std::vector<kAction> ActList;

//don't think I'm using this anywhere
/* inline kAction convert_to_action(WordId a) {
  std::vector<kAction> actList = {kAction::sh, kAction::la, kAction::ra, kAction::re, kAction::la2, kAction::ra2};
  return actList[a];
} */

class TransitionParser: public Parse {
  public:

  //use specifically when generating from model
  TransitionParser();
  
  TransitionParser(Words tags);

  TransitionParser(Words sent, Words tags); 
 
  TransitionParser(Words sent, Words tags, int num_particles);

  virtual bool leftArc() = 0;

  virtual bool rightArc() = 0;

  virtual kAction oracleNext(const Parse& gold_arcs) const = 0;
 
  void pop_buffer() {
    ++buffer_next_;
  }

  void pop_stack() {
    stack_.pop_back();
  }

  void push_stack(WordIndex i) {
    stack_.push_back(i);
  }

  void append_action(kAction a) {
    actions_.push_back(a);
  }

  void reset_importance_weight() {
    importance_weight_ = 0;
  }

  void set_importance_weight(double w) {
    importance_weight_ = -std::log(w);
  }

  void add_importance_weight(double w) {
    importance_weight_ -= std::log(w);
  }

  //void set_log_particle_weight(double w) -> set_weight

  void set_particle_weight(double w) {
    set_weight(-std::log(w));
  }

  void add_particle_weight(double w) {
    add_weight(-std::log(w));
  }

  void add_beam_weight(double w) {
    if (beam_weight_==0)
      beam_weight_ = w;
    else
      beam_weight_ = neg_log_sum_exp(beam_weight_, w); //add in log space
  }

  void set_num_particles(int n) {
    num_particles_ = n;
  }

  int stack_depth() const {
    return stack_.size();
  }

  bool stack_empty() const {
    return stack_.empty();
  }

  WordIndex stack_top() const {
    return stack_.back();
  }

  WordIndex stack_top_second() const {
    return stack_.rbegin()[1];
  }

  WordIndex buffer_next() const {
    return buffer_next_;
  }

  bool buffer_empty() const {
    return (buffer_next_ >= static_cast<int>(size())); 
  }

  WordId next_word() const {
    return word_at(buffer_next_);
  }

  WordId next_tag() const {
    return tag_at(buffer_next_);
  }

  ActList actions() const {
    return actions_;
  }
   
  //do we need this?
  /* kAction last_action() const {
    return actions_.back();
  } */

  unsigned num_actions() const {
    return actions_.size();
  }

  std::vector<std::string> action_str_list() const {
    const std::vector<std::string> action_names {"sh", "la", "ra", "re", "la2", "ra2"};
    std::vector<std::string> list = "";
    for (kAction a: actions_)
      list.push_back(action_names[static_cast<int>(a)]);
    return list; 
  }
  
  void print_actions() const {
    for (auto act: action_str_list())
      std::cout << act << " ";
    std::cout << std::endl;
  }

  double particle_weight() const {
    return weight();
  }

   double importance_weight() const {
    return importance_weight_;
  }

  double beam_weight() const {
    return beam_weight_;
  }

  int num_particles() const {
    return num_particles_;
  }

  double weighted_particle_weight() const {
    return (weight() - std::log(num_particles_));
  }

  double weighted_importance_weight() const {
    return (importance_weight_ - std::log(num_particles_));
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
  Indices stack_;
  WordIndex buffer_next_;
  ActList actions_;
  double importance_weight_; 
  double beam_weight_; //cummulative beam log particle weight
  int num_particles_;
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

}
#endif
