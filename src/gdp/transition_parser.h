#ifndef _GDP_TR_PARSER_H_
#define _GDP_TR_PARSER_H_

#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "utils/random.h"
#include "corpus/dict.h"
#include "corpus/model_config.h"
#include "corpus/parse_data_set.h"
#include "gdp/utils.h"
#include "gdp/parser.h"

namespace oxlm {

class TransitionParser: public Parser {
  public:

  //constructor for generating from model
  TransitionParser(const boost::shared_ptr<ModelConfig>& config);
  
  TransitionParser(const TaggedSentence& parse, const boost::shared_ptr<ModelConfig>& config);  
  
  TransitionParser(const TaggedSentence& parse, int num_particles, const boost::shared_ptr<ModelConfig>& config);  

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
  
  void set_importance_weight(Real w) {
    importance_weight_ = w;
  }

  void add_importance_weight(Real w) {
    importance_weight_ += w;
  }

  void set_particle_weight(Real w) {
    set_weight(w);
  }

  void add_particle_weight(Real w) {
    add_weight(w);
  }

  void add_log_particle_weight(Real w) {
    if (weight() == 0)
      set_weight(w);
    else
      set_weight(neg_log_sum_exp(weight(), w)); //add in log space
  }

  void add_beam_weight(Real w) {
    if (beam_weight_ == 0)
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
    if (!root_first() && (buffer_next_ == static_cast<int>(size())))
      return 0;
    else
      return buffer_next_;
  }

  bool buffer_empty() const {
    //for root-last, root occurs after the end of sentence
    if (root_first())
      return (buffer_next_ >= static_cast<int>(size()));
    else
      return (buffer_next_ > static_cast<int>(size()));
  }

  WordId next_word() const {
    return word_at(buffer_next());
  }

  WordId next_tag() const {
    return tag_at(buffer_next());
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

  int num_labels() const {
    return config_->num_labels;
  }

  bool root_first() const {
    return config_->root_first;
  }

  std::vector<std::string> action_str_list() const {
    const std::vector<std::string> action_names {"sh", "la", "ra", "re", "la2", "ra2"};
    std::vector<std::string> list;
    for (kAction a: actions_)
      list.push_back(action_names[static_cast<int>(a)]);
    return list; 
  }
  
  void print_actions() const {
    for (auto act: action_str_list())
      std::cout << act << " ";
    std::cout << std::endl;
  }

  Real particle_weight() const {
    return weight();
  }

   Real importance_weight() const {
    return importance_weight_;
  }

  Real beam_weight() const {
    return beam_weight_;
  }

  //number of particles associated with (partial) parse
  int num_particles() const {
    return num_particles_;
  }

  Real weighted_particle_weight() const {
    return (weight() - std::log(num_particles_));
  }

  Real weighted_importance_weight() const {
    return (importance_weight_ - std::log(num_particles_));
  }

  boost::shared_ptr<ModelConfig> config() const {
    return boost::shared_ptr<ModelConfig>(config_);
  }

  bool pyp_model() const {
    return config_->pyp_model;
  }

  bool lexicalised() const {
    return config_->lexicalised;
  }

  std::string context_type() const {
    return config_->context_type;
  }

  Context map_context(Indices ind) const {
    Words words;
    WordsList features;
    for (auto i: ind) {
      if (i >= 0) {
        if (config_->lexicalised)
          words.push_back(word_at(i));
        else {
          words.push_back(tag_at(i));
          //if (tag_at(i)==-1)
          //  std::cout << i << " ";
        }
        features.push_back(features_at(i));
      } else {
        words.push_back(0);
        features.push_back(Words(1, 0));
      }          
    }
    
    return Context(words, features);
  }

  /*  functions for context vectors  */

  Words word_tag_next_children_context() const {
    Words ctx(8, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size()-1));
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size()-1));
      
      ctx[2] = word_at(stack_.at(stack_.size()-1));
      ctx[5] = tag_at(stack_.at(stack_.size()-1));
      if (l1 >= 0) 
        ctx[3] = tag_at(l1);
      if (r1 >= 0) 
        ctx[4] = tag_at(r1);
    }

    if (stack_.size() >= 2) { 
      ctx[0] = word_at(stack_.at(stack_.size()-2));
    }

    if (!buffer_empty()) {
      ctx[7] = tag_at(buffer_next());
      WordIndex p = arc_at(buffer_next());
      if (p >= 0) {
        ctx[6] = tag_at(p);
        //ctx[1] = word_at(p);
      }
    }

    return ctx;
  }
  Words word_tag_next_ngram_context() const {
    Words ctx(4, 0);
 
    WordIndex i = buffer_next();

    if (!buffer_empty()) {
      ctx[3] = tag_at(i);
    }

    if (i >= 1) { 
      ctx[2] = word_at(i-1); 
    }
    if (i >= 2) { 
      ctx[1] = word_at(i-2); 
    }
    if (i >= 3) { 
      ctx[0] = word_at(i-3); 
    }
    
    return ctx;
  }

  Indices children_lookahead_context() const {
    Indices ctx(9, -1);
    if (stack_.size() >= 1) { 
      ctx[0] = stack_.at(stack_.size()-1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size()-1)); 
      ctx[2] = leftmost_child_at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[3] = stack_.at(stack_.size()-2);
      ctx[4] = rightmost_child_at(stack_.at(stack_.size()-2));
      ctx[5] = leftmost_child_at(stack_.at(stack_.size()-2));
    }

    if (buffer_next() < size())
      ctx[6] = buffer_next();
    if (buffer_next() + 1 < size())
      ctx[7] = buffer_next() + 1;
    if (buffer_next() + 2 < size())
      ctx[8] = buffer_next() + 2;

    return ctx;
  }

  Indices children_context() const {
    Indices ctx(6, -1);
    if (stack_.size() >= 1) { 
      ctx[0] = stack_.at(stack_.size()-1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size()-1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[3] = stack_.at(stack_.size()-2);
      ctx[4] = rightmost_child_at(stack_.at(stack_.size()-2));
      ctx[5] = leftmost_child_at(stack_.at(stack_.size()-2));
    }

    return ctx;
  }

  Indices children_ngram_context() const {
    Indices ctx(9, -1);
    if (stack_.size() >= 1) { 
      ctx[0] = stack_.at(stack_.size()-1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size()-1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[3] = stack_.at(stack_.size()-2);
      ctx[4] = rightmost_child_at(stack_.at(stack_.size()-2));
      ctx[5] = leftmost_child_at(stack_.at(stack_.size()-2));
    }

    if (buffer_next() >= 1)
      ctx[6] = buffer_next()-1;
    if (buffer_next() >= 2)
      ctx[7] = buffer_next()-2;
    if (buffer_next() >= 3)
      ctx[8] = buffer_next()-3;
    
    return ctx;
  }

  Indices extended_children_context() const {
    Indices ctx(12, -1);
    if (stack_.size() >= 1) { 
      ctx[0] = stack_.at(stack_.size()-1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size()-1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size()-1));
      ctx[3] = second_rightmost_child_at(stack_.at(stack_.size()-1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[5] = stack_.at(stack_.size()-2);
      ctx[6] = rightmost_child_at(stack_.at(stack_.size()-2));
      ctx[7] = leftmost_child_at(stack_.at(stack_.size()-2));
      ctx[8] = second_rightmost_child_at(stack_.at(stack_.size()-2));
      ctx[9] = second_leftmost_child_at(stack_.at(stack_.size()-2));
    }

    if (stack_.size() >= 3) {
      ctx[10] = stack_.at(stack_.size()-3);
    }
    if (stack_.size() >= 4) {
      ctx[11] = stack_.at(stack_.size()-4);
    } 

    return ctx;
  }

  Indices more_extended_children_context() const {
    Indices ctx(16, -1);
    if (stack_.size() >= 1) { 
      ctx[0] = stack_.at(stack_.size()-1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size()-1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size()-1));
      ctx[3] = second_rightmost_child_at(stack_.at(stack_.size()-1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size()-1));
      ctx[5] = rightmost_grandchild_at(stack_.at(stack_.size()-1));
      ctx[6] = leftmost_grandchild_at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[7] = stack_.at(stack_.size()-2);
      ctx[8] = rightmost_child_at(stack_.at(stack_.size()-2));
      ctx[9] = leftmost_child_at(stack_.at(stack_.size()-2));
      ctx[10] = second_rightmost_child_at(stack_.at(stack_.size()-2));
      ctx[11] = second_leftmost_child_at(stack_.at(stack_.size()-2));
      ctx[12] = rightmost_grandchild_at(stack_.at(stack_.size()-2));
      ctx[13] = leftmost_grandchild_at(stack_.at(stack_.size()-2));
    }

    if (stack_.size() >= 3) {
      ctx[10] = stack_.at(stack_.size()-3);
    }
    if (stack_.size() >= 4) {
      ctx[11] = stack_.at(stack_.size()-4);
    } 

    return ctx;
  }

 Words tag_children_context() const {
    Words ctx(7, 0);
    if (stack_.size() >= 1) { 
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size()-1));
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size()-1));
      
      ctx[6] = tag_at(stack_.at(stack_.size()-1));
      if (l1 >= 0)
        ctx[2] = tag_at(l1); //
      if (r1 >= 0)
        ctx[4] = tag_at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex l2 = leftmost_child_at(stack_.at(stack_.size()-2));
      WordIndex r2 = rightmost_child_at(stack_.at(stack_.size()-2));

      ctx[5] = tag_at(stack_.at(stack_.size()-2));
      if (l2 >= 0)
        ctx[1] = tag_at(l2);
      if (r2 >= 0)
        ctx[3] = tag_at(r2); //
    }
    if (stack_.size() >= 3) {
      ctx[0] = tag_at(stack_.at(stack_.size()-3));
    }
    /*if (stack_.size() >= 4) {
      ctx[0] = tag_at(stack_.at(stack_.size()-4));
    } */ 
    return ctx;
  }

 Words word_tag_children_context() const {
    Words ctx(9, 0);
    if (stack_.size() >= 1) { 
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size()-1));
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size()-1));
      
      ctx[8] = tag_at(stack_.at(stack_.size()-1));
      ctx[1] = word_at(stack_.at(stack_.size()-1));
      if (l1 >= 0)
        ctx[4] = tag_at(l1); //
      if (r1 >= 0)
        ctx[6] = tag_at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex r2 = rightmost_child_at(stack_.at(stack_.size()-2));
      WordIndex l2 = leftmost_child_at(stack_.at(stack_.size()-2));

      ctx[7] = tag_at(stack_.at(stack_.size()-2));
      ctx[0] = word_at(stack_.at(stack_.size()-2));
      if (l2 >= 0)
        ctx[3] = tag_at(l2);
      if (r2 >= 0)
        ctx[5] = tag_at(r2); //
    }
    if (stack_.size() >= 3) {
      ctx[2] = tag_at(stack_.at(stack_.size()-3));
    } 
     
    return ctx;
  }

 Indices next_children_context() const {
    Indices ctx(8, -1);
    
    if (stack_.size() >= 1) { 
      ctx[0] = stack_.at(stack_.size()-1);
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size()-1));
      if (r1 != buffer_next()) 
        ctx[1] = r1;
      ctx[2] = second_rightmost_child_at(stack_.at(stack_.size()-1));
      ctx[3] = leftmost_child_at(stack_.at(stack_.size()-1));
      ctx[4] = arc_at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[5] = stack_.at(stack_.size()-2);
    }

    if (!buffer_empty()) {
      ctx[6] = leftmost_child_at(buffer_next());
      ctx[7] = second_leftmost_child_at(buffer_next());
    }

    return ctx;
  }

  Indices next_children_lookahead_context() const {
    Indices ctx(11, -1);
    
    if (stack_.size() >= 1) { 
      ctx[0] = stack_.at(stack_.size()-1);
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size()-1));
      ctx[1] = r1;
      ctx[2] = second_rightmost_child_at(stack_.at(stack_.size()-1));
      ctx[3] = leftmost_child_at(stack_.at(stack_.size()-1));
      ctx[4] = arc_at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[5] = stack_.at(stack_.size()-2);
    }

    if (!buffer_empty()) {
      ctx[6] = leftmost_child_at(buffer_next());
      ctx[7] = second_leftmost_child_at(buffer_next());
    }

    if (buffer_next() < size())
      ctx[8] = buffer_next();
    if (buffer_next() + 1 < size())
      ctx[9] = buffer_next() + 1;
    if (buffer_next() + 2 < size())
      ctx[10] = buffer_next() + 2;

    return ctx;
  }

  Indices extended_next_children_context() const {
    Indices ctx(13, -1);
   
    if (stack_.size() >= 1) { 
      ctx[0] = stack_.at(stack_.size()-1);
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size()-1));
      if (r1 != buffer_next()) 
        ctx[1] = r1;
      ctx[2] = second_rightmost_child_at(stack_.at(stack_.size()-1));
      ctx[3] = leftmost_child_at(stack_.at(stack_.size()-1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size()-1));
      ctx[5] = arc_at(stack_.at(stack_.size()-1));
      if (ctx[5] >= 0)
        ctx[6] = arc_at(ctx[5]); //grandparent
    }
    if (stack_.size() >= 2) { 
      ctx[7] = stack_.at(stack_.size()-2);
      ctx[8] = rightmost_child_at(stack_.at(stack_.size()-2));
      ctx[9] = leftmost_child_at(stack_.at(stack_.size()-2));
    }

    if (!buffer_empty()) {
      ctx[10] = leftmost_child_at(buffer_next());
      ctx[11] = second_leftmost_child_at(buffer_next());
    }

    if (stack_.size() >= 3) {
      ctx[12] = stack_.at(stack_.size()-3);
    }

    return ctx;
  }

  Words tag_next_children_context() const {
    Words ctx(6, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size()-1));
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size()-1));
      
      ctx[5] = tag_at(stack_.at(stack_.size()-1));
      if (l1 >= 0)
        ctx[2] = tag_at(l1); 
      if (r1 >= 0 && (r1 != buffer_next())) 
        ctx[3] = tag_at(r1); 

    }

    if (!buffer_empty()) {
      WordIndex bl1 = leftmost_child_at(buffer_next());
      if (bl1 >= 0)
        ctx[4] = tag_at(bl1); 
    }
    
    if (stack_.size() >= 2) { 
      ctx[1] = tag_at(stack_.at(stack_.size()-2)); 
    }
    
    if (stack_.size() >= 3) { 
      ctx[0] = tag_at(stack_.at(stack_.size()-3)); 
    }

    return ctx;
  }

  Words tag_next_children_word_context() const {
    Words ctx(7, 0);
    
    if (stack_.size() >= 1) { 
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size()-1));
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size()-1));
      
      ctx[6] = tag_at(stack_.at(stack_.size()-1));
      ctx[0] = word_at(stack_.at(stack_.size()-1));
      if (l1 >= 0)
        ctx[3] = tag_at(l1); 
      if (r1 >= 0 && (r1 != buffer_next())) 
        ctx[4] = tag_at(r1); 

    }

    if (!buffer_empty()) {
      WordIndex bl1 = leftmost_child_at(buffer_next());
      if (bl1 >= 0)
        ctx[5] = tag_at(bl1); 
    }
    
    if (stack_.size() >= 2) { 
      ctx[2] = tag_at(stack_.at(stack_.size()-2)); 
    }
    
    if (stack_.size() >= 3) { 
      ctx[1] = tag_at(stack_.at(stack_.size()-3)); 
    }

    return ctx;
  }
 
  static bool cmp_particle_weights(const boost::shared_ptr<TransitionParser>& p1, 
                          const boost::shared_ptr<TransitionParser>& p2) {
    //null should be the biggest
    if ((p1 == nullptr) || (p1->num_particles() == 0))
      return false;
    else if ((p2 == nullptr) || (p2->num_particles() == 0))
      return true;
    else
      return (p1->particle_weight() < p2->particle_weight());
  }


  static bool cmp_weighted_importance_weights(const boost::shared_ptr<TransitionParser>& p1, 
                                       const boost::shared_ptr<TransitionParser>& p2) {
    //null or no particles should be the biggest
    if ((p1 == nullptr) || (p1->num_particles() == 0))
      return false;
    else if ((p2 == nullptr) || (p2->num_particles() == 0))
      return true;
    else
      return (p1->weighted_importance_weight() < p2->weighted_importance_weight());
  } 

 private:
  Indices stack_;
  WordIndex buffer_next_;
  ActList actions_;
  Real importance_weight_; 
  Real beam_weight_; 
  int num_particles_;
  boost::shared_ptr<ModelConfig> config_;
};

}
#endif
