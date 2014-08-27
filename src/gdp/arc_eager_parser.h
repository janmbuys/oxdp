#ifndef _GDP_AE_PARSER_H_
#define _GDP_AE_PARSER_H_

#include "transition_parser.h"

namespace oxlm {

class ArcEagerParser : public TransitionParser {
  public:

  ArcEagerParser();

  ArcEagerParser(Words sent);

  ArcEagerParser(Words sent, Words tags);

  ArcEagerParser(Words sent, Words tags, int num_particles);

  ArcEagerParser(const ParsedSentence& parse);

  bool shift();
  
  bool shift(WordId w);

  bool leftArc();

  bool rightArc();
  
  bool rightArc(WordId w);
  
  bool reduce();

  kAction oracleNext(const ParsedSentence& gold_parse) const;

  bool left_arc_valid() const {
    //stack_size 1 -> stack top is root
    if (stack_depth() < 2)
      return false;    
    WordIndex i = stack_top();
    return (!has_parent_at(i));
  }

  bool reduce_valid() const {
    WordIndex i = stack_top();
    //if STOP, should not have parent, else it should
    if (tag_at(i) == 1)
      return !has_parent_at(i);
    else
      return has_parent_at(i);
  }

  bool is_terminal_configuration() const {
    //last word generated is STOP
    return (!stack_empty() && (tag_at(stack_top()) == 1)); 
    
    // && !buffer_next_has_child());
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
    return tag_next_children_word_distance_context(); //lexicalized, smaller context (order 8)
    //return tag_next_children_distance_some_context(); //smaller context
    //return tag_next_children_distance_context(); //full
    //return tag_next_children_word_context(); //lexicalized, full context (?)
  }

  Words tag_context() const {
    return tag_next_children_some_context(); //smaller context (order 6)
  }
 
  //TODO is there a better way to do this?
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

  //TODO update where this is used
  /*bool is_complete_parse() const {
    for (WordIndex i = 1; i < arcs_.size() - 1; ++i) {
      if (!arcs_.has_parent(i) && (tags_.at(i)!=1))
        return false;
    }

    return ((buffer_next_ >= 3) && !buffer_next_has_child());
  } */

};

}

#endif
