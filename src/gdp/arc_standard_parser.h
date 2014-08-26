#ifndef _GDP_AS_PARSER_H
#define _GDP_AS_PARSER_H

#include "transition_parser.h"

namespace oxlm {

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

}

#endif
