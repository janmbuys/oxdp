#include "arc_eager_parser.h"

namespace oxlm {

bool ArcEagerParser::shift() {
  WordIndex i = buffer_next();
  pop_buffer();
  push_stack(i);
  append_action(kAction::sh);
  return true;
}

bool ArcEagerParser::shift(WordId w) {
  //assume the tag has been generated already
  WordIndex i = size() - 1;
  push_word(w);
  push_arc();
  if (!buffer_empty()) 
    pop_buffer();
  push_stack(i);
  append_action(kAction::sh);
  return true;
}

bool ArcEagerParser::reduce() {
  if (!reduce_valid())
    return false;
  pop_stack();
  append_action(kAction::re);
  return true;
} 

bool ArcEagerParser::leftArc() {
  if (!left_arc_valid())
    return false;
    
  //add left arc and reduce
  WordIndex i = stack_top();
  WordIndex j = buffer_next();
  set_arc(i, j);
  pop_stack();
  append_action(kAction::la);
  //take (first) left-most and closest left-child
  //if ((buffer_left_child_ > -1) && (buffer_left_most_child_ == -1))
  //  buffer_left_most_child_ = buffer_left_child_;
  //buffer_left_child_ = i;
  return true;
}

bool ArcEagerParser::rightArc() {
  //add right arc and shift
  WordIndex i = stack_top();
  WordIndex j = buffer_next();
  set_arc(j, i);
  pop_buffer();
  push_stack(j);
  append_action(kAction::ra);
  return true;
}

bool ArcEagerParser::rightArc(WordId w) {
  //add right arc and shift
  WordIndex i = stack_top();
  //assume the tag has been generated already
  WordIndex j = size() - 1;
  push_word(w);
  push_arc();
  
  set_arc(j, i);
  if (!buffer_empty()) 
    pop_buffer();
  push_stack(j);
  append_action(kAction::ra);
  return true;
}

//predict the next action according to the oracle
kAction ArcEagerParser::oracleNext(const ArcList& gold_arcs) const {
  kAction a = kAction::sh;

  //maybe change so that we can assume stack_depth > 0 
  //force generation of stop asap in training examples
  if (stack_empty()) //|| (buffer_next() < static_cast<int>(sentence_length())) && (tag_at(buffer_next())==1)))
    return a;

  WordIndex i = stack_top();
  //if la or ra is valid
  if (!buffer_empty()) {    
    WordIndex j = buffer_next();
    if (gold_arcs.has_arc(i, j)) {
      //add left arc eagerly
      a = kAction::la; 
    } else if (gold_arcs.has_arc(j, i)) {
      //add right arc eagerly
      a = kAction::ra; 
    } else if (reduce_valid()) {  
      //reduce if it has a parent and all its children
      if (child_count_at(i) >= gold_arcs.child_count_at(i)) 
        a = kAction::re;
  
      //test if we should, else shift
      /*for (WordIndex k = i - 1; ((k >= 0) && (a==kAction::sh)); --k) {
        //std::cout << k << " ";
        //if we need to reduce i to be able to add the arc
        if (gold_arcs.has_arc(k, j) || gold_arcs.has_arc(j, k)) 
          a = kAction::re;
      } */
    }
  } 

  return a;
}

}
