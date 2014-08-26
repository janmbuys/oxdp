#include "arc_standard_parser.h"

namespace oxlm {

bool ArcStandardParser::shift() {
  WordIndex i = buffer_next();
  pop_buffer();
  push_stack(i);
  append_action(kAction::sh);
  return true;
}

bool ArcStandardParser::shift(WordId w) {
  WordIndex i = sentence_length();
  push_word(w);
  push_arc();
  if (!is_buffer_empty()) 
    pop_buffer();
  push_stack(i);
  append_action(kAction::sh);
  return true;
}

bool ArcStandardParser::leftArc() {
  if (!left_arc_valid())
    return false;
    
  WordIndex j = stack_top();
  pop_stack();
  WordIndex i = stack_top();
  pop_stack();
  push_stack(j);
  set_arc(i, j);
  append_action(kAction::la);
  return true;
}

bool ArcStandardParser::rightArc() {
  WordIndex j = stack_top();
  pop_stack();
  WordIndex i = stack_top();
  set_arc(j, i);
  append_action(kAction::ra);
  return true;
}
 
//predict the next action according to the oracle
kAction ArcStandardParser::oracleNext(const ArcList& gold_arcs) const {
  kAction a = kAction::re;

  //assume not in terminal configuration 
  if (stack_depth() < 2)
    a = kAction::sh; 
  else {
    WordIndex i = stack_top_second();
    WordIndex j = stack_top();
    if (gold_arcs.has_arc(i, j) && child_count_at(i)==gold_arcs.child_count_at(i)) 
      a = kAction::la;
    else if (gold_arcs.has_arc(j, i) && child_count_at(j)==gold_arcs.child_count_at(j)) 
      a = kAction::ra;
    else if (!is_buffer_empty()) 
      a = kAction::sh;
  }
    
  return a;
}

//predict the next action according to the oracle, modified for evaluation and error analysis
//can maybe formulate in terms of loss of each action
kAction ArcStandardParser::oracleDynamicNext(const ArcList& gold_arcs) const { //, ArcList prop_arcs) {
  kAction a = kAction::re;
            
  //assume not in terminal configuration 
  if (stack_depth() < 2)
    a = kAction::sh; 
  else {
    WordIndex i = stack_top_second();
    WordIndex j = stack_top();
    if (gold_arcs.has_arc(i, j) && (child_count_at(i) >= gold_arcs.child_count_at(i))) {
      a = kAction::la; 
      //if (prop_arcs.has_arc(i, j)) 
      //  a = kAction::la2;
    } else if (gold_arcs.has_arc(j, i) && (child_count_at(j) >= gold_arcs.child_count_at(j))) { 
      a = kAction::ra; 
      //if (prop_arcs.has_arc(j, i)) 
      //  a = kAction::ra2;
    } else if (!is_buffer_empty()) 
      a = kAction::sh;
  }
   
  //return re if there shouldn't be an arc, but can't do anthing else either
  return a;
}

}
