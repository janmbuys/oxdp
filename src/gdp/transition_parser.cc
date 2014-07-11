#include "transition_parser.h"

namespace oxlm {

void AccuracyCounts::countAccuracy(const ArcStandardParser& prop_parse, const ArcList& gold_arcs) {
  //resimulate the computation of the proposed action sequence to compute accuracy
  ArcStandardParser simul(prop_parse.sentence(), prop_parse.tags());
        
  for (auto& a: prop_parse.actions()) {
    kAction next = simul.oracleDynamicNext(gold_arcs);
    //count when shifted/reduced when it should have shifted/reduced
    if (next==kAction::sh) {
      inc_shift_gold();
      if (a==kAction::sh)
        inc_shift_count();
    } else if (next==kAction::la || next==kAction::ra) {
      inc_reduce_gold();
      if (a==kAction::la || a==kAction::ra) //counts either direction
        inc_reduce_count();
    } /*  else if (next==kAction::la2 || next==kAction::ra2) {
          //for reduce: if there exists an arc it should be added...
          //Unless we know that it is added later in the given parse
        //action taken is considered the gold action
        if (a==kAction::sh) {
          inc_shift_count();
          inc_shift_gold();
        } else {
          inc_reduce_count();
          inc_reduce_gold();
        }
      } */

    if (simul.is_buffer_empty() && next==kAction::re)
      inc_final_reduce_error_count();
    
    simul.execute_action(a);
  }

  inc_num_sentences();
  if (gold_arcs==prop_parse.arcs())
    inc_complete_sentences();
  for (unsigned i = 1; i < gold_arcs.size(); ++i)
    if (prop_parse.arcs().has_arc(i, 0) && gold_arcs.has_arc(i, 0)) 
      inc_root_count();

  add_total_length(gold_arcs.size() - 1);
  add_directed_count(prop_parse.directed_accuracy_count(gold_arcs));
  add_undirected_count(prop_parse.undirected_accuracy_count(gold_arcs));
}

void AccuracyCounts::countAccuracy(const ArcEagerParser& prop_parse, const ArcList& gold_arcs) {
  //resimulate the computation of the proposed action sequence to compute accuracy
  ArcEagerParser simul(prop_parse.sentence(), prop_parse.tags());
    
  for (auto& a: prop_parse.actions()) {
    kAction next = simul.oracleNext(gold_arcs);
    
    //include more sophisticated statistics later
    //count when shifted/reduced when it should have shifted/reduced
    if (next==kAction::sh || next==kAction::ra) {
      inc_shift_gold();
      if (a==kAction::sh || a==kAction::ra)
        inc_shift_count();
    } else if (next==kAction::la || next==kAction::re) {
      inc_reduce_gold();
      if (a==kAction::la || a==kAction::re) 
        inc_reduce_count();
    } 
    
    simul.execute_action(a);
  }
            
  inc_num_sentences();
  if (gold_arcs==prop_parse.arcs())
    inc_complete_sentences();
  for (unsigned i = 1; i < gold_arcs.size(); ++i)
    if (prop_parse.arcs().has_arc(i, 0) && gold_arcs.has_arc(i, 0)) 
      inc_root_count();

  add_total_length(gold_arcs.size() - 1);
  add_directed_count(prop_parse.directed_accuracy_count(gold_arcs));
  add_undirected_count(prop_parse.undirected_accuracy_count(gold_arcs));
}

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

bool ArcEagerParser::shift() {
  WordIndex i = buffer_next();
  pop_buffer();
  buffer_left_most_child_ = -1;
  buffer_left_child_ = -1;
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
  if ((buffer_left_child_ > -1) && (buffer_left_most_child_ == -1))
    buffer_left_most_child_ = buffer_left_child_;
  buffer_left_child_ = i;
  return true;
}

bool ArcEagerParser::rightArc() {
  //add right arc and shift
  WordIndex i = stack_top();
  WordIndex j = buffer_next();
  set_arc(j, i);
  pop_buffer();
  buffer_left_most_child_ = -1;
  buffer_left_child_ = -1;
  push_stack(j);
  append_action(kAction::ra);
  return true;
}

//predict the next action according to the oracle
kAction ArcEagerParser::oracleNext(const ArcList& gold_arcs) const {
  kAction a = kAction::sh;

  //maybe change so that we can assume stack_depth > 0 
  if (is_stack_empty())
    return a;

  WordIndex i = stack_top();
  //if la or ra is valid
  if (!is_buffer_empty()) {    
    WordIndex j = buffer_next();
    if (gold_arcs.has_arc(i, j)) {
      //add left arc eagerly
      a = kAction::la; 
    } else if (gold_arcs.has_arc(j, i)) {
      //add right arc eagerly
      a = kAction::ra; 
    } 
  }

  //test if we may reduce, else shift
  if (has_parent(i) && (child_count_at(i) >= gold_arcs.child_count_at(i))) {
      //reduce if it has a parent and all its children
      //stronger, prefer when we want to achieve no spurious ambiguity
      a = kAction::re;
  }
    
  return a;
}

}
