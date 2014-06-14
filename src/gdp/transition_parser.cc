#include "transition_parser.h"

namespace oxlm {

//TODO
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


bool TransitionParser::shift() {
  WordIndex i = buffer_.back();
  buffer_.pop_back();
  stack_.push_back(i);
  actions_.push_back(kAction::sh);
  return true;
}

bool TransitionParser::shift(WordId w) {
  WordIndex i = sentence_.size();
  sentence_.push_back(w);
  arcs_.push_back();
  if (buffer_.size() > 0) 
    buffer_.pop_back();  
  stack_.push_back(i);
  actions_.push_back(kAction::sh);
  return true;
}

bool TransitionParser::buffer_tag(WordId t) {
  buffer_.push_back(tags_.size());
  tags_.push_back(t);
  return true;
}

bool ArcStandardParser::leftArc() {
  WordIndex j = stack_.back();
  stack_.pop_back();
  WordIndex i = stack_.back();
  //check to ensure 0 is root
  if (i==0) {
    stack_.push_back(j);
    return false;
  }

  stack_.pop_back();
  stack_.push_back(j);
  arcs_.set_arc(i, j);
  actions_.push_back(kAction::la);
  return true;
}

bool ArcStandardParser::rightArc() {
  WordIndex j = stack_.back();
  stack_.pop_back();
  WordIndex i = stack_.back();
  arcs_.set_arc(j, i);
  actions_.push_back(kAction::ra);
  return true;
}

  //predict the next action according to the oracle, modified for evaluation and error analysis
  //can maybe formulate in terms of loss of each action
kAction ArcStandardParser::oracleDynamicNext(const ArcList& gold_arcs) const { //, ArcList prop_arcs) {
  kAction a = kAction::re;
    
  //assume not in terminal configuration 
  if (stack_depth() < 2)
    a = kAction::sh; 
  else {
    WordIndex i = stack_.rbegin()[1];
    WordIndex j = stack_.rbegin()[0];
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

//predict the next action according to the oracle
kAction ArcStandardParser::oracleNext(const ArcList& gold_arcs) const {
  kAction a = kAction::re;

  //assume not in terminal configuration 
  if (stack_depth() < 2)
    a = kAction::sh; 
  else {
    WordIndex i = stack_.rbegin()[1];
    WordIndex j = stack_.rbegin()[0];
    if (gold_arcs.has_arc(i, j) && child_count_at(i)==gold_arcs.child_count_at(i)) 
      a = kAction::la;
    else if (gold_arcs.has_arc(j, i) && child_count_at(j)==gold_arcs.child_count_at(j)) 
      a = kAction::ra;
    else if (!is_buffer_empty()) 
      a = kAction::sh;
  }
    
  return a;
}

}
