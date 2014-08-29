#include "arc_eager_parser.h"

namespace oxlm {

ArcEagerParser::ArcEagerParser(): 
  TransitionParser()
{
}

ArcEagerParser::ArcEagerParser(Words sent): 
  TransitionParser(sent)
{
}

ArcEagerParser::ArcEagerParser(Words sent, Words tags): 
  TransitionParser(sent, tags)
{
}

ArcEagerParser::ArcEagerParser(Words sent, Words tags, int num_particles):
  TransitionParser(sent, tags, num_particles) 
{
}

ArcEagerParser::ArcEagerParser(const ParsedSentence& parse):
  TransitionParser(parse)
{
}   

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
kAction ArcEagerParser::oracleNext(const ParsedSentence& gold_parse) const {
  kAction a = kAction::sh;

  //maybe change so that we can assume stack_depth > 0 
  //force generation of stop asap in training examples
  if (stack_empty()) //|| (buffer_next() < static_cast<int>(sentence_length())) && (tag_at(buffer_next())==1)))
    return a;

  WordIndex i = stack_top();
  //if la or ra is valid
  if (!buffer_empty()) {    
    WordIndex j = buffer_next();
    if (gold_parse.has_arc(i, j)) {
      //add left arc eagerly
      a = kAction::la; 
    } else if (gold_parse.has_arc(j, i)) {
      //add right arc eagerly
      a = kAction::ra; 
    } else if (reduce_valid()) {  
      //if (child_count_at(i) >= gold_arcs.child_count_at(i)) 
      //  a = kAction::re;
      //reduce if i has its children
      a = kAction::re;
      for (WordIndex k = 1; k < size(); ++k) {
        if (gold_parse.has_arc(k, i) && !has_arc(k, i)) {
          a = kAction::sh;
          break;
        }
      }
  
      //alternatively, test if we should, else shift
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

bool ArcEagerParser::isTerminalConfiguration() const {
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

bool ArcEagerParser::executeAction(kAction a) {
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

Words ArcEagerParser::wordContext() const {
  return word_tag_next_children_context();  //(order 6)
  //return word_tag_next_context();
}

Words ArcEagerParser::tagContext() const {
  return tag_next_children_some_context(); //smaller context (order 6)
}
 
//problem is we can't append the action before it has been executed
Words ArcEagerParser::tagContext(kAction a) const {
  Words ctx = tag_next_children_some_context(); //smaller context (order 6)
  //Words ctx = tag_next_children_context(); //full context
  ctx.push_back(ctx.back());
  if (a == kAction::ra)
    ctx.at(ctx.size()-2) = 1;
  else
    ctx.at(ctx.size()-2) = 0;
  return ctx;
}

Words ArcEagerParser::actionContext() const {
  return tag_next_children_word_distance_context(); //lexicalized, smaller context (order 8)
  //return tag_next_children_distance_some_context(); //smaller context
  //return tag_next_children_distance_context(); //full
  //return tag_next_children_word_context(); //lexicalized, full context (?)
}

}
