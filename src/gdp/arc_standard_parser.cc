#include "arc_standard_parser.h"

namespace oxlm {

ArcStandardParser::ArcStandardParser():
  TransitionParser() 
{
}

ArcStandardParser::ArcStandardParser(Words sent):
  TransitionParser(sent) 
{
}

ArcStandardParser::ArcStandardParser(Words sent, Words tags):
  TransitionParser(sent, tags) 
{
}

ArcStandardParser::ArcStandardParser(Words sent, Words tags, int num_particles):
  TransitionParser(sent, tags, num_particles) 
{
}

ArcStandardParser::ArcStandardParser(const ParsedSentence& parse):
  TransitionParser(parse) 
{
}

bool ArcStandardParser::shift() {
  WordIndex i = buffer_next();
  pop_buffer();
  push_stack(i);
  append_action(kAction::sh);
  return true;
}

bool ArcStandardParser::shift(WordId w) {
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
kAction ArcStandardParser::oracleNext(const ParsedSentence& gold_parse) const {
  kAction a = kAction::re;

  //assume not in terminal configuration 
  if (stack_depth() < 2)
    a = kAction::sh; 
  else {
    WordIndex i = stack_top_second();
    WordIndex j = stack_top();
    if (gold_parse.has_arc(i, j)) {
      a = kAction::la;
      //check that i has all its children
      for (WordIndex k = 1; k < size(); ++k) {
        if (gold_parse.has_arc(k, i) && !has_arc(k, i)) {
          a = kAction::re;
          break;
        }
      }
    }
    else if (gold_parse.has_arc(j, i)) {
      a = kAction::ra;
      //check that j has all its children
      for (WordIndex k = 1; k < size(); ++k) {
        if (gold_parse.has_arc(k, j) && !has_arc(k, j)) {
          a = kAction::re;
          break;
        }
      }
    }
    if ((a == kAction::re) && !buffer_empty()) 
      a = kAction::sh;
  }
    
  return a;
}

}
