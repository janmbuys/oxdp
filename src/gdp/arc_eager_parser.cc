#include "arc_eager_parser.h"

namespace oxlm {

ArcEagerParser::ArcEagerParser(const boost::shared_ptr<ModelConfig>& config): 
  TransitionParser(config)
{
}

ArcEagerParser::ArcEagerParser(const TaggedSentence& parse, const boost::shared_ptr<ModelConfig>& config):
  TransitionParser(parse, config)
{
}   

ArcEagerParser::ArcEagerParser(const TaggedSentence& parse, int num_particles, const boost::shared_ptr<ModelConfig>& config):
  TransitionParser(parse, num_particles, config) 
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

kAction ArcEagerParser::oracleNext(const ParsedSentence& gold_parse) const {
  kAction a = kAction::sh;
  if (stack_empty()) 
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
      //reduce if i has all its children
      a = kAction::re;
      for (WordIndex k = 1; k < size(); ++k) {
        if (gold_parse.has_arc(k, i) && !has_arc(k, i)) {
          a = kAction::sh;
          break;
        }
      }
  
      //alternative sh/re oracle: test if we have to reduce, else shift
      /* for (WordIndex k = i - 1; ((k >= 0) && (a==kAction::sh)); --k) {
       * if (gold_arcs.has_arc(k, j) || gold_arcs.has_arc(j, k)) 
       *   a = kAction::re; //need to reduce i to be able to add the arc
      } */
    }
  } 

  return a;
}

bool ArcEagerParser::inTerminalConfiguration() const {
  if (root_first()) 
    return buffer_empty(); //can have an incomplete parse
  else
    return (buffer_empty() && (stack_depth() == 1));
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
  if (pyp_model())
    return word_tag_next_children_context(); //order 7
  else  
    return word_next_children_context(); //order 7
}

Words ArcEagerParser::tagContext() const {
  return tag_next_children_context(); //order 7
}
 
//TODO remove when right-arc is redefined
Words ArcEagerParser::tagContext(kAction a) const {
  Words ctx = tag_next_children_context(); // order 7+1
  ctx.push_back(ctx.back());
  if (a == kAction::ra)
    ctx.at(ctx.size()-2) = 1;
  else
    ctx.at(ctx.size()-2) = 0;
  return ctx;
}

Words ArcEagerParser::actionContext() const {
  if (pyp_model()) {
    if (lexicalised())
      return tag_next_children_word_context(); //order 8
    else
      return tag_next_children_context(); //order 7
  } else {
    return word_next_children_context(); //order 7
  }
}

void ArcEagerParser::extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const {
  ArcEagerParser parser(static_cast<TaggedSentence>(*this), config()); 

  for (kAction& a: actions()) {
    if (a == kAction::sh || a == kAction::ra) {
      //tag prediction
      examples->add_tag_example(DataPoint(parser.next_tag(), parser.tagContext(a)));  
       
      //word prediction
      examples->add_word_example(DataPoint(parser.next_word(), parser.wordContext()));  
    }  

    //action prediction
    examples->add_action_example(DataPoint(static_cast<WordId>(a), parser.actionContext()));
    parser.executeAction(a);
  }
} 

}
