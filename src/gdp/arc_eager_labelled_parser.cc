#include "arc_eager_labelled_parser.h"

namespace oxlm {

ArcEagerLabelledParser::ArcEagerLabelledParser(const boost::shared_ptr<ModelConfig>& config): 
  TransitionParser(config),
  action_labels_()
{
}

ArcEagerLabelledParser::ArcEagerLabelledParser(const TaggedSentence& parse, const boost::shared_ptr<ModelConfig>& config):
  TransitionParser(parse, config),
  action_labels_()
{
}   

ArcEagerLabelledParser::ArcEagerLabelledParser(const TaggedSentence& parse, int num_particles, const boost::shared_ptr<ModelConfig>& config):
  TransitionParser(parse, num_particles, config), 
  action_labels_()
{
}

bool ArcEagerLabelledParser::shift() {
  WordIndex i = buffer_next();
  pop_buffer();
  push_stack(i);
  append_action(kAction::sh);
  append_action_label(-1);
  return true;
}

bool ArcEagerLabelledParser::shift(WordId w) {
  //assume the tag has been generated already
  WordIndex i = size() - 1;
  push_word(w);
  push_arc();
  if (!buffer_empty()) 
    pop_buffer();
  push_stack(i);
  append_action(kAction::sh);
  append_action_label(-1);
  return true;
}

bool ArcEagerLabelledParser::reduce() {
  if (!reduce_valid())
    return false;
  pop_stack();
  append_action(kAction::re);
  append_action_label(-1);
  return true;
} 

bool ArcEagerLabelledParser::leftArc(WordId l) {
  if (!left_arc_valid())
    return false;
    
  //add left arc and reduce
  WordIndex i = stack_top();
  WordIndex j = buffer_next();
  set_arc(i, j);
  set_label(i, l);
  pop_stack();
  append_action(kAction::la);
  append_action_label(l);
  return true;
}

bool ArcEagerLabelledParser::rightArc(WordId l) {
  WordIndex i = stack_top();
  WordIndex j = buffer_next();
  set_arc(j, i);
  set_label(j, l);
  //don't shift
  //pop_buffer();
  //push_stack(j);
  append_action(kAction::ra);
  append_action_label(l);
  return true;
}

/*
bool ArcEagerLabelledParser::rightArc(WordId l, WordId w) {
  //add right arc and shift
  WordIndex i = stack_top();
  //assume the tag has been generated already
  WordIndex j = size() - 1;
  push_word(w);
  push_arc();
  
  set_arc(j, i);
  set_label(j, l);
  //don't shift
  //if (!buffer_empty()) 
  // pop_buffer();
  //push_stack(j);
  append_action(kAction::ra);
  append_action_label(l);
  return true;
} */

//Give the label for reduce action, if at all valid 
WordId ArcEagerLabelledParser::oracleNextLabel(const ParsedSentence& gold_parse) const {
  WordId lab = -1;

  if (!stack_empty()) {
    WordIndex i = stack_top();
    if (!buffer_empty()) {    
      WordIndex j = buffer_next();
      if (gold_parse.has_arc(i, j)) {
        lab = gold_parse.label_at(i);
      } else if (gold_parse.has_arc(j, i)) {
        lab = gold_parse.label_at(j);
      } 
    }
  }
    
  return lab;
}

//predict the next action according to the oracle
kAction ArcEagerLabelledParser::oracleNext(const ParsedSentence& gold_parse) const {
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
      //reduce if i has its children
      a = kAction::re;
      for (WordIndex k = 1; k < size(); ++k) {
        if (gold_parse.has_arc(k, i) && !has_arc(k, i)) {
          a = kAction::sh;
          break;
        }
      }
  
      //alternatively, test if we should reduce, else shift
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

bool ArcEagerLabelledParser::inTerminalConfiguration() const {
  if (root_first()) 
    return buffer_empty(); //can have an incomplete parse, don't enforce final reduces
  else
    return (buffer_empty() && (stack_depth() == 1)); //parser should do reduces before last shift
}

bool ArcEagerLabelledParser::executeAction(kAction a, WordId l) {
  switch(a) {
  case kAction::sh:
    return shift();
  case kAction::la:
    return leftArc(l);
  case kAction::ra:
    return rightArc(l);
  case kAction::re:
    return reduce();
  default: 
    std::cerr << "action not implemented" << std::endl;
    return false;
  }
}

Context ArcEagerLabelledParser::wordContext() const {
  if (pyp_model())
    return Context(word_tag_next_children_context()); //order 9
  else {
    if (context_type() == "extended") 
      return map_context(extended_next_children_context()); //order 14
    else if (context_type() == "lookahead")
      return map_context(next_children_lookahead_context()); //order 12
    else
      return map_context(next_children_context()); //order 8
  }
}

Context ArcEagerLabelledParser::tagContext() const {
  return Context(tag_next_children_context()); //order 7
}
 
Context ArcEagerLabelledParser::actionContext() const {
  if (pyp_model()) {
    if (lexicalised())
      return Context(tag_next_children_word_context()); //order 8
    else
      return Context(tag_next_children_context()); //order 7
  } else {
   if (context_type() == "extended") 
      return map_context(extended_next_children_context()); //order 14
    else if (context_type() == "lookahead")
      return map_context(next_children_lookahead_context()); //order 12
    else
      return map_context(next_children_context()); //order 8
  }
}

void ArcEagerLabelledParser::extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const {
  ArcEagerLabelledParser parser(static_cast<TaggedSentence>(*this), config()); 

  for (unsigned i = 0; i < actions().size(); ++i) {
    kAction a = actions().at(i);
    WordId lab = action_label_at(i);

    if (a == kAction::sh) {   // || a == kAction::ra) {
      //tag prediction
      examples->add_tag_example(DataPoint(parser.next_tag(), parser.tagContext()));  
       
      //word prediction
      examples->add_word_example(DataPoint(parser.next_word(), parser.wordContext()));  
      //std::cout << parser.next_word() << ": ";
      //for (auto w: parser.wordContext().words)
      //  std::cout << w << " ";
      //std::cout << std::endl;
    }  

    //labelled action prediction 
    WordId lab_act = convert_action(a, lab);
    examples->add_action_example(DataPoint(lab_act, parser.actionContext()));
    parser.executeAction(a, lab);
  }
} 

}
