#include "arc_eager_labelled_parser.h"

namespace oxlm {

ArcEagerLabelledParser::ArcEagerLabelledParser(int num_labels): 
  TransitionParser(),
  num_labels_(num_labels),
  action_labels_()
{
}

ArcEagerLabelledParser::ArcEagerLabelledParser(Words sent, int num_labels): 
  TransitionParser(sent),
  num_labels_(num_labels),
  action_labels_()
{
}

ArcEagerLabelledParser::ArcEagerLabelledParser(Words sent, Words tags, int num_labels): 
  TransitionParser(sent, tags),
  num_labels_(num_labels),
  action_labels_()
{
}

ArcEagerLabelledParser::ArcEagerLabelledParser(Words sent, Words tags, int num_particles, int num_labels):
  TransitionParser(sent, tags, num_particles),
  num_labels_(num_labels),
  action_labels_()
{
}

ArcEagerLabelledParser::ArcEagerLabelledParser(const TaggedSentence& parse, int num_labels):
  TransitionParser(parse),
  num_labels_(num_labels),
  action_labels_()
{
}   

ArcEagerLabelledParser::ArcEagerLabelledParser(const TaggedSentence& parse, int num_particles, int num_labels):
  TransitionParser(parse, num_particles), 
  num_labels_(num_labels),
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
  //take (first) left-most and closest left-child
  //if ((buffer_left_child_ > -1) && (buffer_left_most_child_ == -1))
  //  buffer_left_most_child_ = buffer_left_child_;
  //buffer_left_child_ = i;
  return true;
}

bool ArcEagerLabelledParser::rightArc(WordId l) {
  //add right arc and shift
  WordIndex i = stack_top();
  WordIndex j = buffer_next();
  set_arc(j, i);
  set_label(j, l);
  pop_buffer();
  push_stack(j);
  append_action(kAction::ra);
  append_action_label(l);
  return true;
}

bool ArcEagerLabelledParser::rightArc(WordId l, WordId w) {
  //add right arc and shift
  WordIndex i = stack_top();
  //assume the tag has been generated already
  WordIndex j = size() - 1;
  push_word(w);
  push_arc();
  
  set_arc(j, i);
  set_label(j, l);
  if (!buffer_empty()) 
    pop_buffer();
  push_stack(j);
  append_action(kAction::ra);
  append_action_label(l);
  return true;
}

//Give the label for reduce action, if at all valid 
WordId ArcEagerLabelledParser::oracleNextLabel(const ParsedSentence& gold_parse) const {
  WordId lab = -1;

  //assume not in terminal configuration 
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

bool ArcEagerLabelledParser::inTerminalConfiguration() const {
  //last word generated is STOP
  //return (!stack_empty() && (tag_at(stack_top()) == 1)); 
    
  // && !buffer_next_has_child());
  //return (!is_stack_empty() && (stack_top() == static_cast<int>(sentence_length() - 1))); // && !buffer_next_has_child());

  //return ((tag_at(stack_top()) == 1)); // && !buffer_next_has_child());
  //if (is_generating()) 
  //  return ((buffer_next() >= 3) && (stack_depth() == 1)); 
  //else     
  return (buffer_empty() && (stack_depth() == 1));
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

Words ArcEagerLabelledParser::wordContext() const {
  //return word_children_distance_context(); //lbl model (order 8)
  return word_tag_next_children_context();  //(order 6)
  //return word_tag_next_context();
}

Words ArcEagerLabelledParser::tagContext() const {
  return tag_next_children_some_context(); //smaller context (order 6)
}
 
//problem is we can't append the action before it has been executed
Words ArcEagerLabelledParser::tagContext(kAction a) const {
  //Words ctx = tag_next_children_some_context(); //smaller context (order 6)
  Words ctx = tag_next_children_context(); //full context (order 8)
  ctx.push_back(ctx.back());
  if (a == kAction::ra)
    ctx.at(ctx.size()-2) = 1;
  else
    ctx.at(ctx.size()-2) = 0;
  return ctx;
}

Words ArcEagerLabelledParser::actionContext() const {
  //return word_children_distance_context(); //lbl model (order 8)
  //return tag_next_children_distance_some_context(); //smaller context (order 5)
  //return tag_next_children_distance_context(); //full (order 8)
  return tag_next_children_word_context(); //lexicalized, full context (order 8)
}

void ArcEagerLabelledParser::extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const {
  ArcEagerLabelledParser parser(static_cast<TaggedSentence>(*this), num_labels_); 

  for (unsigned i = 0; i < actions().size(); ++i) {
    kAction a = actions().at(i);
    WordId lab = action_label_at(i);

    if (a == kAction::sh || a == kAction::ra) {
      //tag prediction
      examples->add_tag_example(DataPoint(parser.next_tag(), parser.tagContext(a)));  
       
      //word prediction
      //if (!(word_examples == nullptr))  //do we want to do this?
      examples->add_word_example(DataPoint(parser.next_word(), parser.wordContext()));  
    }  

    //labelled action prediction 
    WordId lab_act = convert_action(a, lab);
    examples->add_action_example(DataPoint(lab_act, parser.actionContext()));
    parser.executeAction(a, lab);
  }
} 

}
