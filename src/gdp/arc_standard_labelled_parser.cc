#include "arc_standard_labelled_parser.h"

namespace oxlm {

ArcStandardLabelledParser::ArcStandardLabelledParser(int num_labels):
  TransitionParser(),
  num_labels_(num_labels),
  action_labels_()
{
}

ArcStandardLabelledParser::ArcStandardLabelledParser(Words sent, int num_labels):
  TransitionParser(sent),
  num_labels_(num_labels),
  action_labels_()
{
}

ArcStandardLabelledParser::ArcStandardLabelledParser(Words sent, Words tags, int num_labels):
  TransitionParser(sent, tags),
  num_labels_(num_labels),
  action_labels_()
{
}

ArcStandardLabelledParser::ArcStandardLabelledParser(Words sent, Words tags, int num_particles, int num_labels):
  TransitionParser(sent, tags, num_particles),
  num_labels_(num_labels),
  action_labels_()
{
}

ArcStandardLabelledParser::ArcStandardLabelledParser(const TaggedSentence& parse, int num_labels):
  TransitionParser(parse),
  num_labels_(num_labels),
  action_labels_()
{
}

ArcStandardLabelledParser::ArcStandardLabelledParser(const TaggedSentence& parse, int num_particles, int num_labels):
  TransitionParser(parse, num_particles),
  num_labels_(num_labels),
  action_labels_()
{
}

bool ArcStandardLabelledParser::shift() {
  WordIndex i = buffer_next();
  pop_buffer();
  push_stack(i);
  append_action(kAction::sh);
  append_action_label(-1);
  return true;
}

bool ArcStandardLabelledParser::shift(WordId w) {
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

bool ArcStandardLabelledParser::leftArc(WordId l) {
  if (!left_arc_valid())
    return false;
    
  WordIndex j = stack_top();
  pop_stack();
  WordIndex i = stack_top();
  pop_stack();
  push_stack(j);
  set_arc(i, j);
  set_label(i, l);
  append_action(kAction::la);
  append_action_label(l);
  return true;
}

bool ArcStandardLabelledParser::rightArc(WordId l) {
  WordIndex j = stack_top();
  pop_stack();
  WordIndex i = stack_top();
  set_arc(j, i);
  set_label(j, l);
  append_action(kAction::ra);
  append_action_label(l);
  return true;
}
 
//Give the label for reduce action, if at all valid 
WordId ArcStandardLabelledParser::oracleNextLabel(const ParsedSentence& gold_parse) const {
  WordId lab = -1;

  //assume not in terminal configuration 
  if (stack_depth() >= 2) {
    WordIndex i = stack_top_second();
    WordIndex j = stack_top();

    if (gold_parse.has_arc(i, j)) {
      lab = gold_parse.label_at(i);
    }
    else if (gold_parse.has_arc(j, i)) {
      lab = gold_parse.label_at(j);
    }
  }
    
  return lab;
}

//predict the next action according to the oracle
kAction ArcStandardLabelledParser::oracleNext(const ParsedSentence& gold_parse) const {
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

bool ArcStandardLabelledParser::inTerminalConfiguration() const {
  //if (is_generating()) return ((buffer_next() >= 3) && (stack_depth() == 1)); //&& !buffer_next_has_child());
  return (buffer_empty() && (stack_depth() == 1));
}

bool ArcStandardLabelledParser::executeAction(kAction a, WordId l) {
  switch(a) {
  case kAction::sh:
    return shift();
  case kAction::la:
    return leftArc(l);
  case kAction::ra:
    return rightArc(l);
  default: 
    std::cerr << "action not implemented" << std::endl;
    return false;
  }
} 

//TODO update contexts for labelled case

//(ideally would assert length of order)
Words ArcStandardLabelledParser::wordContext() const {
  //return word_children_distance_context(); //lbl model (order 8)
  return word_tag_next_children_context(); //best context (order 6)
  //return linear_word_tag_next_context(); //best perplexity
  //return word_tag_next_context(); 
}

Words ArcStandardLabelledParser::tagContext() const {
  return tag_children_context();  //best full context (order 9)
  //return linear_tag_context();
  //return tag_some_children_context(); //best smaller context (order 5)
}

Words ArcStandardLabelledParser::actionContext() const {
  //return word_children_distance_context(); //lbl model (order 8)
  //return word_tag_children_context(); //best full context, lexicalized (order 10)
  //return word_tag_some_children_distance_context(); //best smaller context, lexicalized (order 8)
  return tag_children_context(); //best full context (order 9)
  //return tag_some_children_distance_context(); //best smaller context (order 6)
}

void ArcStandardLabelledParser::extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(*this), num_labels_); 
 
  //note that we are extracting the initial shift as an example
  for (unsigned i = 0; i < actions().size(); ++i) {
    kAction a = actions().at(i);
    WordId lab = action_label_at(i);

    if (a == kAction::sh) {
      DataPoint point(parser.next_tag(), parser.tagContext());

      //tag prediction
      examples->add_tag_example(DataPoint(parser.next_tag(), parser.tagContext()));  
       
      //word prediction
      //if (!(word_examples == nullptr))  //do we want to do this?
      examples->add_word_example(DataPoint(parser.next_word(), parser.wordContext()));  
    } 

    //labelled action prediction 
    WordId lab_act = convert_action(a, lab);
    //std::cout << static_cast<WordId>(a) << "," << lab << "," << lab_act << " ";
    examples->add_action_example(DataPoint(lab_act, parser.actionContext()));
    parser.executeAction(a, lab);
  }

  //std::cout << std::endl;
} 

}
