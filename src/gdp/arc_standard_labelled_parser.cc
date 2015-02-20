#include "arc_standard_labelled_parser.h"

namespace oxlm {

ArcStandardLabelledParser::ArcStandardLabelledParser(const boost::shared_ptr<ModelConfig>& config):
  TransitionParser(config),
  action_labels_()
{
}

ArcStandardLabelledParser::ArcStandardLabelledParser(const TaggedSentence& parse, const boost::shared_ptr<ModelConfig>& config):
  TransitionParser(parse, config),
  action_labels_()
{
}

ArcStandardLabelledParser::ArcStandardLabelledParser(const TaggedSentence& parse, int num_particles, const boost::shared_ptr<ModelConfig>& config):
  TransitionParser(parse, num_particles, config),
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

Context ArcStandardLabelledParser::wordContext() const {
  if (pyp_model())
    return Context(word_tag_next_children_context()); //order 7
    //return word_tag_next_ngram_context(); // best perplexity
  else {
    if (context_type() == "extended")
      return map_context(extended_children_context()); //order 13
    else if (context_type() == "more-extended")
      return map_context(more_extended_children_context()); //order 17
    else if (context_type() == "with-ngram")
      return map_context(children_ngram_context()); //order 13
    else if (context_type() == "lookahead")
      return map_context(children_lookahead_context()); //order 10
    else
      return map_context(children_context()); //order 7
  }
}

Context ArcStandardLabelledParser::tagContext() const {
  return Context(tag_children_context());  //order 8
}

Context ArcStandardLabelledParser::actionContext() const {
  if (pyp_model()) {
    if (lexicalised())
      return Context(word_tag_children_context()); //order 10
    else
      return Context(tag_children_context()); //order 8
  } else {
    if (context_type() == "extended")
      return map_context(extended_children_context()); //order 13
    else if (context_type() == "more-extended")
      return map_context(more_extended_children_context()); //order 17
    else if (context_type() == "with-ngram")
      return map_context(children_ngram_context()); //order 13
    else if (context_type() == "lookahead")
      return map_context(children_lookahead_context()); //order 10
    else
      return map_context(children_context()); //order 7
  }
}

void ArcStandardLabelledParser::extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(*this), config()); 
 
  //note that we are extracting the initial shift as an example
  for (unsigned i = 0; i < actions().size(); ++i) {
    kAction a = actions().at(i);
    WordId lab = action_label_at(i);

    if (a == kAction::sh) {
      //tag prediction
      examples->add_tag_example(DataPoint(parser.next_tag(), parser.tagContext()));  
       
      //word prediction
      examples->add_word_example(DataPoint(parser.next_word(), parser.wordContext()));  
    } 

    //labelled action prediction 
    WordId lab_act = convert_action(a, lab);
    examples->add_action_example(DataPoint(lab_act, parser.actionContext()));
    parser.executeAction(a, lab);
  }
} 

}
