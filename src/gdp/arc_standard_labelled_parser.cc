#include "arc_standard_labelled_parser.h"

namespace oxlm {

ArcStandardLabelledParser::ArcStandardLabelledParser(
    const boost::shared_ptr<ModelConfig>& config)
    : TransitionParser(config) {}

ArcStandardLabelledParser::ArcStandardLabelledParser(
    const TaggedSentence& parse, const boost::shared_ptr<ModelConfig>& config)
    : TransitionParser(parse, config) {}

ArcStandardLabelledParser::ArcStandardLabelledParser(
    const TaggedSentence& parse, int num_particles,
    const boost::shared_ptr<ModelConfig>& config)
    : TransitionParser(parse, num_particles, config) {}

bool ArcStandardLabelledParser::shift() {
  WordIndex i = buffer_next();
  pop_buffer();
  push_stack(i);
  append_action(kAction::sh);
  append_action_label(-1);
  return true;
}

bool ArcStandardLabelledParser::shift(WordId w) {
  WordIndex i = size() - 1;
  push_word(w);
  add_feature_at(i, get_word_feature(w));
  push_arc();
  if (!buffer_empty()) pop_buffer();
  push_stack(i);
  append_action(kAction::sh);
  append_action_label(-1);
  return true;
}

bool ArcStandardLabelledParser::leftArc(WordId l) {
  if (!left_arc_valid()) return false;

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

bool ArcStandardLabelledParser::leftArc2(WordId l) {
  WordIndex k = stack_top();
  pop_stack();
  WordIndex j = stack_top();
  pop_stack();
  WordIndex i = stack_top();
  pop_stack();
  push_stack(j);
  push_stack(k);
  set_arc(i, k);
  set_label(i, l);
  append_action(kAction::la2);
  append_action_label(l);
  return true;
}

bool ArcStandardLabelledParser::rightArc2(WordId l) {
  WordIndex k = stack_top();
  pop_stack();
  WordIndex j = stack_top();
  WordIndex i = stack_top_second();
  set_arc(k, i);
  set_label(k, l);
  append_action(kAction::ra2);
  append_action_label(l);
  return true;
}

WordId ArcStandardLabelledParser::oracleNextLabel(
    const ParsedSentence& gold_parse) const {
  WordId lab = -1;

  kAction a = oracleNext(gold_parse);

  // Assume not in a terminal configuration.
  if (a != kAction::sh && a != kAction::re) {
    WordIndex i = stack_top_second();
    WordIndex j = stack_top();

    if (a == kAction::la) {
      lab = gold_parse.label_at(i);
    } else if (a == kAction::ra) {
      lab = gold_parse.label_at(j);
    } else {
      WordIndex k = stack_top_third();
      if (a == kAction::la2) {
        lab = gold_parse.label_at(i);
      } else if (a == kAction::ra2) {
        lab = gold_parse.label_at(k);
      }
    }
  }

  return lab;
}

kAction ArcStandardLabelledParser::oracleNext(
    const ParsedSentence& gold_parse) const {
  kAction a = kAction::re;

  // Assume not in a terminal configuration.
  if ((stack_depth() < 2) && !buffer_empty()) {
    a = kAction::sh;
  } else {
    WordIndex i = stack_top_second();
    WordIndex j = stack_top();
    if (gold_parse.has_arc(i, j)) {
      a = kAction::la;
      // Ensure that i has all its children.
      for (WordIndex l = 1; l < size(); ++l) {
        if (gold_parse.has_arc(l, i) && !has_arc(l, i)) {
          a = kAction::re;
          break;
        }
      }
    } else if (gold_parse.has_arc(j, i)) {
      a = kAction::ra;
      // Ensure that j has all its children.
      for (WordIndex l = 1; l < size(); ++l) {
        if (gold_parse.has_arc(l, j) && !has_arc(l, j)) {
          a = kAction::re;
          break;
        }
      }
    }

    if ((a == kAction::re) && non_projective() && (stack_depth() >= 3)) {
      WordIndex k = stack_top_third();
      if (gold_parse.has_arc(k, j)) {
        a = kAction::la2;
        // Ensure that k has all its children.
        for (WordIndex l = 1; l < size(); ++l) {
          if (gold_parse.has_arc(l, k) && !has_arc(l, k)) {
            a = kAction::re;
            break;
          }
        }
      } else if (gold_parse.has_arc(j, k)) {
        a = kAction::ra2;
        // Ensure that j has all its children.
        for (WordIndex l = 1; l < size(); ++l) {
          if (gold_parse.has_arc(l, j) && !has_arc(l, j)) {
            a = kAction::re;
            break;
          }
        }
      }
    }

    if ((a == kAction::re) && !buffer_empty()) a = kAction::sh;
  }

  return a;
}

bool ArcStandardLabelledParser::inTerminalConfiguration() const {
  return (buffer_empty() && (stack_depth() == 1));
}

bool ArcStandardLabelledParser::executeAction(kAction a, WordId l) {
  switch (a) {
    case kAction::sh:
      return shift();
    case kAction::la:
      return leftArc(l);
    case kAction::ra:
      return rightArc(l);
    case kAction::la2:
      return leftArc2(l);
    case kAction::ra2:
      return rightArc2(l);
    default:
      std::cerr << "action not implemented" << std::endl;
      return false;
  }
}

//TODO check orders
Indices ArcStandardLabelledParser::contextIndices() const {
  if (context_type() == "extended")
    return extended_children_context();  // order 13
  if (context_type() == "extended-3rd")
    return third_extended_children_context();  // order 18
  else if (context_type() == "more-extended")
    return more_extended_children_context();  // order 17
  else if (context_type() == "more-extended-3rd")
    return third_more_extended_children_context();  // order 24
  else if (context_type() == "with-ngram")
    return children_ngram_context();  // order 13
  else if (context_type() == "extended-with-ngram")
    return extended_children_ngram_context();  // order 16
  else if (context_type() == "more-extended-with-ngram")
    return more_extended_children_ngram_context();  // order 21
  else if (context_type() == "lookahead")
    return children_lookahead_context();  // order 10
  else if (context_type() == "extended-lookahead")
    return extended_children_lookahead_context();  // order 16
  else if (context_type() == "standard-3rd")
    return third_children_context();  // order 10
  else
    return children_context();  // order 7
}

Context ArcStandardLabelledParser::wordContext() const {
  if (pyp_model()) {
    return Context(word_tag_next_children_context());  // order 6
  } else if (context_type() == "stack-action") {
    Context ctx = stack_action_context();
    if (predict_pos()) {
      ctx.features.back().push_back(
          features_at(buffer_next())[0]);  // adds next tag as a feature
    }
    return ctx;
  } else {
    Context ctx = map_context(contextIndices());  // order 9
    if (predict_pos()) {
      ctx.features.back().push_back(features_at(buffer_next())[0]);
    }
    return ctx;
  }
}

Context ArcStandardLabelledParser::tagContext() const {
  if (pyp_model()) {
    if (lexicalised()) {
      return Context(word_tag_children_context());  // order 9
    }
    else {
      return Context(tag_children_context());  // order 7
    }
  } else if (context_type() == "stack-action") {
    return stack_action_context();  // order 9
  } else {
    return map_context(contextIndices());
  }
}

Context ArcStandardLabelledParser::actionContext() const {
  if (pyp_model()) {
    if (lexicalised()) {
      return Context(word_tag_children_context());  // order 9
    } else {
      return Context(tag_children_context());  // order 7
    }
  } else if (context_type() == "stack-action") {
    return stack_action_context();  // order 9
  } else {
    return map_context(contextIndices());
  }
}

void ArcStandardLabelledParser::extractExamples(
    const boost::shared_ptr<ParseDataSet>& examples,
    const ParsedSentence& gold_sent) const {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(*this),
                                   config());

  for (unsigned i = 0; i < actions().size(); ++i) {
    kAction a = actions().at(i);
    WordId lab = action_label_at(i);

    // Add oracle action as training example.
    kAction gold_a = parser.oracleNext(gold_sent);
    WordId gold_lab = parser.oracleNextLabel(gold_sent);

    if (gold_a == kAction::sh) {
      examples->add_tag_example(
          DataPoint(parser.next_tag(), parser.tagContext()));
      examples->add_word_example(
          DataPoint(parser.next_word(), parser.wordContext()));
    }

    // Labelled action prediction.
    if (gold_a != kAction::re) {
      examples->add_action_example(
          DataPoint(convert_action(gold_a, gold_lab), parser.actionContext()));
    }

    parser.executeAction(a, lab);
  }
}

void ArcStandardLabelledParser::extractExamples(
    const boost::shared_ptr<ParseDataSet>& examples) const {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(*this),
                                   config());

  // Note that we are extracting the initial shift as an example.
  for (unsigned i = 0; i < actions().size(); ++i) {
    kAction a = actions().at(i);
    WordId lab = action_label_at(i);

    if (a == kAction::sh) {
      examples->add_tag_example(
          DataPoint(parser.next_tag(), parser.tagContext(), id()));
      examples->add_word_example(
          DataPoint(parser.next_word(), parser.wordContext(), id()));
    }

    // Labelled action prediction.
    WordId lab_act = convert_action(a, lab);
    examples->add_action_example(
        DataPoint(lab_act, parser.actionContext(), id()));
    parser.executeAction(a, lab);
  }
}

}  // namespace oxlm
