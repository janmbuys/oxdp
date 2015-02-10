#ifndef _GDP_AS_LAB_PARSER_H
#define _GDP_AS_LAB_PARSER_H

#include "corpus/parse_data_set.h"
#include "gdp/transition_parser.h"

namespace oxlm {

class ArcStandardLabelledParser : public TransitionParser {
  public:

  ArcStandardLabelledParser(int num_labels);

  ArcStandardLabelledParser(Words sent, int num_labels);

  ArcStandardLabelledParser(Words sent, Words tags, int num_labels);

  ArcStandardLabelledParser(Words sent, Words tags, int num_particles, int num_labels);

  ArcStandardLabelledParser(const TaggedSentence& parse, int num_labels);
  
  ArcStandardLabelledParser(const TaggedSentence& parse, int num_particles, int num_labels);

  bool shift();

  bool shift(WordId w);

  bool leftArc(WordId l); //not overriding any more

  bool rightArc(WordId l); //not overriding any more
  
  kAction oracleNext(const ParsedSentence& gold_parse) const;
  
  WordId oracleNextLabel(const ParsedSentence& gold_parse) const;

  bool inTerminalConfiguration() const;

  bool executeAction(kAction a, WordId l); //not overriding any more
 
  Words wordContext() const;

  Words tagContext() const;
 
  Words actionContext() const;
 
  void extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const;

  void append_action_label(WordId l) {
    action_labels_.push_back(l);
  }

  bool left_arc_valid() const {
    if (stack_depth() < 2)
      return false;
    WordIndex i = stack_top_second();
    return (i != 0);
  }

  WordId action_label_at(int i) const {
    return action_labels_[i];
  }

  WordId convert_action(kAction a, WordId l) const {
    if (a == kAction::sh)
      return 0;
    else if (a == kAction::la)
      return l + 1;
    else if (a == kAction::ra)
      return num_labels_ + l + 1;
    else 
      return -1;
  }

  kAction lookup_action(WordId la) const {
    if (la == 0)
      return kAction::sh;
    else if (la <= num_labels_)
      return kAction::la;
    else if (la <= num_labels_*2)
      return kAction::ra;
    else
      return kAction::re;
  }

  WordId lookup_label(WordId la) const {
    if (la == 0)
      return -1;
    else if (la <= num_labels_)
      return la - 1;
    else if (la <= num_labels_*2)
      return la - num_labels_ - 1;
    else
      return -1;
  }

  int num_labels() const {
    return num_labels_;
  }

  private:
  int num_labels_;
  Words action_labels_;  
};

}

#endif
