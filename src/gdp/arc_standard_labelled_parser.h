#ifndef _GDP_AS_LAB_PARSER_H
#define _GDP_AS_LAB_PARSER_H

#include "corpus/parse_data_set.h"
#include "gdp/transition_parser.h"

namespace oxlm {

class ArcStandardLabelledParser : public TransitionParser {
 public:
  ArcStandardLabelledParser(const boost::shared_ptr<ModelConfig>& config);

  ArcStandardLabelledParser(const TaggedSentence& parse, const boost::shared_ptr<ModelConfig>& config);
  
  ArcStandardLabelledParser(const TaggedSentence& parse, int num_particles, const boost::shared_ptr<ModelConfig>& config);

  bool shift();

  bool shift(WordId w);

  bool leftArc(WordId l); 

  bool rightArc(WordId l); 
  
  kAction oracleNext(const ParsedSentence& gold_parse) const;
  
  WordId oracleNextLabel(const ParsedSentence& gold_parse) const;

  bool inTerminalConfiguration() const;

  bool executeAction(kAction a, WordId l); 
 
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
    return (!(root_first() && (i == 0)));
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
      return num_labels() + l + 1;
    else 
      return -1;
  }

  kAction lookup_action(WordId la) const {
    if (la == 0)
      return kAction::sh;
    else if (la <= num_labels())
      return kAction::la;
    else if (la <= 2*num_labels())
      return kAction::ra;
    else
      return kAction::re;
  }

  WordId lookup_label(WordId la) const {
    if (la == 0)
      return -1;
    else if (la <= num_labels())
      return la - 1;
    else if (la <= 2*num_labels())
      return la - num_labels() - 1;
    else
      return -1;
  }

  private:
  Words action_labels_;  
};

}

#endif
