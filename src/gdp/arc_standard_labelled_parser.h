#ifndef _GDP_AS_LAB_PARSER_H
#define _GDP_AS_LAB_PARSER_H

#include "corpus/parse_data_set.h"
#include "gdp/transition_parser.h"

namespace oxlm {

// Implements a arc-standard transition-based parser.  
class ArcStandardLabelledParser : public TransitionParser {
 public:
  ArcStandardLabelledParser(const boost::shared_ptr<ModelConfig>& config);

  ArcStandardLabelledParser(const TaggedSentence& parse,
                            const boost::shared_ptr<ModelConfig>& config);

  ArcStandardLabelledParser(const TaggedSentence& parse, int num_particles,
                            const boost::shared_ptr<ModelConfig>& config);

  bool shift();

  bool shift(WordId w);

  bool leftArc(WordId l);

  bool rightArc(WordId l);

  bool leftArc2(WordId l);

  bool rightArc2(WordId l);

  // Returns the label for the oracle reduce action, if valid.
  kAction oracleNext(const ParsedSentence& gold_parse) const;

  // Predicts the next action according to the oracle.
  WordId oracleNextLabel(const ParsedSentence& gold_parse) const;

  bool inTerminalConfiguration() const;

  bool executeAction(kAction a, WordId l);

  Indices contextIndices() const;

  Context wordContext() const;

  Context tagContext() const;

  Context actionContext() const;

  void extractExamples(const boost::shared_ptr<ParseDataSet>& examples,
                       const ParsedSentence& gold_sent) const;

  void extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const;

  bool left_arc_valid() const {
    if (stack_depth() < 2) return false;
    WordIndex i = stack_top_second();
    return (i != 0);
  }

  bool left_arc2_valid() const {
    if (stack_depth() < 3) return false;
    WordIndex i = stack_top_third();
    return (i != 0);
  }

  kAction lookup_action(WordId la) const {
    if (la == 0) {
      return kAction::sh;
    } else if (la <= num_labels()) {
      return kAction::la;
    } else if (la <= 2 * num_labels()) {
      return kAction::ra;
    } else if (la <= 3 * num_labels()) {
      return kAction::la2;
    } else if (la <= 4 * num_labels()) {
      return kAction::ra2;
    } else {
      return kAction::re;
    }
  }

  WordId lookup_label(WordId la) const {
    if (la == 0) {
      return -1;
    } else if (la <= num_labels()) {
      return la - 1;
    } else if (la <= 2 * num_labels()) {
      return la - num_labels() - 1;
    } else if (la <= 3 * num_labels()) {
      return la - 2 * num_labels() - 1;
    } else if (la <= 4 * num_labels()) {
      return la - 3 * num_labels() - 1;
    } else {
      return -1;
    }
  }
};

}  // namespace oxlm

#endif
