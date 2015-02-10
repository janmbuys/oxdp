#ifndef _GDP_AS_PARSER_H
#define _GDP_AS_PARSER_H

#include "corpus/parse_data_set.h"
#include "gdp/transition_parser.h"

namespace oxlm {

class ArcStandardParser : public TransitionParser {
 public:
  ArcStandardParser(const boost::shared_ptr<ModelConfig>& config);

  ArcStandardParser(const TaggedSentence& parse, const boost::shared_ptr<ModelConfig>& config);
  
  ArcStandardParser(const TaggedSentence& parse, int num_particles, const boost::shared_ptr<ModelConfig>& config);

  bool shift();

  bool shift(WordId w);

  bool leftArc();

  bool rightArc();
  
  kAction oracleNext(const ParsedSentence& gold_parse) const;
  
  bool inTerminalConfiguration() const;

  bool executeAction(kAction a);
 
  Words wordContext() const;

  Words tagContext() const;
 
  Words actionContext() const;
 
  bool left_arc_valid() const {
    if (stack_depth() < 2)
      return false;
    WordIndex i = stack_top_second();
    if (root_first())
      return (i != 0);
    else 
      return true; // root can never be dependent of left-arc
  }

  void extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const;

};

}

#endif
