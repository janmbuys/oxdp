#ifndef _GDP_AS_PARSER_H
#define _GDP_AS_PARSER_H

#include "gdp/transition_parser.h"
#include "gdp/transition_parser_interface.h"

namespace oxlm {

class ArcStandardParser : public TransitionParser, public TransitionParserInterface {
  public:

  ArcStandardParser();

  ArcStandardParser(Words sent);

  ArcStandardParser(Words sent, Words tags);

  ArcStandardParser(Words sent, Words tags, int num_particles);

  ArcStandardParser(const ParsedSentence& parse);
  
  ArcStandardParser(const ParsedSentence& parse, int num_particles);

  bool shift() override;

  bool shift(WordId w);

  bool leftArc() override;

  bool rightArc() override;
  
  kAction oracleNext(const ParsedSentence& gold_parse) const override;
  
  bool inTerminalConfiguration() const override;

  bool executeAction(kAction a) override;
 
  Words wordContext() const override;

  Words tagContext() const override;
 
  Words actionContext() const override;
 
  bool left_arc_valid() const {
    if (stack_depth() < 2)
      return false;
    WordIndex i = stack_top_second();
    return (i != 0);
  }

  //just in case this might help
  //but this should be static...
  /*
  size_t reduce_context_size() const {
    return reduce_context().size();
  }

  size_t shift_context_size() const {
    return shift_context().size();
  }

  size_t tag_context_size() const {
    return tag_context().size();
  }  */

};

}

#endif
