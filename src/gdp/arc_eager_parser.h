#ifndef _GDP_AE_PARSER_H_
#define _GDP_AE_PARSER_H_

#include "corpus/parse_data_set.h"
#include "gdp/transition_parser.h"

//Currently not developing this class
namespace oxlm {

class ArcEagerParser : public TransitionParser {
 public:
  ArcEagerParser(const boost::shared_ptr<ModelConfig>& config);

  ArcEagerParser(const TaggedSentence& parse, const boost::shared_ptr<ModelConfig>& config);

  ArcEagerParser(const TaggedSentence& parse, int num_particles, const boost::shared_ptr<ModelConfig>& config);

  bool shift();
  
  bool shift(WordId w);

  bool leftArc();

  bool rightArc();
  
  bool rightArc(WordId w);
  
  bool reduce();

  kAction oracleNext(const ParsedSentence& gold_parse) const;

  bool inTerminalConfiguration() const;
 
  bool executeAction(kAction a);
 
  Words wordContext() const;
 
  Words tagContext() const;
 
  Words tagContext(kAction a) const;

  Words actionContext() const;
 
  void extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const;

  bool left_arc_valid() const {
    if (stack_depth() == 0)
      return false;    
    WordIndex i = stack_top();
    return (!has_parent_at(i) && !(root_first() && (i == 0)));
  }

  bool reduce_valid() const {
    if (stack_depth() == 0)
      return false;    
    WordIndex i = stack_top();
    return (!has_parent_at(i));
  }

  static bool cmp_reduce_particle_weights(const boost::shared_ptr<ArcEagerParser>& p1, 
                                 const boost::shared_ptr<ArcEagerParser>& p2) {
  //null should be the biggest
  if (p1 == nullptr)
    return false;
  else if (p2 == nullptr)
    return true;
  //then particles that cannot reduce
  else if (!p1->reduce_valid())
    return false;
  else if (!p2->reduce_valid())
    return true;
  else
    return (p1->particle_weight() < p2->particle_weight());
  }

};

}

#endif
