#ifndef _GDP_AE_PARSER_H_
#define _GDP_AE_PARSER_H_

#include "corpus/parse_data_set.h"
#include "gdp/transition_parser.h"

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
    //stack_size 1 -> stack top is root
    if (stack_depth() < 2)
      return false;    
    WordIndex i = stack_top();
    return (!has_parent_at(i));
  }

  bool reduce_valid() const {
    WordIndex i = stack_top();
    //if STOP, should not have parent, else it should
    if (tag_at(i) == 1)
      return !has_parent_at(i);
    else
      return has_parent_at(i);
  }

  //TODO update where this is used
  /*bool is_complete_parse() const {
    for (WordIndex i = 1; i < arcs_.size() - 1; ++i) {
      if (!arcs_.has_parent(i) && (tags_.at(i)!=1))
        return false;
    }

    return ((buffer_next_ >= 3) && !buffer_next_has_child());
  } */

  static bool cmp_reduce_particle_weights(const boost::shared_ptr<ArcEagerParser>& p1, 
                                 const boost::shared_ptr<ArcEagerParser>& p2) {
  //null should be the biggest
  if (p1 == nullptr)
    return false;
  else if (p2 == nullptr)
    return true;
  //then those that cannot reduce
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
