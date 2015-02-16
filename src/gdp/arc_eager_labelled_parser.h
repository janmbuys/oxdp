#ifndef _GDP_AE_LAB_PARSER_H_
#define _GDP_AE_LAB_PARSER_H_

#include "corpus/parse_data_set.h"
#include "gdp/transition_parser.h"

namespace oxlm {

class ArcEagerLabelledParser : public TransitionParser {
  public:

  ArcEagerLabelledParser(const boost::shared_ptr<ModelConfig>& config);

  ArcEagerLabelledParser(const TaggedSentence& parse, const boost::shared_ptr<ModelConfig>& config);

  ArcEagerLabelledParser(const TaggedSentence& parse, int num_particles, const boost::shared_ptr<ModelConfig>& config);

  bool shift();
  
  bool shift(WordId w);

  bool leftArc(WordId l);

  bool rightArc(WordId l);
  
  //bool rightArc(WordId l, WordId w);
  
  bool reduce();

  WordId oracleNextLabel(const ParsedSentence& gold_parse) const;

  kAction oracleNext(const ParsedSentence& gold_parse) const;

  bool inTerminalConfiguration() const;
 
  bool executeAction(kAction a, WordId l);
 
  Words wordContext() const;
 
  Words tagContext() const;
 
  //Words tagContext(kAction a) const;

  Words actionContext() const;
 
  void extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const;

  void append_action_label(WordId l) {
    action_labels_.push_back(l);
  }

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
    return (has_parent_at(i));
  }

  WordId action_label_at(int i) const {
    return action_labels_[i];
  }

  static bool cmp_reduce_particle_weights(const boost::shared_ptr<ArcEagerLabelledParser>& p1, 
                                 const boost::shared_ptr<ArcEagerLabelledParser>& p2) {
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

  WordId convert_action(kAction a, WordId l) const {
    if (a == kAction::sh)
      return 0;
    else if (a == kAction::la)
      return l + 1;
    else if (a == kAction::ra)
      return num_labels() + l + 1;
    else if (a == kAction::re)
      return 2*num_labels() + 1;
    else
      return -1;
  }

  kAction lookup_action(WordId l) const {
    if (l == 0)
      return kAction::sh;
    else if (l <= num_labels())
      return kAction::la;
    else if (l <= 2*num_labels())
      return kAction::ra;
    else if (l == 2*num_labels() + 1)
      return kAction::re;
    else
      return kAction::re;
  }

  bool shift_action(WordId l) const {
    return ((l == 0) || ((l > num_labels() + 1) && (l < 2*num_labels() + 1)));
  }

  WordId lookup_label(WordId l) const {
    if (l == 0)
      return -1;
    else if (l <= num_labels())
      return l - 1;
    else if (l <= 2*num_labels())
      return l - num_labels() - 1;
    else //include reduce
      return -1;
  }

 private:
  Words action_labels_;  
};

}

#endif
