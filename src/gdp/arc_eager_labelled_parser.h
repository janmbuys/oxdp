#ifndef _GDP_AE_LAB_PARSER_H_
#define _GDP_AE_LAB_PARSER_H_

#include "corpus/parse_data_set.h"
#include "gdp/transition_parser.h"
#include "gdp/transition_parser_interface.h"

namespace oxlm {

class ArcEagerLabelledParser : public TransitionParser {
  public:

  ArcEagerLabelledParser(int num_labels);

  ArcEagerLabelledParser(Words sent, int num_labels);

  ArcEagerLabelledParser(Words sent, Words tags, int num_labels);

  ArcEagerLabelledParser(Words sent, Words tags, int num_particles, int num_labels);

  ArcEagerLabelledParser(const TaggedSentence& parse, int num_labels);

  ArcEagerLabelledParser(const TaggedSentence& parse, int num_particles, int num_labels);

  bool shift();
  
  bool shift(WordId w);

  bool leftArc(WordId l);

  bool rightArc(WordId l);
  
  bool rightArc(WordId l, WordId w);
  
  bool reduce();

  WordId oracleNextLabel(const ParsedSentence& gold_parse) const;

  kAction oracleNext(const ParsedSentence& gold_parse) const;

  bool inTerminalConfiguration() const;
 
  bool executeAction(kAction a, WordId l);
 
  Words wordContext() const;
 
  Words tagContext() const;
 
  Words tagContext(kAction a) const;

  Words actionContext() const;
 
  void extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const;

  void append_action_label(WordId l) {
    action_labels_.push_back(l);
  }

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

  WordId action_label_at(int i) const {
    return action_labels_[i];
  }

  //TODO update where this is used
  /*bool is_complete_parse() const {
    for (WordIndex i = 1; i < arcs_.size() - 1; ++i) {
      if (!arcs_.has_parent(i) && (tags_.at(i)!=1))
        return false;
    }

    return ((buffer_next_ >= 3) && !buffer_next_has_child());
  } */

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
      return num_labels_ + l + 1;
    else if (a == kAction::re)
      return 2*num_labels_ + 1;
    else
      return -1;
  }

  kAction lookup_action(WordId l) const {
    if (l == 0)
      return kAction::sh;
    else if (l <= num_labels_)
      return kAction::la;
    else if (l <= 2*num_labels_)
      return kAction::ra;
    else if (l == 2*num_labels_ + 1)
      return kAction::re;
    else
      return kAction::re;
  }

  WordId lookup_label(WordId l) const {
    if (l == 0)
      return -1;
    else if (l <= num_labels_)
      return l - 1;
    else if (l <= num_labels_*2)
      return l - num_labels_ - 1;
    else //include reduce
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
