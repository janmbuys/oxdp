#ifndef _GDP_TRP_H_
#define _GDP_TRP_H_

#include<string>
#include<functional>
#include<cstdlib>

#include<corpus/corpus.h>

namespace oxlm {

enum class kAction : WordId {sh, la, ra, re, la2, ra2};

typedef std::vector<WordIndex> WxList;
typedef std::vector<Words> WordsList;
typedef std::vector<kAction> ActList;

class ArcList {

public:
  ArcList(): 
    arcs_(),
    child_count_()
  {
  }

  ArcList(unsigned n): 
    arcs_(n, -1),
    child_count_(n, 0)
  { 
  }

  void push_back() {
    arcs_.push_back(-1);
    child_count_.push_back(0);
  }

  void set_arc(WordIndex i, WordIndex j) {
    //node i has parent j
    arcs_[i] = j;
    if (j >= 0)
      ++child_count_[j];
  }

  void set_arcs(const WxList& arcs) {
    for (int i = 0; i < static_cast<int>(size()); ++i) {
      set_arc(i, arcs[i]);
    }
  }

  WxList arcs() const {
    return arcs_;
  }

  WordIndex at(WordIndex i) const {
    return arcs_[i];
  }

  bool has_parent(WordIndex i) const {
    return (arcs_[i] >= 0);
  }

  int child_count_at(WordIndex i) const {
    return child_count_[i];
  }

  unsigned size() const {
    return arcs_.size();
  }

  bool has_arc(WordIndex i, WordIndex j) const {
    return (arcs_[i] == j);
  }

  bool is_projective_dependency() const {
    for (int i = 0; i < static_cast<int>(arcs_.size() - 1); ++i)
      for (int j = i + 1; j < static_cast<int>(arcs_.size()); ++j)
        if ((arcs_[i]<i &&
              (arcs_[j]<i && arcs_[j]>arcs_[i])) ||
            ((arcs_[i]>i && arcs_[i]>j) &&
              (arcs_[j]<i || arcs_[j]>arcs_[i])) ||
            ((arcs_[i]>i && arcs_[i]<j) &&
              (arcs_[j]>i && arcs_[j]<arcs_[i])))
          return false;
    return true;
  }

  bool operator==(const ArcList& a) const {
    return (arcs_==a.arcs());
  }

private:
  WxList arcs_;
  std::vector<int> child_count_;
};

class TransitionParser {
  public:

  TransitionParser():  
    stack_(),
    buffer_(),
    arcs_(),
    actions_(),
    sentence_(),
    tags_(),
    liw_{0},
    lpw_{0},
    num_particles_{1}
  {
  }

  TransitionParser(Words sent):  
    stack_(),
    buffer_(sent.size(), 0),
    arcs_(sent.size()),
    actions_(),
    sentence_(sent),
    tags_(),
    liw_{0},
    lpw_{0},
    num_particles_{1}
  {
    //sentence already includes root (0)
    for (int i = 0; i < (int)sentence_.size(); ++i)
      buffer_[i] = sentence_.size() - i - 1; 
  }

  TransitionParser(Words sent, Words tags):  
    stack_(),
    buffer_(sent.size(), 0),
    arcs_(sent.size()),
    actions_(),
    sentence_(sent),
    tags_(tags),
    liw_{0},
    lpw_{0},
    num_particles_{1}
  {
    //sentence already includes root (0)
    for (int i = 0; i < (int)sentence_.size(); ++i)
      buffer_[i] = sentence_.size() - i - 1; 
  }

  bool shift();

  bool shift(WordId w);
  
  bool buffer_tag(WordId t);

  /* 
  //not currently used
  void reduce() {
    stack_.pop_back();
    actions_.push_back(kAction:re);
  } */

  virtual bool leftArc() = 0;

  virtual bool rightArc() = 0;

  //not compulsory any more
  //virtual bool SentenceOracle(WxList gold_arcs) = 0;

  bool execute_action(kAction a) {
    switch(a) {
    case kAction::sh:
      return shift();
    case kAction::la:
      return leftArc();
    case kAction::ra:
      return rightArc();
    default: 
      //other cases not implemented
      std::cerr << "action not implemented" << std::endl;
      return false;
    }
  }

  void reset_importance_weight() {
    liw_ = 0;
  }

  void set_importance_weight(double w) {
    liw_ = -std::log(w);
  }

  void add_importance_weight(double w) {
    liw_ -= std::log(w);
  }

  void set_log_particle_weight(double w) {
    lpw_ = w;
  }

  void set_particle_weight(double w) {
    lpw_ = -std::log(w);
  }

  void add_particle_weight(double w) {
    lpw_ -= std::log(w);
  }

  void set_num_particles(int n) {
    num_particles_ = n;
  }

  std::string actions_str() const {
    const std::vector<std::string> action_names {"sh", "la", "ra", "re", "la2", "ra2"};
    std::string seq_str = "";
    for (kAction a: actions_)
      seq_str += action_names[static_cast<int>(a)] + " ";
    return seq_str; 
  }
  
  void print_arcs() const {
    for (auto a: arcs_.arcs())
      std::cout << a << " ";
    //std::cout << std::endl;
  }

  void print_sentence(Dict& dict) const {
    for (auto a: sentence_)
      std::cout << dict.lookup(a) << " ";
    std::cout << std::endl;
  }

  void print_tags(Dict& dict) const {
    for (auto a: tags_)
      std::cout << dict.lookupTag(a) << " ";
    std::cout << std::endl;
  }

  unsigned sentence_length() const {
    return sentence_.size();
  }

  Words sentence() const {
    return sentence_;
  }

  Words tags() const {
    return tags_;
  }

  ArcList arcs() const {
    return arcs_;
  }

  double importance_weight() const {
    return liw_;
  }

  double particle_weight() const {
    return lpw_;
  }

  int num_particles() const {
    return num_particles_;
  }

  double weighted_importance_weight() const {
    return (liw_ - std::log(num_particles_));
  }

  double weighted_particle_weight() const {
    return (lpw_- std::log(num_particles_));
  }

  ActList actions() const {
    return actions_;
  }
   
  unsigned num_actions() const {
    return actions_.size();
  }

  //number of children at sentence position i
  int child_count_at(int i) const {
    return arcs_.child_count_at(i);
  }

  bool has_parent(int i) const {
    return arcs_.has_parent(i);
  }

  int stack_depth() const {
    return stack_.size();
  }

  int buffer_length() const {
    return buffer_.size();
  }

  WordIndex stack_top() const {
    return stack_.back();
  }

  WordId next_word() const {
    return sentence_[buffer_.back()];
  }

  WordId next_tag() const {
    return tags_[buffer_.back()];
  }

  WordIndex buffer_next() const {
    return buffer_.back();
  }

  bool is_buffer_empty() const {
    return buffer_.empty();
  }

  bool is_terminal_configuration() const {
    return (buffer_.empty() && (stack_.size() == 1));
  }

  unsigned directed_accuracy_count(ArcList g_arcs) const {
    unsigned count = 0;
    for (unsigned j = 1; j < arcs_.size(); ++j) {
      if (arcs_.at(j)==g_arcs.at(j))
        ++count;
    }
    return count;
  }

  unsigned undirected_accuracy_count(ArcList g_arcs) const {
    unsigned count = 0;
    for (unsigned j = 1; j < arcs_.size(); ++j) {
      if ((arcs_.at(j)==g_arcs.at(j)) || (arcs_.has_parent(j) && g_arcs.at(arcs_.at(j))==static_cast<int>(j)))
        ++count;
    }
    return count;
  }

  //**functions that call the context vector functions for a given configuration
  //(ideally would assert length of order)
  Words shift_context() const {
    return word_tag_next_context();
  }

  Words reduce_context() const {
    return tag_context();
  }

  Words arc_context() const {
    return tag_context();
  }

  //****functions for context vectors: each function defined for a specific context length

  //overloaded in direct use, but this shouldn't be a problem
  Words tag_context() const {
    Words ctx(2, 0);
    
    if (stack_.size() >= 1) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[0] = tags_.at(stack_.at(stack_.size()-2));
    }

    return ctx;
  }

  Words word_context() const {
    Words ctx(2, 0);
    
    if (stack_.size() >= 1) { 
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
    }

    return ctx;
  }
  
  Words word_tag_context() const {
    Words ctx(4, 0);
    
    //word and pos context of 2
    if (stack_.size() >= 1) { 
      ctx[1] = sentence_.at(stack_.at(stack_.size()-1));
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) {
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
      ctx[2] = tags_.at(stack_.at(stack_.size()-2));
    }

    return ctx;
  }

  Words word_tag_less_context() const {
    Words ctx(3, 0);

    //word context 1 and pos context 2
    if (stack_.size() >= 1) { 
      ctx[0] = sentence_.at(stack_.at(stack_.size()-1));
      ctx[2] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[1] = tags_.at(stack_.at(stack_.size()-2));
    }

    return ctx;
  }

  Words word_tag_next_context() const {
    Words ctx(5, 0);
    
    //word context 2, pos context 2 + next token
    if (stack_.size() >= 1) { 
      ctx[2] = sentence_.at(stack_.at(stack_.size()-1));
      ctx[3] = tags_.at(stack_.at(stack_.size()-1));
    }
    if (stack_.size() >= 2) { 
      ctx[0] = sentence_.at(stack_.at(stack_.size()-2));
      ctx[1] = tags_.at(stack_.at(stack_.size()-2));
    }
    if (buffer_.size() > 0)
      ctx[4] = tags_.at(buffer_.back());

    return ctx;
  }

  protected:
  WxList stack_;
  WxList buffer_;
  ArcList arcs_;
  ActList actions_;
  private:
  Words sentence_;
  Words tags_;
  double liw_; //log importance weight
  double lpw_; //log particle weight
  int num_particles_;
};


class ArcStandardParser : public TransitionParser {
  public:

  ArcStandardParser():
    TransitionParser() {
  }

  ArcStandardParser(Words sent):
    TransitionParser(sent) {
  }

  ArcStandardParser(Words sent, Words ptags):
    TransitionParser(sent, ptags) {
  }

  bool leftArc();

  bool rightArc();
  
  bool left_arc_valid() const {
    if (stack_.size() < 2)
      return false;
    WordIndex i = stack_.rbegin()[1];
    return (i != 0);
  }

  kAction oracleNext(const ArcList& gold_arcs) const;
  
  kAction oracleDynamicNext(const ArcList& gold_arcs) const;
  
};

class AccuracyCounts {

public:
  AccuracyCounts(): 
    reduce_count_{0},
    reduce_gold_{0},
    shift_count_{0},
    shift_gold_{0},
    final_reduce_error_count_{0},
    total_length_{0},
    directed_count_{0},
    undirected_count_{0}, 
    root_count_{0},
    complete_sentences_{0},
    num_sentences_{0}
  {
  }

  void inc_reduce_count() {
    ++reduce_count_;
  }

  void inc_reduce_gold() {
    ++reduce_gold_;
  }

  void inc_shift_count() {
    ++shift_count_;
  }

  void inc_shift_gold() {
    ++shift_gold_;
  }

  void inc_final_reduce_error_count() {
    ++final_reduce_error_count_;
  }

  void inc_complete_sentences() {
    ++complete_sentences_;
  }

  void inc_root_count() {
    ++root_count_;
  }

  void inc_num_sentences() {
    ++num_sentences_;
  }

  void add_total_length(int l) {
    total_length_ += l; 
  }

  void add_directed_count(int l) {
    directed_count_ += l; 
  }

  void add_undirected_count(int l) {
    undirected_count_ += l; 
  }
   
  void countAccuracy(const ArcStandardParser& prop_parse, const ArcList& gold_arcs); 

  double directed_accuracy() const {
    return (directed_count_ + 0.0)/total_length_;
  }

  double undirected_accuracy() const {
    return (undirected_count_ + 0.0)/total_length_;
  }

  double complete_accuracy() const {
    return (complete_sentences_ + 0.0)/num_sentences_;
  }

  double root_accuracy() const {
    return (root_count_ + 0.0)/num_sentences_;
  }

  double arc_dir_precision() const {
    return (directed_count_ + 0.0)/undirected_count_;
  }

  double reduce_recall() const {
    return (reduce_count_ + 0.0)/reduce_gold_;
  }

  double shift_recall() const {
    return (shift_count_ + 0.0)/shift_gold_;
  }

  double final_reduce_error_rate() const {
    return (final_reduce_error_count_ + 0.0)/total_length_;
  }

private:
    int reduce_count_; 
    int reduce_gold_;
    int shift_count_;
    int shift_gold_;
    int final_reduce_error_count_;
    int total_length_;
    int directed_count_;
    int undirected_count_;
    int root_count_;
    int complete_sentences_;
    int num_sentences_;
};

inline bool cmp_importance_weights(ArcStandardParser& p1, ArcStandardParser& p2) {
  return (p1.importance_weight() < p2.importance_weight());
}
 
inline bool cmp_particle_weights(ArcStandardParser& p1, ArcStandardParser& p2) {
  return (p1.particle_weight() < p2.particle_weight());
}

inline bool cmp_particle_ptr_weights(const std::unique_ptr<ArcStandardParser>& p1, const std::unique_ptr<ArcStandardParser>& p2) {
  //null should be the biggest
  if (p1 == nullptr)
    return false;
  else if (p2 == nullptr)
    return true;
  else
    return (p1->particle_weight() < p2->particle_weight());
}

inline bool cmp_weighted_particle_ptr_weights(const std::unique_ptr<ArcStandardParser>& p1, const std::unique_ptr<ArcStandardParser>& p2) {
  //null or no particles should be the biggest
  if ((p1 == nullptr) || (p1->num_particles() == 0))
    return false;
  else if ((p2 == nullptr) || (p2->num_particles() == 0))
    return true;
  else
    return (p1->weighted_particle_weight() < p2->weighted_particle_weight());
}

inline bool cmp_weighted_importance_ptr_weights(const std::unique_ptr<ArcStandardParser>& p1, const std::unique_ptr<ArcStandardParser>& p2) {
  //null or no particles should be the biggest
  if ((p1 == nullptr) || (p1->num_particles() == 0))
    return false;
  else if ((p2 == nullptr) || (p2->num_particles() == 0))
    return true;
  else
    return (p1->weighted_importance_weight() < p2->weighted_importance_weight());
}

inline kAction convert_to_action(WordId a) {
  std::vector<kAction> actList = {kAction::sh, kAction::la, kAction::ra, kAction::re, kAction::la2, kAction::ra2};
  return actList[a];
}

}
#endif
