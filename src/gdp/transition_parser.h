#ifndef _CGDP_SRP_H_
#define _CGDP_SRP_H_

#include<string>
#include<functional>
#include<cstdlib>

#include<corpus/corpus.h>

namespace oxlm {

enum class Action : WordId {sh, la, ra, re, la2, ra2};

typedef std::vector<WordIndex> WxList;
typedef std::vector<Words> WordsList;
typedef std::vector<Action> ActList;

inline std::vector<int> count_children(WxList g_arcs) {
  std::vector<int> g_child_count(g_arcs.size(), 0);
  for (unsigned j = 1; j < g_arcs.size(); ++j)
    ++g_child_count[g_arcs[j]];
  return g_child_count;
} 

class AccuracyCounts {

public:
  AccuracyCounts() : 
    reduce_count{0},
    reduce_gold{0},
    shift_count{0},
    shift_gold{0},
    final_reduce_error_count{0},
    total_length{0},
    directed_count{0},
    undirected_count{0}, 
    root_count{0},
    complete_sentences{0},
    num_sentences{0}
    {
    }

  void inc_reduce_count() {
    ++reduce_count;
  }

  void inc_reduce_gold() {
    ++reduce_gold;
  }

  void inc_shift_count() {
    ++shift_count;
  }

  void inc_shift_gold() {
    ++shift_gold;
  }

  void inc_final_reduce_error_count() {
    ++final_reduce_error_count;
  }

  void inc_complete_sentences() {
    ++complete_sentences;
  }

  void inc_root_count() {
    ++root_count;
  }

  void inc_num_sentences() {
    ++num_sentences;
  }

  void add_total_length(int l) {
    total_length += l; 
  }

  void add_directed_count(int l) {
    directed_count += l; 
  }

  void add_undirected_count(int l) {
    undirected_count += l; 
  }
   
  double const directed_accuracy() {
    return (directed_count + 0.0)/total_length;
  }

  double const undirected_accuracy() {
    return (undirected_count + 0.0)/total_length;
  }

  double const complete_accuracy() {
    return (complete_sentences + 0.0)/num_sentences;
  }

  double const root_accuracy() {
    return (root_count + 0.0)/num_sentences;
  }

  double const arc_dir_precision() {
    return (directed_count + 0.0)/undirected_count;
  }

  double const reduce_recall() {
    return (reduce_count + 0.0)/reduce_gold;
  }

  double const shift_recall() {
    return (shift_count + 0.0)/shift_gold;
  }

  double const final_reduce_error_rate() {
    return (final_reduce_error_count + 0.0)/total_length;
  }

protected:
    int reduce_count; 
    int reduce_gold;
    int shift_count;
    int shift_gold;
    int final_reduce_error_count;
    int total_length;
    int directed_count;
    int undirected_count;
    int root_count;
    int complete_sentences;
    int num_sentences;
};

class TransitionParser {
  public:
  TransitionParser(Words sent, unsigned context_size):  
    stack_(),
    buffer_(sent.size()),
    arcs_(sent.size(), -1),
    child_count_(sent.size(), 0),
    actions_(),
    action_contexts_(),
    sentence(sent),
    postags(),
    ctx_size{context_size},
    liw{0},
    lpw{0}
  {
    //sentence already includes root (0)
    for (int i = 0; i < (int)sentence.size(); ++i)
      buffer_[i] = sentence.size() - i - 1; 
  }

  TransitionParser(Words sent, Words posts, unsigned context_size):  
    stack_(),
    buffer_(sent.size()),
    arcs_(sent.size(), -1),
    child_count_(sent.size(), 0),
    actions_(),
    action_contexts_(),
    sentence(sent),
    postags(posts),
    ctx_size{context_size},
    liw{0},
    lpw{0}
  {
    //sentence already includes root (0)
    for (int i = 0; i < (int)sentence.size(); ++i)
      buffer_[i] = sentence.size() - i - 1; 
  }

  TransitionParser(unsigned context_size):  
    stack_(),
    buffer_(),
    arcs_(),
    child_count_(),
    actions_(),
    action_contexts_(),
    sentence(),
    postags(),
    ctx_size{context_size},
    liw{0},
    lpw{0}
  {
  }

  bool shift() {
    WordIndex i = buffer_.back();
    buffer_.pop_back();
    stack_.push_back(i);
    actions_.push_back(Action::sh);
    return true;
  }

  bool shift(WordId w) {
    WordIndex i = sentence.size();
    sentence.push_back(w);
    arcs_.push_back(-1);
    child_count_.push_back(0);
    stack_.push_back(i);
    actions_.push_back(Action::sh);
    return true;
  }

  //not currently used
  /*
  void reduce() {
    stack_.pop_back();
    actions_.push_back(Action:re);
    action_contexts_.push_back(context());
  } */

  virtual bool left_arc() = 0;

  virtual bool right_arc() = 0;

  virtual bool sentence_oracle(WxList gold_arcs) = 0;

  bool execute_action(Action a) {
    switch(a) {
    case Action::sh:
      return shift();
    case Action::la:
      return left_arc();
    case Action::ra:
      return right_arc();
    default: 
      //other cases not implemented
      std::cerr << "action not implemented" << std::endl;
      return false;
    }
  }

  std::string const actions_str() {
    const std::vector<std::string> action_names {"sh", "la", "ra", "re", "la2", "ra2"};
    std::string seq_str = "";
    for (Action a: actions_)
      seq_str += action_names[static_cast<int>(a)] + " ";
    return seq_str; 
  }

  
  void const print_arcs() {
    for (auto a: arcs_)
      std::cout << a << " ";
    //std::cout << std::endl;
  }

  void const print_sentence(Dict& dict) {
    for (auto a: sentence)
      std::cout << dict.Convert(a) << " ";
    std::cout << std::endl;
  }

  void const print_postags(Dict& dict) {
    for (auto a: postags)
      std::cout << dict.ConvertPOS(a) << " ";
    std::cout << std::endl;
  }

  void const print_action_contexts(Dict& dict) {
    for (auto a: action_contexts_) {
      std::cout << "(" << dict.Convert(sentence.at(a[0])) << " " << dict.Convert(sentence.at(a[1])) << ") ";
    }
    std::cout << std::endl;
  }

  //context for next word prediction
  WordsList const shift_contexts() {
    return extract_word_shift_contexts();
  } 

  WordsList const tag_contexts() {
    return extract_tag_contexts([&](Action a) {return a==Action::sh;});
  }

  //left arc or right arc
  ActList const arc_predictions() {
    ActList prd;
    for (auto a: actions_)
      if (a==Action::la || a==Action::ra) 
        prd.push_back(a);
    return prd;
  }
   
  WordsList const arc_contexts() {
    return extract_word_pos_contexts([&](Action a) {return (a==Action::la || a==Action::ra);});
  }

  //shift or reduce
  ActList const reduce_predictions() {
    ActList prd;
    for (auto a: actions_) {
      if (a==Action::sh) 
        prd.push_back(a);
      else if (a==Action::la || a==Action::ra) 
        prd.push_back(Action::re);
    }
    
    return prd;
  }

  unsigned const sentence_length() {
    return sentence.size();
  }

  unsigned const action_context_size() {
    return action_contexts_.size();
  }

  Words const get_sentence() {
    return sentence;
  }

  Words const get_postags() {
    return postags;
  }

  WxList const arcs() {
    return arcs_;
  }

  unsigned const get_context_size() {
    return ctx_size;
  }

  double const importance_weight() {
    return liw;
  }

  double const particle_weight() {
    return lpw;
  }

  void reset_importance_weight() {
    liw = 0;
  }

  void set_importance_weight(double w) {
    liw = -std::log(w);
  }

  void add_importance_weight(double w) {
    liw -= std::log(w);
  }

  void set_log_particle_weight(double w) {
    lpw = w;
  }

  void set_particle_weight(double w) {
    lpw = -std::log(w);
  }

  void add_particle_weight(double w) {
    lpw -= std::log(w);
  }

  //shift or left arc or right arc
  ActList const action_predictions() {
    ActList prd(actions_);  //copy to be consistent with other methods
    return prd;
  }
  
  //action and reduce have same context 
  WordsList const action_contexts() {
    return extract_word_pos_contexts([&](Action a) {return true;});
  }

  //word
  WordsList const extract_word_contexts(std::function<bool (Action)> pred) {
    WordsList cnts;
    for (unsigned i = 0; i < actions_.size(); ++i) {
      if (pred(actions_[i])) {
        Words cnt = Words(ctx_size);
        for (unsigned j = 0; j < ctx_size; ++j) {
          WordIndex k = action_contexts_[i][j];
          cnt[j] = sentence[k];
        }
        cnts.push_back(cnt);
      }  
    }
    return cnts; 
  }

  //word and pos for shift
  WordsList const extract_word_shift_contexts() {
    WordsList cnts;
    for (unsigned i = 0; i < actions_.size(); ++i) {
      if (actions_[i]==Action::sh) {
        Words cnt = Words(5); 
        cnt[0] = sentence.at(action_contexts_[i][0]);
        cnt[1] = postags.at(action_contexts_[i][0]);
        cnt[2] = sentence.at(action_contexts_[i][1]);
        cnt[3] = postags.at(action_contexts_[i][1]);
        cnt[4] = postags.at(cnts.size());
        cnts.push_back(cnt);
      }  
    }
    return cnts; 
  }

  //word and pos
  WordsList const extract_word_pos_contexts(std::function<bool (Action)> pred) {
    WordsList cnts;
    for (unsigned i = 0; i < actions_.size(); ++i) {
      if (pred(actions_[i])) {
        //ctx_size==2
        Words cnt = Words(ctx_size*2);
        for (unsigned j = 0; j < ctx_size; ++j) {
          WordIndex k = action_contexts_[i][j];
          cnt[j] = sentence[k];
          cnt[j+2] = postags[k];
        }
        cnts.push_back(cnt);
      }  
    }
    return cnts; 
  }

  //word (1) and pos (2)
  WordsList const extract_word_posl_contexts(std::function<bool (Action)> pred) {
    WordsList cnts;
    for (unsigned i = 0; i < actions_.size(); ++i) {
      if (pred(actions_[i])) {
        Words cnt = Words(3); 
        cnt[0] = sentence.at(action_contexts_[i][1]);
        cnt[1] = postags.at(action_contexts_[i][0]);
        cnt[2] = postags.at(action_contexts_[i][1]);
        cnts.push_back(cnt);
      }  
    }
    return cnts; 
  }

  WordsList const extract_tag_contexts(std::function<bool (Action)> pred) {
    WordsList cnts;
    for (unsigned i = 0; i < actions_.size(); ++i) {
      if (pred(actions_[i])) {
        Words cnt = Words(ctx_size);
        for (unsigned j = 0; j < ctx_size; ++j) {
          WordIndex k = action_contexts_[i][j];
          cnt[j] = postags[k];
        }
        cnts.push_back(cnt);
      }  
    }
    return cnts; 
  }
  
  WxList const context() {
    WxList ctx(ctx_size, 0);

    //hardcoded --all that works
    //if (stack_.size() >= 1)
    if (stack_.size() >= 2) {
      ctx[0] = stack_.at(stack_.size()-2);
      ctx[1] = stack_.at(stack_.size()-1);
    }

    //for (unsigned i = 1; (i <= stack_.size()) && (i <= ctx_size); ++i)
    // ctx[ctx_size-i] = stack_.at(stack_.size()-i);
    return ctx;
  }

  Words const word_context() {
    //return the whole stack
    if (stack_.size() < ctx_size) {
      Words ctx(ctx_size, 0);
      return ctx;        
    }
     
  Words ctx(stack_.size(), 0);
    for (unsigned i = 0; i < stack_.size(); ++i)
      ctx[i] = sentence[stack_[i]];
    return ctx;
  }

  Words const pos_context() {
    //return the whole stack
    if (stack_.size() < ctx_size) {
      Words ctx(ctx_size, 0);
      return ctx;        
    }
     
    Words ctx(stack_.size(), 0);
    for (unsigned i = 0; i < stack_.size(); ++i)
      ctx[i] = postags[stack_[i]];
    return ctx;
  }

  Words const word_pos_context() {
    Words ctx(2*ctx_size, 0);
    
    //ctx_size=2
    if (stack_.size() >= ctx_size) { 
      ctx[0] = sentence.at(stack_.at(stack_.size()-2));
      ctx[1] = sentence.at(stack_.at(stack_.size()-1));
      
      ctx[2] = postags.at(stack_.at(stack_.size()-2));
      ctx[3] = postags.at(stack_.at(stack_.size()-1));
    }

    return ctx;
  }

  Words const word_posl_context() {
          
    Words ctx(3, 0);
    //ctx_size=2
    if (stack_.size() >= ctx_size) { 
      ctx[0] = sentence.at(stack_.at(stack_.size()-1));
      ctx[1] = postags.at(stack_.at(stack_.size()-2));
      ctx[2] = postags.at(stack_.at(stack_.size()-1));
    }

    return ctx;
  }

  Words const word_posx_context() {
    Words ctx(5, 0);
    
    //ctx_size=2
    if ((stack_.size() >= ctx_size) && (buffer_.size() > 0)) { 
      ctx[0] = sentence.at(stack_.at(stack_.size()-2));
      ctx[1] = postags.at(stack_.at(stack_.size()-2));
      ctx[2] = sentence.at(stack_.at(stack_.size()-1));
      ctx[3] = postags.at(stack_.at(stack_.size()-1));
      ctx[4] = postags.at(buffer_.back());
    }

    return ctx;
  }
  
  ActList const actions() {
    return actions_;
  }
   
  unsigned const num_actions() {
    return actions_.size();
  }

  //number of children at sentence position i
  int const child_count(int i) {
    return child_count_[i];
  }

  bool const has_parent(int i) {
    return (arcs_[i] >= 0);
  }

  int const stack_depth() {
    return stack_.size();
  }

  int const buffer_length() {
    return buffer_.size();
  }

  WordIndex const stack_top() {
    return stack_.back();
  }

  WordId const next_word() {
    //std::cerr << buffer_.size() << " " << buffer_.back() << std::endl;
    return sentence[buffer_.back()];
  }

  WordId const next_tag() {
    return postags[buffer_.back()];
  }

  WordIndex const buffer_next() {
    return buffer_.back();
  }

  bool const is_buffer_empty() {
    return buffer_.empty();
  }

  bool const is_terminal_configuration() {
    return (buffer_.empty() && (stack_.size() == 1));
  }

  unsigned directed_accuracy_count(WxList g_arcs) {
    unsigned count = 0;
    for (unsigned j = 1; j < arcs_.size(); ++j) {
      if (arcs_[j]==g_arcs[j])
        ++count;
    }
    return count;
  }

  unsigned undirected_accuracy_count(WxList g_arcs) {
    unsigned count = 0;
    for (unsigned j = 1; j < arcs_.size(); ++j) {
      if ((arcs_[j]==g_arcs[j]) || (arcs_[j]>=0 && g_arcs[arcs_[j]]==static_cast<int>(j)))
        ++count;
    }
    return count;
  }
  
  protected:
  WxList stack_;
  WxList buffer_;
  WxList arcs_;
  std::vector<int> child_count_;
  ActList actions_;
  std::vector<WxList> action_contexts_;
  private:
  Words sentence;
  Words postags;
  unsigned ctx_size;  //const
  double liw; //log importance weight
  double lpw; //log particle weight
};


class ArcStandardParser : public TransitionParser {
  public:
  ArcStandardParser(Words sent, unsigned context_size):
    TransitionParser(sent, context_size) {
  }

  ArcStandardParser(Words sent, Words ptags, unsigned context_size):
    TransitionParser(sent, ptags, context_size) {
  }

  ArcStandardParser(unsigned context_size):
    TransitionParser(context_size) {
  }

  bool left_arc_valid() {
    if (stack_.size() < 2)
      return false;
    WordIndex i = stack_.rbegin()[1];
    return (i != 0);
  }

  bool left_arc() {
    WordIndex j = stack_.back();
    stack_.pop_back();
    WordIndex i = stack_.back();
    //check to ensure 0 is root
    if (i==0) {
      stack_.push_back(j);
      return false;
    }

    stack_.pop_back();
    stack_.push_back(j);
    arcs_[i] = j;
    ++child_count_[j];
    actions_.push_back(Action::la);
    return true;
  }

  bool right_arc() {
    WordIndex j = stack_.back();
    stack_.pop_back();
    WordIndex i = stack_.back();
    arcs_[j] = i;
    ++child_count_[i];
    actions_.push_back(Action::ra);
    return true;
  }

  //predict the next action according to the oracle, modified for evaluation and error analysis
  Action oracle_next(WxList gold_arcs, WxList prop_arcs) {
    std::vector<int> gold_child_count = count_children(gold_arcs);
    Action a = Action::re;

    //assume not in terminal configuration 
    if (stack_depth() < 2)
      a = Action::sh; 
    else {
      WordIndex i = stack_.rbegin()[1];
      WordIndex j = stack_.rbegin()[0];
      if (gold_arcs[i]==j && child_count_[i]>=gold_child_count[i]) {
        a = Action::la; 
        //if (prop_arcs[i]==j) 
        //  a = Action::la2;
      } else if (gold_arcs[j]==i && child_count_[j]>=gold_child_count[j]) { 
        a = Action::ra; 
        //if (prop_arcs[i]==j) 
        //  a = Action::ra2;
      } else if (!is_buffer_empty()) 
        a = Action::sh;
    }
   
    //return re if there shouldn't be an arc, but can't do anthing else either
    return a;
  }

  //predict the next action according to the oracle
  Action oracle_next(WxList gold_arcs) {
    std::vector<int> gold_child_count = count_children(gold_arcs);
    Action a = Action::re;

    //assume not in terminal configuration 
    if (stack_depth() < 2)
      a = Action::sh; 
    else {
      WordIndex i = stack_.rbegin()[1];
      WordIndex j = stack_.rbegin()[0];
      if (gold_arcs[i]==j && child_count_[i]==gold_child_count[i]) 
        a = Action::la;
      else if (gold_arcs[j]==i && child_count_[j]==gold_child_count[j]) 
        a = Action::ra;
      else if (!is_buffer_empty()) 
        a = Action::sh;
    }
    
    return a;
  }

  bool sentence_oracle(WxList gold_arcs) {
     
    std::vector<int> gold_child_count = count_children(gold_arcs);
     
    bool is_stuck = false;
    while (!is_terminal_configuration() && !is_stuck) {
      action_contexts_.push_back(context());
      if (stack_depth() < 2) 
        shift();
      else {
        WordIndex i = stack_.rbegin()[1];
        WordIndex j = stack_.rbegin()[0];
        if (gold_arcs[i]==j && child_count_[i]==gold_child_count[i]) 
          left_arc();
        else if (gold_arcs[j]==i && child_count_[j]==gold_child_count[j]) 
          right_arc();
        else if (!is_buffer_empty()) 
          shift();
        else
          is_stuck = true; 
      }
    }

    if (!is_stuck && arcs_!=gold_arcs)
      is_stuck = true;
    return !is_stuck;
  }

  //count when shifted/reduced when it should have shifted/reduced
  void count_accuracy(AccuracyCounts& acc_counts, WxList g_arcs) {
    //for reduce: if there exists an arc it should be added...
    //Unless we know that it is added later in the given parse

    //resimulate the computation of the action sequence to compute accuracy
    ArcStandardParser simul(get_sentence(), get_postags(), get_context_size());
    
    for (auto& a: actions_) {
      Action next = simul.oracle_next(g_arcs, arcs_);
      if (next==Action::sh) {
        acc_counts.inc_shift_gold();
        if (a==Action::sh)
          acc_counts.inc_shift_count();
      } else if (next==Action::la || next==Action::ra) {
        acc_counts.inc_reduce_gold();
        if (a==Action::la || a==Action::ra) //counts either direction
          acc_counts.inc_reduce_count();
      } /*  else if (next==Action::la2 || next==Action::ra2) {
        //action taken is considered the gold action
        if (a==Action::sh) {
          acc_counts.inc_shift_count();
          acc_counts.inc_shift_gold();
        } else {
          acc_counts.inc_reduce_count();
          acc_counts.inc_reduce_gold();
        }
      } */

      if (simul.is_buffer_empty() && next==Action::re)
        acc_counts.inc_final_reduce_error_count();
      
      simul.execute_action(a);
    }

    acc_counts.inc_num_sentences();
    if (g_arcs==arcs_)
      acc_counts.inc_complete_sentences();
    for (unsigned i = 1; i < arcs_.size(); ++i)
      if (arcs_[i]==0 && g_arcs[i]==0) 
        acc_counts.inc_root_count();

    acc_counts.add_total_length(g_arcs.size() - 1);
    acc_counts.add_directed_count(directed_accuracy_count(g_arcs));
    acc_counts.add_undirected_count(undirected_accuracy_count(g_arcs));
    //TODO compute root accuracy 
  }

};

inline bool cmp_importance_weights(ArcStandardParser p1, ArcStandardParser p2) {
  return (p1.importance_weight() < p2.importance_weight());
}
 
inline bool cmp_particle_weights(ArcStandardParser p1, ArcStandardParser p2) {
  return (p1.particle_weight() < p2.particle_weight());
}

inline bool cmp_particle_ptr_weights(const std::unique_ptr<ArcStandardParser>& p1, const std::unique_ptr<ArcStandardParser>& p2) {
  return (p1->particle_weight() < p2->particle_weight());
}

inline bool cmp_particle_importance_weights(ArcStandardParser p1, ArcStandardParser p2) {
  return ((p1.particle_weight() + p1.importance_weight()) < (p2.particle_weight() + p2.importance_weight()));
}

inline bool cmp_normalized_particle_weights(ArcStandardParser p1, ArcStandardParser p2) {
  return ((p1.particle_weight() / p1.num_actions()) < (p2.particle_weight() / p2.num_actions()));
}

inline bool cmp_rnorm_particle_weights(ArcStandardParser p1, ArcStandardParser p2) {
  return ((p1.particle_weight() / (2*p1.sentence_length() - p1.stack_depth())) < (p2.particle_weight() / (2*p2.sentence_length() - p2.stack_depth())));
}

inline Action convert_to_action(WordId a) {
  std::vector<Action> actList = {Action::sh, Action::la, Action::ra, Action::re, Action::la2, Action::ra2};
  return actList[a];
}

inline bool is_projective_dependency(WxList g_arcs) {
  for (int i = 0; i < static_cast<int>(g_arcs.size() - 1); ++i)
    for (int j = i + 1; j < static_cast<int>(g_arcs.size()); ++j)
      if ((g_arcs[i]<i &&
            (g_arcs[j]<i && g_arcs[j]>g_arcs[i])) ||
          ((g_arcs[i]>i && g_arcs[i]>j) &&
            (g_arcs[j]<i || g_arcs[j]>g_arcs[i])) ||
          ((g_arcs[i]>i && g_arcs[i]<j) &&
            (g_arcs[j]>i && g_arcs[j]<g_arcs[i])))
        return false;
    return true;
}

}
#endif
