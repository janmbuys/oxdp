#ifndef _CGDP_SRP_H_
#define _CGDP_SRP_H_

#include<string>
#include<functional>

#include<corpus/corpus.h>

namespace oxlm {

enum class Action : WordId {sh, la, ra, re, la2, ra2};

typedef int WordIndex;
typedef std::vector<WordIndex> WxList;
typedef std::vector<Words> WordsList;
typedef std::vector<WxList> WxCtxList;
typedef std::vector<Action> ActList;

inline Action convert_to_action(WordId a) {
    std::vector<Action> actList = {Action::sh, Action::la, Action::ra, Action::re, Action::la2, Action::ra2};
    return actList[a];
}

class TransitionParser {
  public:
    TransitionParser(Words sent, unsigned context_size):  
      stack_(),
      buffer_(sent.size()+1),
      arcs_(sent.size()+1, -1),
      child_count_(sent.size()+1, 0),
      actions_(),
      action_contexts_(),
      sentence(1, 0),
      ctx_size{context_size}
    {
        //copy of sent, position 0 is 0
        sentence.reserve(sent.size()+1);
        sentence.insert(sentence.end(), sent.begin(), sent.end());

        for (int i = 0; i < (int)sentence.size(); ++i)
            buffer_[i] = sentence.size()-i-1; 
    }

    TransitionParser(unsigned context_size):  
      stack_(),
      buffer_(),
      arcs_(),
      child_count_(),
      actions_(),
      action_contexts_(),
      sentence(),
      ctx_size{context_size}
    {
    }

    //don't yet see this necessary
    //~TransitionParser() { }

    bool shift() {
        //std::cerr << "buffer size: " << buffer_.size() << std::endl;
        WordIndex i = buffer_.back();
        buffer_.pop_back();
        stack_.push_back(i);
        actions_.push_back(Action::sh);
        WxList ctx = context();
        //std::cerr << "context size: " << ctx.size() << std::endl;
        action_contexts_.push_back(ctx);
        return true;
    }

    bool shift(WordId w) {
        //std::cerr << "buffer size: " << buffer_.size() << std::endl;
        WordIndex i = sentence.size();
        sentence.push_back(w);
        arcs_.push_back(-1);
        child_count_.push_back(0);
        
        stack_.push_back(i);
        actions_.push_back(Action::sh);
        WxList ctx = context();
        //std::cerr << "context size: " << ctx.size() << std::endl;
        action_contexts_.push_back(ctx);
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
            //std::cerr << "shift" << std::endl;
            return shift();
        case Action::la:
            //std::cerr << "left arc" << std::endl;
            return left_arc();
        case Action::ra:
            //std::cerr << "right arc" << std::endl;
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
        std::cout << std::endl;
    }

    void const print_sentence(Dict& dict) {
        for (auto a: sentence)
            std::cout << dict.Convert(a) << " ";
        std::cout << std::endl;
    }

    //context for next word prediction
    WordsList const shift_contexts() {
        return extract_contexts([&](Action a) {return a==Action::sh;});
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
        return extract_contexts([&](Action a) {return (a==Action::la || a==Action::ra);});
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

    //shift or left arc or right arc
    ActList const action_predictions() {
        ActList prd(actions_);  //copy to be consistent with other methods
        return prd;
    }
  
    //action and reduce have same context 
    WordsList const action_contexts() {
        return extract_contexts([&](Action a) {return true;});
    }

    WordsList const extract_contexts(std::function<bool (Action)> pred) {
        WordsList cnts;
        for (unsigned i = 0; i < actions_.size(); ++i)
            if (pred(actions_[i])) {
                Words cnt = Words(ctx_size);
                for (unsigned j = 0; j < ctx_size; ++j) {
                    WordIndex k = action_contexts_[i][j];
                    cnt[j] = sentence[k];
                }
                cnts.push_back(cnt);
            }
        return cnts; 
    }

    WxList const context() {
        WxList ctx(ctx_size, 0);
        for (unsigned i = 0; (i < stack_.size()) && (i < ctx_size); ++i)
            ctx.rbegin()[i] = stack_.rbegin()[i];
        return ctx;
    }

    Words const word_context() {
        Words ctx(ctx_size, 0);
        for (unsigned i = 0; (i < stack_.size()) && (i < ctx_size); ++i)
            ctx.rbegin()[i] = sentence[stack_.rbegin()[i]];
        return ctx;
    }

    ActList const actions() {
      return actions_;
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

    WordIndex const stack_top() {
        return stack_.back();
    }

    WordId const next_word() {
        return sentence[buffer_.back()];
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

    static std::vector<int> const count_children(WxList g_arcs) {
        std::vector<int> g_child_count(g_arcs.size(), 0);
        for (unsigned j = 1; j < g_arcs.size(); ++j)
            ++g_child_count[j];
        return g_child_count;
    } 

  protected:
    WxList stack_;
    WxList buffer_;
    WxList arcs_;
    std::vector<int> child_count_;
    ActList actions_;
    WxCtxList action_contexts_;

  private:
    Words sentence;
    const unsigned ctx_size;
};

class ArcStandardParser : public TransitionParser {
  public:
    ArcStandardParser(Words sent, unsigned context_size):
      TransitionParser(sent, context_size) {
        //can assert context_size
    }

    ArcStandardParser(unsigned context_size):
      TransitionParser(context_size) {
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
        action_contexts_.push_back(context());
        return true;
    }

    bool right_arc() {
        WordIndex j = stack_.back();
        stack_.pop_back();
        WordIndex i = stack_.back();
        arcs_[j] = i;
        ++child_count_[i];
        actions_.push_back(Action::ra);
        action_contexts_.push_back(context());
        return true;
    }

    bool sentence_oracle(WxList gold_arcs) {
        std::vector<int> gold_child_count = count_children(gold_arcs);
        bool is_stuck = false;
        while (!is_terminal_configuration() && !is_stuck) {
            if (stack_depth() < 2) 
                shift();
            else {
                WordIndex i = stack_.rbegin()[1];
                WordIndex j = stack_.rbegin()[0];
                if (gold_arcs[i]==j && child_count_[i]==gold_child_count[i]) 
                    left_arc();
                else if (gold_arcs[j]==i && child_count_[j]==gold_child_count[j]) 
                    right_arc();
                else
                   is_stuck = true; 
            }
        }
        if (!is_stuck && arcs_!=gold_arcs)
            is_stuck = true;
        return !is_stuck;
    }
};
 
}

#endif
