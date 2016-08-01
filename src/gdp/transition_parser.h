#ifndef _GDP_TR_PARSER_H_
#define _GDP_TR_PARSER_H_

#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "utils/random.h"
#include "corpus/dict.h"
#include "corpus/model_config.h"
#include "corpus/parse_data_set.h"
#include "gdp/utils.h"
#include "gdp/parser.h"

namespace oxlm {

// Class implementing standard operations of a transition-based parser. 
class TransitionParser : public Parser {
 public:
  // Constructor used for generating sentences.
  TransitionParser(const boost::shared_ptr<ModelConfig>& config);

  TransitionParser(const TaggedSentence& parse,
                   const boost::shared_ptr<ModelConfig>& config);

  TransitionParser(const TaggedSentence& parse, int num_particles,
                   const boost::shared_ptr<ModelConfig>& config);

  void pop_buffer() { ++buffer_next_; }

  void pop_stack() { stack_.pop_back(); }

  void push_stack(WordIndex i) { stack_.push_back(i); }

  void append_action(kAction a) { actions_.push_back(a); }

  void push_tag(WordId tag) {
    TaggedSentence::push_tag(tag, config_->tag_to_feature[tag]);
  }

  void update_tag(WordIndex i, WordId tag) {
    if (config_->pyp_model) {
      set_tag_at(i, tag, tag);
    } else {
      set_tag_at(i, tag, config_->tag_to_feature[tag]);
    }
  }

  void append_action_label(WordId l) { action_labels_.push_back(l); }

  void reset_importance_weight() { importance_weight_ = 0; }

  void set_importance_weight(Real w) { importance_weight_ = w; }

  void add_importance_weight(Real w) { importance_weight_ += w; }

  void set_particle_weight(Real w) { set_weight(w); }

  void add_particle_weight(Real w) { add_weight(w); }

  void add_log_particle_weight(Real w) {
    if (weight() == 0) {
      set_weight(w);
    } else {
      set_weight(neg_log_sum_exp(weight(), w));  // add in log space
    }
  }

  void add_beam_weight(Real w) {
    if (beam_weight_ == 0) {
      beam_weight_ = w;
    } else {
      beam_weight_ = neg_log_sum_exp(beam_weight_, w);  // add in log space
    }
  }

  void set_marginal_weight(Real w) {
    marginal_weight_ = w;
  }

  void set_num_particles(int n) { num_particles_ = n; }

  int stack_depth() const { return stack_.size(); }

  bool stack_empty() const { return stack_.empty(); }

  WordIndex stack_top() const { return stack_.back(); }

  WordIndex stack_top_second() const { return stack_.rbegin()[1]; }

  WordIndex stack_top_third() const { return stack_.rbegin()[2]; }

  WordIndex buffer_next() const {
    if (!root_first() && (buffer_next_ == static_cast<int>(size()))) {
      return 0;
    } else {
      return buffer_next_;
    }
  }

  bool buffer_empty() const {
    // For root-last, the root occurs after the end of sentence.
    if (root_first()) {
      return (buffer_next_ >= static_cast<int>(size()));
    } else {
      return (buffer_next_ > static_cast<int>(size()));
    }
  }

  WordId next_word() const { return word_at(buffer_next()); }

  WordId next_tag() const { return tag_at(buffer_next()); }

  ActList actions() const { return actions_; }

  kAction last_action() const { return actions_.back(); }

  unsigned num_actions() const { return actions_.size(); }

  int num_labels() const { return config_->num_labels; }

  bool root_first() const { return config_->root_first; }

  bool predict_pos() const { return config_->predict_pos; }

  bool non_projective() const {
    return (config_->parser_type == ParserType::arcstandard2);
  }

  std::vector<std::string> action_str_list() const {
    const std::vector<std::string> action_names{"sh", "la",  "ra",
                                                "re", "la2", "ra2"};
    std::vector<std::string> list;
    for (kAction a : actions_) {
      list.push_back(action_names[static_cast<int>(a)]);
    }
    return list;
  }

  void print_actions() const {
    for (auto act : action_str_list()) {
      std::cout << act << " ";
    }
    std::cout << std::endl;
  }

  WordId action_label_at(int i) const { return action_labels_[i]; }

  Real particle_weight() const { return weight(); }

  Real importance_weight() const { return importance_weight_; }

  Real beam_weight() const { return beam_weight_; }

  Real marginal_weight() const { return marginal_weight_; }

  int num_particles() const { return num_particles_; }

  Real weighted_particle_weight() const {
    return (weight() - std::log(num_particles_));
  }

  Real weighted_importance_weight() const {
    return (importance_weight_ - std::log(num_particles_));
  }

  boost::shared_ptr<ModelConfig> config() const {
    return boost::shared_ptr<ModelConfig>(config_);
  }

  bool pyp_model() const { return config_->pyp_model; }

  bool lexicalised() const { return config_->lexicalised; }

  WordId get_word_feature(WordId w) const {
    return config_->getWordFeatures(w)[0];
  }

  std::string context_type() const { return config_->context_type; }

  WordId convert_action(kAction a, WordId l) const {
    if (a == kAction::sh)
      return 0;
    else if (a == kAction::la)
      return l + 1;
    else if (a == kAction::ra)
      return num_labels() + l + 1;
    else if (a == kAction::re)  // if valid in transition system
      return 2 * num_labels() + 1;
    else if (a == kAction::la2)
      return 2 * num_labels() + l + 1;
    else if (a == kAction::ra2)
      return 3 * num_labels() + l + 1;
    else
      return -1;
  }

  // Maps a list of indices to word and feature values.
  Context map_context(Indices ind) const {
    Words words;
    WordsList features;
    for (auto i : ind) {
      if (i >= 0) {
        if (config_->lexicalised) {
          words.push_back(word_at(i));
        } else {
          words.push_back(tag_at(i));
        }
        Words feats(features_at(i));

        if (config_->label_features) {
          feats.push_back(config_->label_feature_index + label_at(i));
        }

        if (config_->distance_features) {
          size_t range = config_->distance_range;
          // Left, right valency.
          feats.push_back(config_->distance_feature_index +
                          std::min(range - 1, left_child_count_at(i)));
          feats.push_back(config_->distance_feature_index + range +
                          std::min(range - 1, right_child_count_at(i)));
          // Distance to buffer_next.
          feats.push_back(config_->distance_feature_index + 2 * range - 1 +
                          std::min((int)(range), buffer_next_ - i));
          // Distance to head.
          if (arc_at(i) >= 0) {
            feats.push_back(config_->distance_feature_index + 3 * range - 1 +
                            std::min((int)(range), std::abs(arc_at(i) - i)));
          }
        }

        features.push_back(feats);
      } else {
        // Ensure that the number of context elements stays consistent.
        words.push_back(0);
        if (config_->lexicalised && config_->predict_pos) {
          features.push_back(Words(2, 0));
        } else {
          features.push_back(Words(1, 0));
        }
      }
    }

    if (config_->predict_pos) features.push_back(Words());
    return Context(words, features);
  }

  //TODO Remove some unnecessary functions here.
  /* Functions for extracting context vectors.  */

  Context stack_action_context() const {
    WordsList features;

    // Stack features.
    for (int i = static_cast<int>(stack_.size()) - config_->stack_ctx_size;
         i < static_cast<int>(stack_.size()); ++i) {
      Indices ctx;
      WordIndex pos = -1;
      if (i >= 0) pos = stack_.at(i);
      ctx.push_back(pos);
      bool allow_children = (static_cast<int>(stack_.size()) - i == 1) ||
                            (static_cast<int>(stack_.size()) - i == 2);

      if (allow_children && (config_->child_ctx_level >= 1)) {
        ctx.push_back(leftmost_child_at(pos));
        ctx.push_back(rightmost_child_at(pos));
      }
      if (allow_children && (config_->child_ctx_level >= 2)) {
        ctx.push_back(second_leftmost_child_at(pos));
        ctx.push_back(second_rightmost_child_at(pos));
      }
      if (allow_children && (config_->child_ctx_level >= 3)) {
        ctx.push_back(leftmost_grandchild_at(pos));
        ctx.push_back(rightmost_grandchild_at(pos));
      }

      for (WordIndex ctx_pos : ctx) {
        if (ctx_pos >= 0) {
          features.push_back(Words(1, features_at(ctx_pos)[1]));  // word
          features.push_back(Words(1, features_at(ctx_pos)[0]));  // tag
          if (config_->label_features) {
            features.back().push_back(config_->label_feature_index +
                                      label_at(ctx_pos));
          }
        } else {
          features.push_back(Words(1, 0));
          features.push_back(Words(1, 0));
        }
      }
    }

    // Action features.
    for (int i = static_cast<int>(actions_.size()) - config_->action_ctx_size;
         i < static_cast<int>(actions_.size()); ++i) {
      if (i >= 0) {
        features.push_back(
            Words(1, config_->action_feature_index +
                         convert_action(actions_[i], action_label_at(i))));
      } else {
        features.push_back(Words(1, config_->action_feature_index - 1));
      }
    }

    if (config_->predict_pos) features.push_back(Words());

    return Context(Words(), features);
  }

  Words action_only_context() const {
    int ctx_size = 6;
    Words features(ctx_size, 0);

    int sh_pos = buffer_next_ - 1;
    for (int i = 0; i < ctx_size; ++i) {
      int pos = static_cast<int>(actions_.size()) - ctx_size + i;
      if (pos >= 0) {
        if (actions_[i] == kAction::sh) {
          features[i] = config_->numActions() + 1 + tag_at(sh_pos);
          --sh_pos;
        } else {
          features[i] = convert_action(actions_[pos], action_label_at(pos)) + 1;
        }
      }
    }

    return features;
  }

  Words word_next_children_context() const {
    Words ctx(5, 0);

    if (stack_.size() >= 1) {
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size() - 1));

      ctx[1] = word_at(stack_.at(stack_.size() - 1));
      if (l1 >= 0) {
        ctx[2] = tag_at(l1);
      }
      if (r1 >= 0) {
        ctx[3] = tag_at(r1);
      }
    }

    if (stack_.size() >= 2) {
      ctx[0] = word_at(stack_.at(stack_.size() - 2));
    }

    if (!buffer_empty()) {
      ctx[4] = tag_at(buffer_next());
    }

    return ctx;
  }

  Words word_tag_next_children_context() const {
    Words ctx(5, 0);

    if (stack_.size() >= 1) {
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size() - 1));

      ctx[1] = word_at(stack_.at(stack_.size() - 1));
      if (l1 >= 0) {
        ctx[2] = tag_at(l1);
      }
      if (r1 >= 0) {
        ctx[3] = tag_at(r1);
      }
    }

    if (stack_.size() >= 2) {
      ctx[0] = word_at(stack_.at(stack_.size() - 2));
    }

    if (!buffer_empty()) {
      ctx[4] = tag_at(buffer_next());
    }

    return ctx;
  }

  Words word_tag_next_ngram_context() const {
    Words ctx(4, 0);

    WordIndex i = buffer_next();

    if (!buffer_empty()) {
      ctx[3] = tag_at(i);
    }

    if (i >= 1) {
      ctx[2] = word_at(i - 1);
    }
    if (i >= 2) {
      ctx[1] = word_at(i - 2);
    }
    if (i >= 3) {
      ctx[0] = word_at(i - 3);
    }

    return ctx;
  }

  Indices children_lookahead_context() const {
    Indices ctx(9, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[3] = stack_.at(stack_.size() - 2);
      ctx[4] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[5] = leftmost_child_at(stack_.at(stack_.size() - 2));
    }

    if (buffer_next() < size()) ctx[6] = buffer_next();
    if (buffer_next() + 1 < size()) ctx[7] = buffer_next() + 1;
    if (buffer_next() + 2 < size()) ctx[8] = buffer_next() + 2;

    return ctx;
  }

  Indices children_context() const {
    Indices ctx(6, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[3] = stack_.at(stack_.size() - 2);
      ctx[4] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[5] = leftmost_child_at(stack_.at(stack_.size() - 2));
    }

    return ctx;
  }

  Indices third_children_context() const {
    Indices ctx(9, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[3] = stack_.at(stack_.size() - 2);
      ctx[4] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[5] = leftmost_child_at(stack_.at(stack_.size() - 2));
    }
    if (stack_.size() >= 3) {
      ctx[3] = stack_.at(stack_.size() - 3);
      ctx[4] = rightmost_child_at(stack_.at(stack_.size() - 3));
      ctx[5] = leftmost_child_at(stack_.at(stack_.size() - 3));
    }

    return ctx;
  }

  Indices children_ngram_context() const {
    Indices ctx(9, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[3] = stack_.at(stack_.size() - 2);
      ctx[4] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[5] = leftmost_child_at(stack_.at(stack_.size() - 2));
    }
    if (buffer_next_ == static_cast<int>(size()) + 1)
      ctx[6] = 0;
    else if (buffer_next_ >= 1)
      ctx[6] = buffer_next_ - 1;
    if (buffer_next_ >= 2) ctx[7] = buffer_next_ - 2;
    if (buffer_next_ >= 3) ctx[8] = buffer_next_ - 3;

    return ctx;
  }

  Indices extended_children_context() const {
    Indices ctx(12, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[3] = second_rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[5] = stack_.at(stack_.size() - 2);
      ctx[6] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[7] = leftmost_child_at(stack_.at(stack_.size() - 2));
      ctx[8] = second_rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[9] = second_leftmost_child_at(stack_.at(stack_.size() - 2));
    }

    if (stack_.size() >= 3) {
      ctx[10] = stack_.at(stack_.size() - 3);
    }
    if (stack_.size() >= 4) {
      ctx[11] = stack_.at(stack_.size() - 4);
    }

    return ctx;
  }

  Indices third_extended_children_context() const {
    Indices ctx(17, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[3] = second_rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[5] = stack_.at(stack_.size() - 2);
      ctx[6] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[7] = leftmost_child_at(stack_.at(stack_.size() - 2));
      ctx[8] = second_rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[9] = second_leftmost_child_at(stack_.at(stack_.size() - 2));
    }
    if (stack_.size() >= 3) {
      ctx[10] = stack_.at(stack_.size() - 3);
      ctx[11] = rightmost_child_at(stack_.at(stack_.size() - 3));
      ctx[12] = leftmost_child_at(stack_.at(stack_.size() - 3));
      ctx[13] = second_rightmost_child_at(stack_.at(stack_.size() - 3));
      ctx[14] = second_leftmost_child_at(stack_.at(stack_.size() - 3));
    }

    if (stack_.size() >= 4) {
      ctx[15] = stack_.at(stack_.size() - 4);
    }
    if (stack_.size() >= 5) {
      ctx[16] = stack_.at(stack_.size() - 5);
    }

    return ctx;
  }

  Indices extended_children_ngram_context() const {
    Indices ctx(15, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[3] = second_rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[5] = stack_.at(stack_.size() - 2);
      ctx[6] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[7] = leftmost_child_at(stack_.at(stack_.size() - 2));
      ctx[8] = second_rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[9] = second_leftmost_child_at(stack_.at(stack_.size() - 2));
    }

    if (stack_.size() >= 3) {
      ctx[10] = stack_.at(stack_.size() - 3);
    }
    if (stack_.size() >= 4) {
      ctx[11] = stack_.at(stack_.size() - 4);
    }

    if (buffer_next_ == static_cast<int>(size()) + 1)
      ctx[12] = 0;
    else if (buffer_next_ >= 1)
      ctx[12] = buffer_next_ - 1;
    if (buffer_next_ >= 2) ctx[13] = buffer_next_ - 2;
    if (buffer_next_ >= 3) ctx[14] = buffer_next_ - 3;

    return ctx;
  }

  Indices extended_children_lookahead_context() const {
    Indices ctx(15, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[3] = second_rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[5] = stack_.at(stack_.size() - 2);
      ctx[6] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[7] = leftmost_child_at(stack_.at(stack_.size() - 2));
      ctx[8] = second_rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[9] = second_leftmost_child_at(stack_.at(stack_.size() - 2));
    }

    if (stack_.size() >= 3) {
      ctx[10] = stack_.at(stack_.size() - 3);
    }
    if (stack_.size() >= 4) {
      ctx[11] = stack_.at(stack_.size() - 4);
    }

    if (buffer_next() < size()) ctx[12] = buffer_next();
    if (buffer_next() + 1 < size()) ctx[13] = buffer_next() + 1;
    if (buffer_next() + 2 < size()) ctx[14] = buffer_next() + 2;

    return ctx;
  }

  Indices more_extended_children_ngram_context() const {
    Indices ctx(20, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[3] = second_rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[5] = rightmost_grandchild_at(stack_.at(stack_.size() - 1));
      ctx[6] = leftmost_grandchild_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[7] = stack_.at(stack_.size() - 2);
      ctx[8] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[9] = leftmost_child_at(stack_.at(stack_.size() - 2));
      ctx[10] = second_rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[11] = second_leftmost_child_at(stack_.at(stack_.size() - 2));
      ctx[12] = rightmost_grandchild_at(stack_.at(stack_.size() - 2));
      ctx[13] = leftmost_grandchild_at(stack_.at(stack_.size() - 2));
    }

    if (stack_.size() >= 3) {
      ctx[14] = stack_.at(stack_.size() - 3);
    }
    if (stack_.size() >= 4) {
      ctx[15] = stack_.at(stack_.size() - 4);
    }

    if (buffer_next_ == static_cast<int>(size()) + 1)
      ctx[16] = 0;
    else if (buffer_next_ >= 1)
      ctx[17] = buffer_next_ - 1;
    if (buffer_next_ >= 2) ctx[18] = buffer_next_ - 2;
    if (buffer_next_ >= 3) ctx[19] = buffer_next_ - 3;

    return ctx;
  }

  Indices more_extended_children_context() const {
    Indices ctx(16, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[3] = second_rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[5] = rightmost_grandchild_at(stack_.at(stack_.size() - 1));
      ctx[6] = leftmost_grandchild_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[7] = stack_.at(stack_.size() - 2);
      ctx[8] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[9] = leftmost_child_at(stack_.at(stack_.size() - 2));
      ctx[10] = second_rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[11] = second_leftmost_child_at(stack_.at(stack_.size() - 2));
      ctx[12] = rightmost_grandchild_at(stack_.at(stack_.size() - 2));
      ctx[13] = leftmost_grandchild_at(stack_.at(stack_.size() - 2));
    }

    if (stack_.size() >= 3) {
      ctx[14] = stack_.at(stack_.size() - 3);
    }
    if (stack_.size() >= 4) {
      ctx[15] = stack_.at(stack_.size() - 4);
    }

    return ctx;
  }

  Indices third_more_extended_children_context() const {
    Indices ctx(23, -1);
    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      ctx[1] = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[2] = leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[3] = second_rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[5] = rightmost_grandchild_at(stack_.at(stack_.size() - 1));
      ctx[6] = leftmost_grandchild_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[7] = stack_.at(stack_.size() - 2);
      ctx[8] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[9] = leftmost_child_at(stack_.at(stack_.size() - 2));
      ctx[10] = second_rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[11] = second_leftmost_child_at(stack_.at(stack_.size() - 2));
      ctx[12] = rightmost_grandchild_at(stack_.at(stack_.size() - 2));
      ctx[13] = leftmost_grandchild_at(stack_.at(stack_.size() - 2));
    }
    if (stack_.size() >= 3) {
      ctx[14] = stack_.at(stack_.size() - 3);
      ctx[15] = rightmost_child_at(stack_.at(stack_.size() - 3));
      ctx[16] = leftmost_child_at(stack_.at(stack_.size() - 3));
      ctx[17] = second_rightmost_child_at(stack_.at(stack_.size() - 3));
      ctx[18] = second_leftmost_child_at(stack_.at(stack_.size() - 3));
      ctx[19] = rightmost_grandchild_at(stack_.at(stack_.size() - 3));
      ctx[20] = leftmost_grandchild_at(stack_.at(stack_.size() - 3));
    }

    if (stack_.size() >= 4) {
      ctx[21] = stack_.at(stack_.size() - 4);
    }
    if (stack_.size() >= 5) {
      ctx[22] = stack_.at(stack_.size() - 5);
    }

    return ctx;
  }

  Words tag_children_third_context() const {
    Words ctx(10, 0);
    if (stack_.size() >= 1) {
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size() - 1));
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));

      ctx[9] = tag_at(stack_.at(stack_.size() - 1));
      if (l1 >= 0) ctx[4] = tag_at(l1);  //
      if (r1 >= 0) ctx[6] = tag_at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex l2 = leftmost_child_at(stack_.at(stack_.size() - 2));
      WordIndex r2 = rightmost_child_at(stack_.at(stack_.size() - 2));

      ctx[8] = tag_at(stack_.at(stack_.size() - 2));
      if (l2 >= 0) ctx[2] = tag_at(l2);
      if (r2 >= 0) ctx[5] = tag_at(r2);  //
    }

    if (stack_.size() >= 3) {
      WordIndex l3 = leftmost_child_at(stack_.at(stack_.size() - 3));
      WordIndex r3 = rightmost_child_at(stack_.at(stack_.size() - 3));

      ctx[7] = tag_at(stack_.at(stack_.size() - 3));
      if (l3 >= 0) ctx[1] = tag_at(l3);
      if (r3 >= 0) ctx[3] = tag_at(r3);
    }

    if (stack_.size() >= 4) {
      ctx[0] = tag_at(stack_.at(stack_.size() - 4));
    }

    return ctx;
  }

  //TODO Check which order this should be.
  Words tag_children_context() const {
    Words ctx(8, 0);
    if (stack_.size() >= 1) {
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size() - 1));
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));

      ctx[7] = tag_at(stack_.at(stack_.size() - 1));
      if (l1 >= 0) ctx[4] = tag_at(l1);
      if (r1 >= 0) ctx[5] = tag_at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex l2 = leftmost_child_at(stack_.at(stack_.size() - 2));
      WordIndex r2 = rightmost_child_at(stack_.at(stack_.size() - 2));

      ctx[6] = tag_at(stack_.at(stack_.size() - 2));
      // if (l2 >= 0)
      //  ctx[1] = tag_at(l2);
      if (r2 >= 0) ctx[2] = tag_at(r2);
    }
    if (stack_.size() >= 3) {
      ctx[3] = tag_at(stack_.at(stack_.size() - 3));
    }
    if (stack_.size() >= 4) {
      // ctx[0] = tag_at(stack_.at(stack_.size()-4));
    }
    return ctx;
  }

  Words word_tag_children_local_context() const {
    Words ctx(8, 0);
    if (stack_.size() >= 1) {
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size() - 1));
      WordIndex ll2 = second_leftmost_child_at(stack_.at(stack_.size() - 1));

      ctx[7] = tag_at(stack_.at(stack_.size() - 1));
      ctx[1] = word_at(stack_.at(stack_.size() - 1));
      if (l1 >= 0) ctx[4] = tag_at(l1);
      if (r1 >= 0) ctx[5] = tag_at(r1);
    }
    if (stack_.size() >= 2) {
      WordIndex r2 = rightmost_child_at(stack_.at(stack_.size() - 2));
      WordIndex l2 = leftmost_child_at(stack_.at(stack_.size() - 2));
      WordIndex rr2 = second_rightmost_child_at(stack_.at(stack_.size() - 2));

      ctx[6] = tag_at(stack_.at(stack_.size() - 2));
      ctx[0] = word_at(stack_.at(stack_.size() - 2));
      if (l2 >= 0) ctx[2] = tag_at(l2);
      if (r2 >= 0) ctx[3] = tag_at(r2);
    }
    
    return ctx;
  }

  Words word_tag_children_context() const {
    Words ctx(8, 0);
    if (stack_.size() >= 1) {
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size() - 1));

      ctx[7] = tag_at(stack_.at(stack_.size() - 1));
      ctx[1] = word_at(stack_.at(stack_.size() - 1));
      if (l1 >= 0) ctx[4] = tag_at(l1);
      if (r1 >= 0) {
        ctx[5] = tag_at(r1);
      }
    }
    if (stack_.size() >= 2) {
      WordIndex r2 = rightmost_child_at(stack_.at(stack_.size() - 2));
      WordIndex l2 = leftmost_child_at(stack_.at(stack_.size() - 2));

      ctx[6] = tag_at(stack_.at(stack_.size() - 2));
      ctx[0] = word_at(stack_.at(stack_.size() - 2));
      if (r2 >= 0) ctx[2] = tag_at(r2);
    }
    if (stack_.size() >= 3) {
      ctx[3] = tag_at(stack_.at(stack_.size() - 3));
    }

    return ctx;
  }

  Indices next_children_context() const {
    Indices ctx(8, -1);

    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));
      if (r1 != buffer_next()) ctx[1] = r1;
      ctx[2] = second_rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[3] = leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[4] = arc_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[5] = stack_.at(stack_.size() - 2);
    }

    if (!buffer_empty()) {
      ctx[6] = leftmost_child_at(buffer_next());
      ctx[7] = second_leftmost_child_at(buffer_next());
    }

    return ctx;
  }

  Indices next_children_lookahead_context() const {
    Indices ctx(11, -1);

    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[1] = r1;
      ctx[2] = second_rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[3] = leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[4] = arc_at(stack_.at(stack_.size() - 1));
    }
    if (stack_.size() >= 2) {
      ctx[5] = stack_.at(stack_.size() - 2);
    }

    if (!buffer_empty()) {
      ctx[6] = leftmost_child_at(buffer_next());
      ctx[7] = second_leftmost_child_at(buffer_next());
    }

    if (buffer_next() < size()) ctx[8] = buffer_next();
    if (buffer_next() + 1 < size()) ctx[9] = buffer_next() + 1;
    if (buffer_next() + 2 < size()) ctx[10] = buffer_next() + 2;

    return ctx;
  }

  Indices extended_next_children_context() const {
    Indices ctx(13, -1);

    if (stack_.size() >= 1) {
      ctx[0] = stack_.at(stack_.size() - 1);
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));
      if (r1 != buffer_next()) ctx[1] = r1;
      ctx[2] = second_rightmost_child_at(stack_.at(stack_.size() - 1));
      ctx[3] = leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[4] = second_leftmost_child_at(stack_.at(stack_.size() - 1));
      ctx[5] = arc_at(stack_.at(stack_.size() - 1));
      if (ctx[5] >= 0) ctx[6] = arc_at(ctx[5]);  // grandparent
    }
    if (stack_.size() >= 2) {
      ctx[7] = stack_.at(stack_.size() - 2);
      ctx[8] = rightmost_child_at(stack_.at(stack_.size() - 2));
      ctx[9] = leftmost_child_at(stack_.at(stack_.size() - 2));
    }

    if (!buffer_empty()) {
      ctx[10] = leftmost_child_at(buffer_next());
      ctx[11] = second_leftmost_child_at(buffer_next());
    }

    if (stack_.size() >= 3) {
      ctx[12] = stack_.at(stack_.size() - 3);
    }

    return ctx;
  }

  Words tag_next_children_context() const {
    Words ctx(5, 0);

    if (stack_.size() >= 1) {
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size() - 1));

      ctx[3] = tag_at(stack_.at(stack_.size() - 1));
      if (l1 >= 0) ctx[0] = tag_at(l1);
      if (r1 >= 0 && (r1 != buffer_next())) ctx[1] = tag_at(r1);
    }

    if (stack_.size() >= 2) {
      ctx[2] = tag_at(stack_.at(stack_.size() - 2));
    }

    if (!buffer_empty()) {
      ctx[4] = tag_at(buffer_next());
    }

    return ctx;
  }

  Words tag_next_children_word_context() const {
    Words ctx(7, 0);

    if (stack_.size() >= 1) {
      WordIndex r1 = rightmost_child_at(stack_.at(stack_.size() - 1));
      WordIndex l1 = leftmost_child_at(stack_.at(stack_.size() - 1));

      ctx[6] = tag_at(stack_.at(stack_.size() - 1));
      ctx[0] = word_at(stack_.at(stack_.size() - 1));
      if (l1 >= 0) ctx[3] = tag_at(l1);
      if (r1 >= 0 && (r1 != buffer_next())) ctx[4] = tag_at(r1);
    }

    if (!buffer_empty()) {
      WordIndex bl1 = leftmost_child_at(buffer_next());
      if (bl1 >= 0) ctx[5] = tag_at(bl1);
    }

    if (stack_.size() >= 2) {
      ctx[2] = tag_at(stack_.at(stack_.size() - 2));
    }

    if (stack_.size() >= 3) {
      ctx[1] = tag_at(stack_.at(stack_.size() - 3));
    }

    return ctx;
  }

  static bool cmp_particle_weights(
      const boost::shared_ptr<TransitionParser>& p1,
      const boost::shared_ptr<TransitionParser>& p2) {
    // Null should be the biggest.
    if ((p1 == nullptr) || (p1->num_particles() == 0))
      return false;
    else if ((p2 == nullptr) || (p2->num_particles() == 0))
      return true;
    else
      return (p1->particle_weight() < p2->particle_weight());
  }

  static bool cmp_weighted_importance_weights(
      const boost::shared_ptr<TransitionParser>& p1,
      const boost::shared_ptr<TransitionParser>& p2) {
    // Null or no particles should be the biggest.
    if ((p1 == nullptr) || (p1->num_particles() == 0))
      return false;
    else if ((p2 == nullptr) || (p2->num_particles() == 0))
      return true;
    else
      return (p1->weighted_importance_weight() <
              p2->weighted_importance_weight());
  }

 private:
  Indices stack_;
  WordIndex buffer_next_;
  ActList actions_;
  Words action_labels_;
  Real importance_weight_;
  Real beam_weight_;
  Real marginal_weight_;
  int num_particles_;
  boost::shared_ptr<ModelConfig> config_;
};

}  // namespace oxlm

#endif
