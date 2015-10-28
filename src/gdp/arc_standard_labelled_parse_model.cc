#include "gdp/arc_standard_labelled_parse_model.h"

namespace oxlm {

template <class ParsedWeights>
ArcStandardLabelledParseModel<ParsedWeights>::ArcStandardLabelledParseModel(
    boost::shared_ptr<ModelConfig> config)
    : config_(config) {}

template <class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::reallocateParticles(
    AslParserList* beam_stack, unsigned num_particles) {
  std::vector<Real> weight(beam_stack->size(), L_MAX);
  Real sum = beam_stack->at(0)->weighted_particle_weight();
  for (unsigned i = 0; i < beam_stack->size(); ++i) {
    if (beam_stack->at(i)->num_particles() > 0) {
      weight[i] = beam_stack->at(i)->weighted_particle_weight();
      if (i > 0) sum = neg_log_sum_exp(sum, weight[i]);
    }
  }

  std::vector<int> sample_counts(beam_stack->size(), 0);
  unsigned best_i = 0;
  Real best_weight = beam_stack->at(0)->particle_weight();

  for (unsigned i = 0; i < beam_stack->size(); ++i) {
    if (beam_stack->at(i)->num_particles() > 0) {
      sample_counts[i] =
          (int)(std::exp(sum - beam_stack->at(i)->weighted_particle_weight()) *
                num_particles);
      if (beam_stack->at(i)->particle_weight() < best_weight) {
        best_weight = beam_stack->at(i)->particle_weight();
        best_i = i;
      }
    }
  }

  for (unsigned i = 0; i < beam_stack->size(); ++i) {
    if (beam_stack->at(i)->num_particles() > 0) {
      beam_stack->at(i)->set_num_particles(sample_counts[i]);
    }
  }
}

template <class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::resampleParticleParticles(
    AslParserList* beam_stack, MT19937& eng, unsigned num_particles) {
  std::vector<Real> importance_w(beam_stack->size(), L_MAX);
  bool valid_sample = false;
  for (unsigned i = 0; i < importance_w.size(); ++i) {
    if (beam_stack->at(i)->num_particles() > 0) {
      importance_w[i] = beam_stack->at(i)->weighted_particle_weight();
      valid_sample = true;
    }
  }

  // Resample according to particle weight.
  std::vector<int> sample_counts(beam_stack->size(), 0);
  if (valid_sample) {
    multinomial_distribution_log<Real> part_mult(importance_w);
    for (unsigned i = 0; i < num_particles;) {
      unsigned pi = part_mult(eng);
      ++sample_counts[pi];
      ++i;
    }
  }

  for (unsigned i = 0; i < beam_stack->size(); ++i) {
    beam_stack->at(i)->set_num_particles(sample_counts[i]);
  }
}

template <class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::resampleParticles(
    AslParserList* beam_stack, MT19937& eng, unsigned num_particles) {
  std::vector<Real> importance_w(beam_stack->size(), L_MAX);
  bool valid_sample = false;

  for (unsigned i = 0; i < importance_w.size(); ++i) {
    if (beam_stack->at(i)->num_particles() > 0) {
      importance_w[i] = beam_stack->at(i)->weighted_importance_weight();
      valid_sample = true;
    }
  }

  // Resample according to importance weight.
  std::vector<int> sample_counts(beam_stack->size(), 0);
  if (valid_sample) {
    multinomial_distribution_log<Real> part_mult(importance_w);
    for (unsigned i = 0; i < num_particles;) {
      unsigned pi = part_mult(eng);
      ++sample_counts[pi];
      ++i;
    }
  }

  for (unsigned i = 0; i < beam_stack->size(); ++i) {
    beam_stack->at(i)->set_num_particles(sample_counts[i]);
    beam_stack->at(i)->reset_importance_weight();
  }
}

template <class ParsedWeights>
ArcStandardLabelledParser
ArcStandardLabelledParseModel<ParsedWeights>::greedyParseSentence(
    const ParsedSentence& sent,
    const boost::shared_ptr<ParsedWeights>& weights) {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(sent), config_);

  while (!parser.buffer_empty()) {
    Reals action_probs = weights->predictAction(parser.actionContext());
    WordIndex pred = arg_min(action_probs, 0);
    if ((parser.stack_depth() < 2) ||
        (config_->root_first && (parser.stack_depth() == 2))) {
      pred = 0;
    } else if (!config_->root_first && (parser.buffer_next() == 0)) {
      // For root last, enforce reduces so that the parse forms a tree.
      pred = arg_min(action_probs, 1);
    } else if (pred == 0) {
      parser.add_particle_weight(action_probs[pred]);
    }

    // Reduce until a shift action is chosen.
    while (pred > 0) {
      kAction re_act = parser.lookup_action(pred);
      WordId re_label = parser.lookup_label(pred);
      if (re_act == kAction::la) {
        parser.leftArc(re_label);
      } else {
        parser.rightArc(re_label);
      }
      parser.add_particle_weight(action_probs[pred]);

      action_probs = weights->predictAction(parser.actionContext());
      pred = arg_min(action_probs, 0);
      if ((parser.stack_depth() < 2) ||
          (config_->root_first && (parser.stack_depth() == 2))) {
        pred = 0;
      } else if (!config_->root_first && (parser.buffer_next() == 0)) {
        pred = arg_min(action_probs, 1);
      } else if (pred == 0) {
        parser.add_particle_weight(action_probs[pred]);
      }
    }

    // Shift.
    Real tagp = weights->predictTag(parser.next_tag(), parser.tagContext());
    Real wordp = weights->predictWord(parser.next_word(), parser.wordContext());
    parser.shift();
    parser.add_particle_weight(tagp);
    parser.add_particle_weight(wordp);
  }

  // Final reduce actions.
  while (!parser.inTerminalConfiguration()) {
    Reals action_probs = weights->predictAction(parser.actionContext());
    WordIndex pred = arg_min(action_probs, 1);
    if (!parser.left_arc_valid()) {
      pred = arg_min(action_probs, config_->num_labels + 1);
    }
    kAction re_act = parser.lookup_action(pred);
    WordId re_label = parser.lookup_label(pred);

    if (re_act == kAction::la) {
      parser.leftArc(re_label);
    } else {
      parser.rightArc(re_label);
    }
    parser.add_particle_weight(action_probs[pred]);
  }

  return parser;
}

template <class ParsedWeights>
ArcStandardLabelledParser
ArcStandardLabelledParseModel<ParsedWeights>::beamParseSentence(
    const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
    unsigned beam_size) {
  AslParserList beam_stack;
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(
      static_cast<TaggedSentence>(sent), config_));

  for (unsigned i = 0; ((i < sent.size() - 1) ||
                        ((config_->root_first || !config_->complete_parse) &&
                         (i == sent.size() - 1)));
       ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) {
      if (beam_stack[j]->num_particles() < 0) {
        beam_stack[j]->set_num_particles(1);
        continue;
      }

      // Reduce actions are taken direction deterministic.
      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      Real shiftp = action_probs[0];

      if ((beam_stack[j]->stack_depth() >= 2) && (j < beam_size)) {
        // Add best reduce action.
        WordIndex reduce_pred = arg_min(action_probs, 1);
        if (!beam_stack[j]->left_arc_valid()) {
          reduce_pred = arg_min(action_probs, config_->num_labels + 1);
        }

        Real reducep = action_probs[reduce_pred];
        kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
        WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

        if (j < beam_size) {
          beam_stack.push_back(
              boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
          if (re_act == kAction::la) {
            beam_stack.back()->leftArc(re_label);
          } else {
            beam_stack.back()->rightArc(re_label);
          }
          beam_stack.back()->add_particle_weight(reducep);
        }
      } else {
        shiftp = 0;
      }

      // Shift.
      if (config_->tag_pos) {
        Reals tag_probs = weights->predictTag(beam_stack[j]->tagContext());
        Reals word_probs = weights->predictWordOverTags(
            beam_stack[j]->next_word(), beam_stack[j]->wordContext());
        for (unsigned k = 0; k < tag_probs.size(); ++k) {
          tag_probs[k] += word_probs[k];
        }

        // Sort tag+word probability.
        std::vector<int> indices(tag_probs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&tag_probs](const int i, const int j) {
                    return (tag_probs[i] < tag_probs[j]);
                  });

        // Most likely tag.
        WordIndex tag_pred = indices[0];
        beam_stack[j]->update_tag(i, tag_pred);

        // Second most likely tag.
        WordIndex tag2_pred = indices[1];

        beam_stack.push_back(
            boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
        beam_stack.back()->update_tag(i, tag2_pred);
        beam_stack.back()->shift();
        beam_stack.back()->add_particle_weight(shiftp);
        // Hack to communicate that item should be skipped.
        beam_stack.back()->set_num_particles(-1);  

        beam_stack.back()->add_particle_weight(tag_probs[tag2_pred]);
        beam_stack.back()->add_importance_weight(tag_probs[tag2_pred]);
      }

      Real tagp = weights->predictTag(beam_stack[j]->next_tag(),
                                      beam_stack[j]->tagContext());
      Real wordp = weights->predictWord(beam_stack[j]->next_word(),
                                        beam_stack[j]->wordContext());

      beam_stack[j]->shift();
      beam_stack[j]->add_particle_weight(shiftp);
      beam_stack[j]->add_particle_weight(wordp);
      beam_stack[j]->add_particle_weight(tagp);
      beam_stack[j]->add_importance_weight(wordp);
      beam_stack[j]->add_importance_weight(tagp);
    }

    // Prune the beam.
    if (beam_stack.size() > beam_size) {
      std::sort(beam_stack.begin(), beam_stack.end(),
                TransitionParser::cmp_particle_weights);
      for (int j = beam_stack.size() - 1; (j >= beam_size); --j) {
        beam_stack.pop_back();
      }
    }
  }

  // Completion: Greedily reduce each item.
  for (unsigned j = 0; (j < beam_stack.size()); ++j) {
    while (beam_stack[j]->stack_depth() >= 2) {
      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      WordIndex reduce_pred = arg_min(action_probs, 1);
      if (!beam_stack[j]->left_arc_valid())
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);

      Real reducep = action_probs[reduce_pred];
      kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

      if (re_act == kAction::la) {
        beam_stack[j]->leftArc(re_label);
      } else {
        beam_stack[j]->rightArc(re_label);
      }
      beam_stack[j]->add_particle_weight(reducep);
    }

    // For root-last: final shift and reduce.
    if (config_->complete_parse && !config_->root_first) {
      Real shiftp = weights->predictAction(0, beam_stack[j]->actionContext());
      Real tagp = weights->predictTag(beam_stack[j]->next_tag(),
                                      beam_stack[j]->tagContext());
      Real wordp = weights->predictWord(beam_stack[j]->next_word(),
                                        beam_stack[j]->wordContext());

      beam_stack[j]->shift();
      beam_stack[j]->add_particle_weight(shiftp);
      beam_stack[j]->add_particle_weight(wordp);
      beam_stack[j]->add_particle_weight(tagp);
      beam_stack[j]->add_importance_weight(wordp);
      beam_stack[j]->add_importance_weight(tagp);

      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      // Oonly left-arc is valid.
      WordId reduce_pred = arg_min(action_probs, 1, config_->num_labels + 1);
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

      beam_stack[j]->leftArc(re_label);
      beam_stack[j]->add_particle_weight(action_probs[reduce_pred]);
    }
  }

  // Sum over identical parses in the final beam.
  vector<bool> duplicate(beam_stack.size(), false);
  if (config_->sum_over_beam) {
    for (unsigned i = 0; (i < beam_stack.size() - 1); ++i) {
      if (!duplicate[i])
        for (unsigned j = i + 1; (j < beam_stack.size()); ++j) {
          if (ParsedSentence::eq_arcs(beam_stack[i], beam_stack[j])) {
            beam_stack[i]->add_log_particle_weight(
                beam_stack[j]->particle_weight());
            duplicate[j] = true;
          }
        }
    }
  }

  std::sort(beam_stack.begin(), beam_stack.end(),
            TransitionParser::cmp_particle_weights);

  // Compute beam weight.
  for (unsigned i = 0; (i < beam_stack.size()); ++i) {
    if (!duplicate[i]) {
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight());
    }
  }

  if (beam_stack.size() == 0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent),
                                     config_);
  } else {
    return ArcStandardLabelledParser(*beam_stack[0]);
  }
}

template <class ParsedWeights>
ArcStandardLabelledParser
ArcStandardLabelledParseModel<ParsedWeights>::beamParticleParseSentence(
    const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
    unsigned num_particles, const boost::shared_ptr<ParseDataSet>& examples) {
  AslParserList beam_stack;
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(
      static_cast<TaggedSentence>(sent), static_cast<int>(num_particles),
      config_));

  for (unsigned i = 0; ((i < sent.size() - 1) ||
                        ((config_->root_first || !config_->complete_parse) &&
                         (i == sent.size() - 1)));
       ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) {
      int num_samples = beam_stack[j]->num_particles();
      if (num_samples == 0) {
        continue;
      } else if (num_samples < 0) {
        beam_stack[j]->set_num_particles(-num_samples);
        continue;
      }

      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      int shift_count = 0;
      int reduce_count = 0;

      Real shiftp = action_probs[0];
      Real tot_reducep = log_one_min(shiftp);

      if ((beam_stack[j]->stack_depth() < 2) ||
          (config_->root_first && config_->complete_parse &&
           (beam_stack[j]->stack_depth() == 2))) {
        shift_count = num_samples;
        shiftp = 0;
      } else {
        shift_count = std::round(std::exp(-shiftp) * num_samples);
        reduce_count = num_samples - shift_count;

        if (config_->direction_deterministic) {
          WordIndex reduce_pred =
              arg_min(action_probs, 1, 2 * config_->num_labels + 1);
          if (config_->parser_type == ParserType::arcstandard2) {
            reduce_pred = arg_min(action_probs, 1);
          }

          if (!beam_stack[j]->left_arc2_valid() &&
              (reduce_pred <= 3 * config_->num_labels)) {
            reduce_pred = arg_min(action_probs, 1, 2 * config_->num_labels + 1);
          }
          if (!beam_stack[j]->left_arc_valid()) {
            reduce_pred = arg_min(action_probs, config_->num_labels + 1);
          }

          if (reduce_count > 0) {
            kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
            WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (re_act == kAction::la) {
              beam_stack.back()->leftArc(re_label);
            } else if (re_act == kAction::ra) {
              beam_stack.back()->rightArc(re_label);
            } else if (re_act == kAction::la2) {
              beam_stack.back()->leftArc2(re_label);
            } else if (re_act == kAction::ra2) {
              beam_stack.back()->rightArc2(re_label);
            }
            beam_stack.back()->add_particle_weight(action_probs[reduce_pred]);
            beam_stack.back()->set_num_particles(reduce_count);
          }
        } else {
          if (config_->parser_type == ParserType::arcstandard2) {
            WordIndex left_reduce2_pred =
                arg_min(action_probs, 2 * config_->num_labels + 1,
                        3 * config_->num_labels + 1);
            WordIndex right_reduce2_pred =
                arg_min(action_probs, 3 * config_->num_labels + 1,
                        4 * config_->num_labels + 1);
            Real left_reduce2p = L_MAX;
            for (unsigned l = 2 * config_->num_labels + 1;
                 l < 3 * config_->num_labels + 1; ++l) {
              left_reduce2p = neg_log_sum_exp(left_reduce2p, action_probs[l]);
            }
            Real right_reduce2p = L_MAX;
            for (unsigned l = 3 * config_->num_labels + 1;
                 l < 4 * config_->num_labels + 1; ++l) {
              right_reduce2p = neg_log_sum_exp(right_reduce2p, action_probs[l]);
            }

            int left_reduce2_count =
                std::round(std::exp(-left_reduce2p) * num_samples);
            if (!beam_stack[j]->left_arc2_valid()) left_reduce2_count = 0;
            int right_reduce2_count =
                std::round(std::exp(-right_reduce2p) * num_samples);
            reduce_count =
                reduce_count - left_reduce2_count - right_reduce2_count;

            if (left_reduce2_count > 0) {
              WordId re_label = beam_stack[j]->lookup_label(left_reduce2_pred);

              beam_stack.push_back(
                  boost::make_shared<ArcStandardLabelledParser>(
                      *beam_stack[j]));
              beam_stack.back()->leftArc2(re_label);
              beam_stack.back()->add_particle_weight(
                  action_probs[left_reduce2_pred]);
              beam_stack.back()->set_num_particles(left_reduce2_count);
            }

            if (right_reduce2_count > 0) {
              WordId re_label = beam_stack[j]->lookup_label(right_reduce2_pred);

              beam_stack.push_back(
                  boost::make_shared<ArcStandardLabelledParser>(
                      *beam_stack[j]));
              beam_stack.back()->rightArc2(re_label);
              beam_stack.back()->add_particle_weight(
                  action_probs[right_reduce2_pred]);
              beam_stack.back()->set_num_particles(right_reduce2_count);
            }
          }

          Real left_reducep = L_MAX;
          for (unsigned l = 1; l < config_->num_labels + 1; ++l) {
            left_reducep = neg_log_sum_exp(left_reducep, action_probs[l]);
          }
          Real right_reducep = L_MAX;
          for (unsigned l = config_->num_labels + 1;
               l < 2 * config_->num_labels + 1; ++l) {
            right_reducep = neg_log_sum_exp(right_reducep, action_probs[l]);
          }

          WordIndex left_reduce_pred =
              arg_min(action_probs, 1, config_->num_labels + 1);
          Real left_reduce_pred_prob = action_probs[left_reduce_pred];

          WordIndex right_reduce_pred =
              arg_min(action_probs, config_->num_labels + 1,
                      2 * config_->num_labels + 1);
          Real right_reduce_pred_prob = action_probs[right_reduce_pred];

          // Second best labels.
          action_probs[left_reduce_pred] = std::numeric_limits<Real>::max();
          WordIndex left_reduce_pred2 =
              arg_min(action_probs, 1, config_->num_labels + 1);
          Real left_reduce_pred_prob2 = action_probs[left_reduce_pred];

          action_probs[right_reduce_pred] = std::numeric_limits<Real>::max();
          WordIndex right_reduce_pred2 =
              arg_min(action_probs, config_->num_labels + 1,
                      2 * config_->num_labels + 1);
          Real right_reduce_pred_prob2 = action_probs[right_reduce_pred];

          int left_reduce_count =
              std::round(std::exp(-left_reducep) * num_samples);
          if (!beam_stack[j]->left_arc_valid()) left_reduce_count = 0;
          double left_pred_sum =
              neg_log_sum_exp(left_reduce_pred_prob, left_reduce_pred_prob2);
          int left_reduce_count2 =
              std::floor(std::exp(-left_reduce_pred_prob2 + left_pred_sum) *
                         left_reduce_count);

          int right_reduce_count = reduce_count - left_reduce_count;
          left_reduce_count -= left_reduce_count2;
          double right_pred_sum =
              neg_log_sum_exp(right_reduce_pred_prob, right_reduce_pred_prob2);
          int right_reduce_count2 =
              std::floor(std::exp(-right_reduce_pred_prob2 + right_pred_sum) *
                         right_reduce_count);
          right_reduce_count -= right_reduce_count2;

          if (left_reduce_count > 0) {
            WordId re_label = beam_stack[j]->lookup_label(left_reduce_pred);

            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->leftArc(re_label);
            beam_stack.back()->add_particle_weight(left_reduce_pred_prob);
            beam_stack.back()->set_num_particles(left_reduce_count);
          }

          if (left_reduce_count2 > 0) {
            WordId re_label = beam_stack[j]->lookup_label(left_reduce_pred2);

            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->leftArc(re_label);
            beam_stack.back()->add_particle_weight(left_reduce_pred_prob2);
            beam_stack.back()->set_num_particles(left_reduce_count2);
          }

          if (right_reduce_count > 0) {
            WordId re_label = beam_stack[j]->lookup_label(right_reduce_pred);

            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(right_reduce_pred_prob);
            beam_stack.back()->set_num_particles(right_reduce_count);
          }

          if (right_reduce_count2 > 0) {
            WordId re_label = beam_stack[j]->lookup_label(right_reduce_pred2);

            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(right_reduce_pred_prob2);
            beam_stack.back()->set_num_particles(right_reduce_count2);
          }
        }
      }

      if (shift_count == 0) {
        beam_stack[j]->set_num_particles(0);
      } else {
        if (config_->tag_pos) {
          Reals tag_probs = weights->predictTag(beam_stack[j]->tagContext());
          Reals word_probs = weights->predictWordOverTags(
              beam_stack[j]->next_word(), beam_stack[j]->wordContext());
          for (unsigned k = 0; k < tag_probs.size(); ++k) {
            tag_probs[k] += word_probs[k];
          }

          // Sort tag+word probobility.
          std::vector<int> indices(tag_probs.size());
          std::iota(indices.begin(), indices.end(), 0);
          std::sort(indices.begin(), indices.end(),
                    [&tag_probs](const int i, const int j) {
            return (tag_probs[i] < tag_probs[j]);
          });

          // Most likely tag.
          WordIndex tag_pred = indices[0];
          beam_stack[j]->update_tag(i, tag_pred);

          // Add second most likely tag if probability is high enough.
          WordIndex tag2_pred = indices[1];
          WordIndex tag3_pred = indices[2];
          double pred_sum = neg_log_sum_exp(
              neg_log_sum_exp(tag_probs[tag_pred], tag_probs[tag2_pred]),
              tag_probs[tag3_pred]);

          int shift2_count = std::floor(
              std::exp(-tag_probs[tag2_pred] + pred_sum) * shift_count);
          int shift3_count = std::floor(
              std::exp(-tag_probs[tag3_pred] + pred_sum) * shift_count);

          if (shift2_count > 0) {
            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->update_tag(i, tag2_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp);
            // Hack to communicate that item should be skipped.
            beam_stack.back()->set_num_particles(-shift2_count);  

            beam_stack.back()->add_particle_weight(tag_probs[tag2_pred]);
            beam_stack.back()->add_importance_weight(tag_probs[tag2_pred]);
            shift_count -= shift2_count;
          }

          // add third most likely tag if prob is high enough
          if (shift3_count > 0) {
            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->update_tag(i, tag3_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp);
            beam_stack.back()->set_num_particles(-shift3_count);

            beam_stack.back()->add_particle_weight(tag_probs[tag3_pred]);
            beam_stack.back()->add_importance_weight(tag_probs[tag3_pred]);
            shift_count -= shift3_count;
          }
        }

        Real tagp = weights->predictTag(beam_stack[j]->next_tag(),
                                        beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(),
                                          beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp);
        beam_stack[j]->set_num_particles(shift_count);

        beam_stack[j]->add_particle_weight(wordp);
        beam_stack[j]->add_particle_weight(tagp);
        beam_stack[j]->add_importance_weight(wordp);
        beam_stack[j]->add_importance_weight(tagp);
      }
    }

    reallocateParticles(&beam_stack, num_particles);
  }

  // Completion: Greedily reduce each item.
  for (unsigned j = 0; (j < beam_stack.size()); ++j) {
    while ((beam_stack[j]->num_particles() > 0) &&
           (beam_stack[j]->stack_depth() >= 2)) {
      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();

      WordIndex reduce_pred =
          arg_min(action_probs, 1, 2 * config_->num_labels + 1);
      if (config_->parser_type == ParserType::arcstandard2)
        reduce_pred = arg_min(action_probs, 1);

      if (!beam_stack[j]->left_arc2_valid() &&
          (reduce_pred <= 3 * config_->num_labels))
        reduce_pred = arg_min(action_probs, 1, 2 * config_->num_labels + 1);
      if (!beam_stack[j]->left_arc_valid())
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);

      Real reducep = action_probs[reduce_pred];
      kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

      if (re_act == kAction::la) {
        beam_stack[j]->leftArc(re_label);
      } else if (re_act == kAction::ra) {
        beam_stack[j]->rightArc(re_label);
      } else if (re_act == kAction::la2) {
        beam_stack[j]->leftArc2(re_label);
      } else if (re_act == kAction::ra2) {
        beam_stack[j]->rightArc2(re_label);
      }

      beam_stack[j]->add_particle_weight(reducep);
      beam_stack[j]->set_num_particles(num_samples);
    }

    // For root-last: final shift and reduce.
    if (config_->complete_parse && !config_->root_first) {
      Real shiftp = weights->predictAction(0, beam_stack[j]->actionContext());
      Real tagp = weights->predictTag(beam_stack[j]->next_tag(),
                                      beam_stack[j]->tagContext());
      Real wordp = weights->predictWord(beam_stack[j]->next_word(),
                                        beam_stack[j]->wordContext());

      beam_stack[j]->shift();
      beam_stack[j]->add_particle_weight(shiftp);
      beam_stack[j]->add_particle_weight(wordp);
      beam_stack[j]->add_particle_weight(tagp);
      beam_stack[j]->add_importance_weight(wordp);
      beam_stack[j]->add_importance_weight(tagp);

      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      // Only left-arc is valid.
      WordId reduce_pred = arg_min(action_probs, 1, config_->num_labels + 1);
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

      beam_stack[j]->leftArc(re_label);
      beam_stack[j]->add_particle_weight(action_probs[reduce_pred]);
    }
  }

  // Sum over identical parses in final beam.
  vector<bool> duplicate(beam_stack.size(), false);
  if (config_->sum_over_beam) {
    for (unsigned i = 0; (i < beam_stack.size() - 1); ++i) {
      if (!duplicate[i]) {
        for (unsigned j = i + 1; (j < beam_stack.size()); ++j) {
          if (ParsedSentence::eq_arcs(beam_stack[i], beam_stack[j])) {
            beam_stack[i]->add_log_particle_weight(
                beam_stack[j]->particle_weight());
            duplicate[j] = true;
          }
        }
      }
    }
  }

  std::sort(beam_stack.begin(), beam_stack.end(),
            TransitionParser::cmp_particle_weights);

  // Compute beam weight.
  for (unsigned i = 0; (i < beam_stack.size()); ++i) {
    if (!duplicate[i] && (beam_stack[i]->num_particles() > 0)) {
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight());
    }
  }

  if (beam_stack.size() == 0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent),
                                     config_);
  } else {
    return ArcStandardLabelledParser(*beam_stack[0]);
  }
}

template <class ParsedWeights>
ArcStandardLabelledParser
ArcStandardLabelledParseModel<ParsedWeights>::particleParseSentence(
    const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
    MT19937& eng, unsigned num_particles,
    const boost::shared_ptr<ParseDataSet>& examples) {
  AslParserList beam_stack;
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(
      static_cast<TaggedSentence>(sent), static_cast<int>(num_particles),
      config_));

  for (unsigned i = 0; ((i < sent.size() - 1) ||
                        ((config_->root_first || !config_->complete_parse) &&
                         (i == sent.size() - 1)));
       ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) {
      int num_samples = beam_stack[j]->num_particles();
      if (num_samples == 0) {
        continue;
      } else if (num_samples < 0) {
        beam_stack[j]->set_num_particles(-num_samples);
        continue;
      }
      
      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      int shift_count = 0;
      int reduce_count = 0;

      Real shiftp = action_probs[0];
      Real tot_reducep = log_one_min(shiftp);

      if ((beam_stack[j]->stack_depth() < 2) ||
          (config_->root_first && config_->complete_parse &&
           (beam_stack[j]->stack_depth() == 2))) {
        shift_count = num_samples;
        shiftp = 0;
      } else {
        shift_count = std::round(std::exp(-shiftp) * num_samples);
        reduce_count = num_samples - shift_count;

        if (config_->direction_deterministic) {
          WordIndex reduce_pred =
              arg_min(action_probs, 1, 2 * config_->num_labels + 1);
          if (config_->parser_type == ParserType::arcstandard2) {
            reduce_pred = arg_min(action_probs, 1);
          }

          if (!beam_stack[j]->left_arc2_valid() &&
              (reduce_pred <= 3 * config_->num_labels)) {
            reduce_pred = arg_min(action_probs, 1, 2 * config_->num_labels + 1);
          }
          if (!beam_stack[j]->left_arc_valid()) {
            reduce_pred = arg_min(action_probs, config_->num_labels + 1);
          }

          if (reduce_count > 0) {
            kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
            WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (re_act == kAction::la) {
              beam_stack.back()->leftArc(re_label);
            } else if (re_act == kAction::ra) {
              beam_stack.back()->rightArc(re_label);
            } else if (re_act == kAction::la2) {
              beam_stack.back()->leftArc2(re_label);
            } else if (re_act == kAction::ra2) {
              beam_stack.back()->rightArc2(re_label);
            }

            beam_stack.back()->add_particle_weight(action_probs[reduce_pred]);
            beam_stack.back()->set_num_particles(reduce_count);
          }
        } else {
          if (config_->parser_type == ParserType::arcstandard2) {
            WordIndex left_reduce2_pred =
                arg_min(action_probs, 2 * config_->num_labels + 1,
                        3 * config_->num_labels + 1);
            WordIndex right_reduce2_pred =
                arg_min(action_probs, 3 * config_->num_labels + 1,
                        4 * config_->num_labels + 1);
            Real left_reduce2p = L_MAX;
            for (unsigned l = 2 * config_->num_labels + 1;
                 l < 3 * config_->num_labels + 1; ++l) {
              left_reduce2p = neg_log_sum_exp(left_reduce2p, action_probs[l]);
            }
            Real right_reduce2p = L_MAX;
            for (unsigned l = 3 * config_->num_labels + 1;
                 l < 4 * config_->num_labels + 1; ++l) {
              right_reduce2p = neg_log_sum_exp(right_reduce2p, action_probs[l]);
            }

            int left_reduce2_count =
                std::round(std::exp(-left_reduce2p) * num_samples);
            if (!beam_stack[j]->left_arc2_valid()) left_reduce2_count = 0;
            int right_reduce2_count =
                std::round(std::exp(-right_reduce2p) * num_samples);
            reduce_count =
                reduce_count - left_reduce2_count - right_reduce2_count;

            if (left_reduce2_count > 0) {
              WordId re_label = beam_stack[j]->lookup_label(left_reduce2_pred);

              beam_stack.push_back(
                  boost::make_shared<ArcStandardLabelledParser>(
                      *beam_stack[j]));
              beam_stack.back()->leftArc2(re_label);
              beam_stack.back()->add_particle_weight(
                  action_probs[left_reduce2_pred]);
              beam_stack.back()->set_num_particles(left_reduce2_count);
            }

            if (right_reduce2_count > 0) {
              WordId re_label = beam_stack[j]->lookup_label(right_reduce2_pred);

              beam_stack.push_back(
                  boost::make_shared<ArcStandardLabelledParser>(
                      *beam_stack[j]));
              beam_stack.back()->rightArc2(re_label);
              beam_stack.back()->add_particle_weight(
                  action_probs[right_reduce2_pred]);
              beam_stack.back()->set_num_particles(right_reduce2_count);
            }
          }

          WordIndex left_reduce_pred =
              arg_min(action_probs, 1, config_->num_labels + 1);
          WordIndex right_reduce_pred =
              arg_min(action_probs, config_->num_labels + 1,
                      2 * config_->num_labels + 1);
          Real left_reducep = L_MAX;
          for (unsigned l = 1; l < config_->num_labels + 1; ++l) {
            left_reducep = neg_log_sum_exp(left_reducep, action_probs[l]);
          }
          Real right_reducep = L_MAX;
          for (unsigned l = config_->num_labels + 1;
               l < 2 * config_->num_labels + 1; ++l) {
            right_reducep = neg_log_sum_exp(right_reducep, action_probs[l]);
          }

          int left_reduce_count =
              std::round(std::exp(-left_reducep) * num_samples);
          if (!beam_stack[j]->left_arc_valid()) left_reduce_count = 0;
          int right_reduce_count = reduce_count - left_reduce_count;

          if (left_reduce_count > 0) {
            WordId re_label = beam_stack[j]->lookup_label(left_reduce_pred);

            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->leftArc(re_label);
            beam_stack.back()->add_particle_weight(
                action_probs[left_reduce_pred]);
            beam_stack.back()->set_num_particles(left_reduce_count);
          }

          if (right_reduce_count > 0) {
            WordId re_label = beam_stack[j]->lookup_label(right_reduce_pred);

            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(
                action_probs[right_reduce_pred]);
            beam_stack.back()->set_num_particles(right_reduce_count);
          }
        }
      }

      if (shift_count == 0) {
        beam_stack[j]->set_num_particles(0);
      } else {
        if (config_->tag_pos) {
          Reals tag_probs = weights->predictTag(beam_stack[j]->tagContext());
          Reals word_probs = weights->predictWordOverTags(
              beam_stack[j]->next_word(), beam_stack[j]->wordContext());
          for (unsigned k = 0; k < tag_probs.size(); ++k) {
            tag_probs[k] += word_probs[k];
          }

          // Sort tag+word probability.
          std::vector<int> indices(tag_probs.size());
          std::iota(indices.begin(), indices.end(), 0);
          std::sort(indices.begin(), indices.end(),
                    [&tag_probs](const int i, const int j) {
            return (tag_probs[i] < tag_probs[j]);
          });

          // Most likely tag.
          WordIndex tag_pred = indices[0];
          beam_stack[j]->update_tag(i, tag_pred);

          // Add second most likely tag if probability is high enough.
          WordIndex tag2_pred = indices[1];
          WordIndex tag3_pred = indices[2];
          double pred_sum = neg_log_sum_exp(
              neg_log_sum_exp(tag_probs[tag_pred], tag_probs[tag2_pred]),
              tag_probs[tag3_pred]);

          int shift2_count = std::floor(
              std::exp(-tag_probs[tag2_pred] + pred_sum) * shift_count);
          int shift3_count = std::floor(
              std::exp(-tag_probs[tag3_pred] + pred_sum) * shift_count);

          if (shift2_count > 0) {
            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->update_tag(i, tag2_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp);
            beam_stack.back()->set_num_particles(-shift2_count);

            beam_stack.back()->add_particle_weight(tag_probs[tag2_pred]);
            beam_stack.back()->add_importance_weight(tag_probs[tag2_pred]);
            shift_count -= shift2_count;
          }

          // Add third most likely tag if prob is high enough.
          if (shift3_count > 0) {
            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->update_tag(i, tag3_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp);
            beam_stack.back()->set_num_particles(-shift3_count); 

            beam_stack.back()->add_particle_weight(tag_probs[tag3_pred]);
            beam_stack.back()->add_importance_weight(tag_probs[tag3_pred]);
            shift_count -= shift3_count;
          }
        }

        Real tagp = weights->predictTag(beam_stack[j]->next_tag(),
                                        beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(),
                                          beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp);
        beam_stack[j]->set_num_particles(shift_count);

        beam_stack[j]->add_particle_weight(wordp);
        beam_stack[j]->add_particle_weight(tagp);
        beam_stack[j]->add_importance_weight(wordp);
        beam_stack[j]->add_importance_weight(tagp);
      }
    }

    resampleParticles(&beam_stack, eng, num_particles);
  }

  // Completion: Greedily reduce each item.
  for (unsigned j = 0; (j < beam_stack.size()); ++j) {
    while ((beam_stack[j]->num_particles() > 0) &&
           (beam_stack[j]->stack_depth() >= 2)) {
      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();

      WordIndex reduce_pred =
          arg_min(action_probs, 1, 2 * config_->num_labels + 1);
      if (config_->parser_type == ParserType::arcstandard2) {
        reduce_pred = arg_min(action_probs, 1);
      }

      if (!beam_stack[j]->left_arc2_valid() &&
          (reduce_pred <= 3 * config_->num_labels)) {
        reduce_pred = arg_min(action_probs, 1, 2 * config_->num_labels + 1);
      }
      if (!beam_stack[j]->left_arc_valid()) {
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);
      }

      Real reducep = action_probs[reduce_pred];
      kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

      if (re_act == kAction::la) {
        beam_stack[j]->leftArc(re_label);
      } else if (re_act == kAction::ra) {
        beam_stack[j]->rightArc(re_label);
      } else if (re_act == kAction::la2) {
        beam_stack[j]->leftArc2(re_label);
      } else if (re_act == kAction::ra2) {
        beam_stack[j]->rightArc2(re_label);
      }

      beam_stack[j]->add_particle_weight(reducep);
      beam_stack[j]->set_num_particles(num_samples);
    }

    // For root-last: Final shift and reduce.
    if (config_->complete_parse && !config_->root_first) {
      Real shiftp = weights->predictAction(0, beam_stack[j]->actionContext());
      Real tagp = weights->predictTag(beam_stack[j]->next_tag(),
                                      beam_stack[j]->tagContext());
      Real wordp = weights->predictWord(beam_stack[j]->next_word(),
                                        beam_stack[j]->wordContext());

      beam_stack[j]->shift();
      beam_stack[j]->add_particle_weight(shiftp);
      beam_stack[j]->add_particle_weight(wordp);
      beam_stack[j]->add_particle_weight(tagp);
      beam_stack[j]->add_importance_weight(wordp);
      beam_stack[j]->add_importance_weight(tagp);

      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      WordId reduce_pred = arg_min(
          action_probs, 1, config_->num_labels + 1);
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

      beam_stack[j]->leftArc(re_label);
      beam_stack[j]->add_particle_weight(action_probs[reduce_pred]);
    }
  }

  // Sum over identical parses in final beam.
  vector<bool> duplicate(beam_stack.size(), false);
  if (config_->sum_over_beam) {
    for (unsigned i = 0; (i < beam_stack.size() - 1); ++i) {
      if (!duplicate[i]) {
        for (unsigned j = i + 1; (j < beam_stack.size()); ++j) {
          if (ParsedSentence::eq_arcs(beam_stack[i], beam_stack[j])) {
            beam_stack[i]->add_log_particle_weight(
                beam_stack[j]->particle_weight());
            duplicate[j] = true;
          }
        }
      }
    }
  }

  std::sort(beam_stack.begin(), beam_stack.end(),
            TransitionParser::cmp_weighted_importance_weights);

  // Compute beam weight.
  for (unsigned i = 0; (i < beam_stack.size()); ++i) {
    if (!duplicate[i] && (beam_stack[i]->num_particles() > 0)) {
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight());
    }
  }

  if (beam_stack.size() == 0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent),
                                     config_);
  } else {
    return ArcStandardLabelledParser(*beam_stack[0]);
  }
}

// Find highest-scoring parse consistent with gold-parse.
template <class ParsedWeights>
ArcStandardLabelledParser
ArcStandardLabelledParseModel<ParsedWeights>::beamParticleGoldParseSentence(
    const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
    unsigned num_particles) {
  AslParserList beam_stack;
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(
      static_cast<TaggedSentence>(sent), static_cast<int>(num_particles),
      config_));

  for (unsigned i = 0; (i < sent.size()); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) {
      int num_samples = beam_stack[j]->num_particles();
      if (num_samples == 0) {
        continue;
      } else if (num_samples < 0) {
        beam_stack[j]->set_num_particles(-num_samples);
        continue;
      }

      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      int shift_count = 0;
      int reduce_count = 0;

      Real shiftp = action_probs[0];
      Real tot_reducep = log_one_min(shiftp);

      if (beam_stack[j]->stack_depth() < 2) {
        shift_count = num_samples;
        shiftp = 0;
      } else {
        kAction gold_act = beam_stack[j]->oracleNext(sent);
        WordId gold_label = beam_stack[j]->oracleNextLabel(sent);

        if (gold_act == kAction::la || gold_act == kAction::ra) {
          WordIndex reduce_pred =
              beam_stack[j]->convert_action(gold_act, gold_label);
          shift_count = std::round(std::exp(-shiftp) * num_samples);
          if (gold_act == kAction::ra)
            shift_count = 0;
          else {
            // Shift is invalid if s0 has all its right children.
            int st = beam_stack[j]->stack_top();
            bool has_right_children = true;
            for (WordIndex l = st + 1; l < beam_stack[j]->size(); ++l) {
              if (sent.has_arc(l, st) && !beam_stack[j]->has_arc(l, st)) {
                has_right_children = false;
                break;
              }
            }
            if (has_right_children) shift_count = 0;
          }

          reduce_count = num_samples - shift_count;

          if (reduce_count > 0) {
            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (gold_act == kAction::la) {
              beam_stack.back()->leftArc(gold_label);
            } else {
              beam_stack.back()->rightArc(gold_label);
            }
            beam_stack.back()->add_particle_weight(action_probs[reduce_pred]);
            beam_stack.back()->set_num_particles(reduce_count);
          }
        } else {
          shift_count = num_samples;
        }
      }

      if (shift_count == 0) {
        beam_stack[j]->set_num_particles(0);
      } else {
        // Don't generate POS tag.

        Real tagp = weights->predictTag(beam_stack[j]->next_tag(),
                                        beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(),
                                          beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp);
        beam_stack[j]->add_particle_weight(wordp);
        beam_stack[j]->add_particle_weight(tagp);
        beam_stack[j]->add_importance_weight(wordp);
        beam_stack[j]->add_importance_weight(tagp);
        beam_stack[j]->set_num_particles(shift_count);
      }
    }

    reallocateParticles(&beam_stack, num_particles);
  }

  // Completion.
  for (unsigned j = 0; (j < beam_stack.size()); ++j) {
    while ((beam_stack[j]->num_particles() > 0) &&
           (beam_stack[j]->stack_depth() >= 2)) {
      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();

      kAction gold_act = beam_stack[j]->oracleNext(sent);
      WordId gold_label = beam_stack[j]->oracleNextLabel(sent);

      if (gold_act == kAction::la || gold_act == kAction::ra) {
        WordIndex reduce_pred =
            beam_stack[j]->convert_action(gold_act, gold_label);
        Real reducep = action_probs[reduce_pred];

        if (gold_act == kAction::la)
          beam_stack[j]->leftArc(gold_label);
        else
          beam_stack[j]->rightArc(gold_label);
        beam_stack[j]->add_particle_weight(reducep);
        beam_stack[j]->set_num_particles(num_samples);
      } else {
        break; // assume that gold does not want a complete parse
      }
    }
  }

  std::sort(beam_stack.begin(), beam_stack.end(),
            TransitionParser::cmp_particle_weights);

  if (beam_stack.size() == 0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent),
                                     config_);
  } else {
    return ArcStandardLabelledParser(*beam_stack[0]);
  }
}

// Sample parse consistent with gold-parse.
template <class ParsedWeights>
ArcStandardLabelledParser
ArcStandardLabelledParseModel<ParsedWeights>::particleGoldParseSentence(
    const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
    MT19937& eng, unsigned num_particles) {
  AslParserList beam_stack;
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(
      static_cast<TaggedSentence>(sent), static_cast<int>(num_particles),
      config_));

  for (unsigned i = 0; (i < sent.size()); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) {
      int num_samples = beam_stack[j]->num_particles();
      if (num_samples == 0) {
        continue;
      } else if (num_samples < 0) {
        beam_stack[j]->set_num_particles(-num_samples);
        continue;
      }

      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      int shift_count = 0;
      int reduce_count = 0;

      Real shiftp = action_probs[0];
      Real tot_reducep = log_one_min(shiftp);

      if (beam_stack[j]->stack_depth() < 2) {
        shift_count = num_samples;
        shiftp = 0;
      } else {
        kAction gold_act = beam_stack[j]->oracleNext(sent);
        WordId gold_label = beam_stack[j]->oracleNextLabel(sent);

        if (gold_act == kAction::la || gold_act == kAction::ra) {
          WordIndex reduce_pred =
              beam_stack[j]->convert_action(gold_act, gold_label);
          shift_count = std::round(std::exp(-shiftp) * num_samples);
          if (gold_act == kAction::ra)
            shift_count = 0;
          else {
            int st = beam_stack[j]->stack_top();
            bool has_right_children = true;
            for (WordIndex l = st + 1; l < beam_stack[j]->size(); ++l) {
              if (sent.has_arc(l, st) && !beam_stack[j]->has_arc(l, st)) {
                has_right_children = false;
                break;
              }
            }
            if (has_right_children) shift_count = 0;
          }

          reduce_count = num_samples - shift_count;

          if (reduce_count > 0) {
            beam_stack.push_back(
                boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (gold_act == kAction::la) {
              beam_stack.back()->leftArc(gold_label);
            } else {
              beam_stack.back()->rightArc(gold_label);
            }
            beam_stack.back()->add_particle_weight(action_probs[reduce_pred]);
            beam_stack.back()->set_num_particles(reduce_count);
          }
        } else {
          shift_count = num_samples;
        }
      }

      if (shift_count == 0) {
        beam_stack[j]->set_num_particles(0);
      } else {
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(),
                                        beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(),
                                          beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp);
        beam_stack[j]->add_particle_weight(wordp);
        beam_stack[j]->add_particle_weight(tagp);
        beam_stack[j]->add_importance_weight(wordp);
        beam_stack[j]->add_importance_weight(tagp);
        beam_stack[j]->set_num_particles(shift_count);
      }
    }

    resampleParticles(&beam_stack, eng, num_particles);
  }

  // Completion.
  for (unsigned j = 0; (j < beam_stack.size()); ++j) {
    while ((beam_stack[j]->num_particles() > 0) &&
           (beam_stack[j]->stack_depth() >= 2)) {
      Reals action_probs =
          weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();

      kAction gold_act = beam_stack[j]->oracleNext(sent);
      WordId gold_label = beam_stack[j]->oracleNextLabel(sent);

      if (gold_act == kAction::la || gold_act == kAction::ra) {
        WordIndex reduce_pred =
            beam_stack[j]->convert_action(gold_act, gold_label);
        Real reducep = action_probs[reduce_pred];

        if (gold_act == kAction::la)
          beam_stack[j]->leftArc(gold_label);
        else
          beam_stack[j]->rightArc(gold_label);
        beam_stack[j]->add_particle_weight(reducep);
        beam_stack[j]->set_num_particles(num_samples);
      } else {
        break;
      }
    }
  }

  std::sort(beam_stack.begin(), beam_stack.end(),
            TransitionParser::cmp_weighted_importance_weights);

  if (beam_stack.size() == 0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent),
                                     config_);
  } else {
    return ArcStandardLabelledParser(*beam_stack[0]);
  }
}

template <class ParsedWeights>
ArcStandardLabelledParser
ArcStandardLabelledParseModel<ParsedWeights>::staticGoldParseSentence(
    const ParsedSentence& sent,
    const boost::shared_ptr<ParsedWeights>& weights) {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(sent), config_);
  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && (a != kAction::re)) {
    a = parser.oracleNext(sent);
    WordId lab = parser.oracleNextLabel(sent);
    if (a != kAction::re) {
      WordId la = parser.convert_action(a, lab);
      Real actionp = weights->predictAction(la, parser.actionContext());
      parser.add_particle_weight(actionp);

      if (a == kAction::sh) {
        Real tagp = weights->predictTag(parser.next_tag(), parser.tagContext());
        Real wordp =
            weights->predictWord(parser.next_word(), parser.wordContext());
        parser.add_particle_weight(tagp);
        parser.add_particle_weight(wordp);
      }

      parser.executeAction(a, lab);
    }
  }

  return parser;
}

template <class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<
    ParsedWeights>::staticGoldParseSentence(const ParsedSentence& sent) {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(sent), config_);
  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && (a != kAction::re)) {
    a = parser.oracleNext(sent);
    WordId lab = parser.oracleNextLabel(sent);
    if (a != kAction::re) parser.executeAction(a, lab);
  }

  return parser;
}

template <class ParsedWeights>
ArcStandardLabelledParser
ArcStandardLabelledParseModel<ParsedWeights>::generateSentence(
    const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng,
    WordIndex sentence_id) {
  unsigned sent_limit = 100;
  ArcStandardLabelledParser parser(config_);
  bool terminate_shift = false;
  // Assuming root first.
  parser.push_tag(1);
  parser.shift(1);
  
  do {
    WordId pred = 0;  // placeholder action (shift)

    if (parser.stack_depth() >= 2) {
      if (parser.size() == sent_limit) {
        std::cout << " SENTENCE LIMITED ";
        terminate_shift = true;
      }

      Reals action_probs = weights->predictAction(parser.actionContext());

      if (terminate_shift) action_probs[0] = L_MAX;
      if (!parser.left_arc_valid()) {
        for (int k = 0; k < config_->num_labels; ++k)
          action_probs[k + 1] = L_MAX;
      }

      // Sample an action.
      multinomial_distribution_log<Real> mult(action_probs);
      pred = mult(eng);
      parser.add_particle_weight(action_probs[pred]);
    }

    kAction act = parser.lookup_action(pred);
    WordId lab = parser.lookup_label(pred);

    if (act == kAction::la) {
      parser.leftArc(lab);
      std::cout << "la ";
    } else if (act == kAction::ra) {
      parser.rightArc(lab);
      std::cout << "ra ";
    } else if (act == kAction::sh) {
      std::cout << "sh ";
      // Sample a tag.
      Reals tag_distr = weights->predictTag(parser.tagContext());
      tag_distr[0] = L_MAX;
      if (config_->root_first || (parser.size() < 2)) tag_distr[1] = L_MAX;

      multinomial_distribution_log<Real> t_mult(tag_distr);
      WordId tag = t_mult(eng);
      Real tagp = tag_distr[tag];
      parser.push_tag(tag);
      parser.add_particle_weight(tagp);

      // Sample a word.
      Reals word_distr = weights->predictWord(parser.wordContext());
      word_distr[0] = L_MAX;
      if (config_->root_first || (parser.size() <= 2)) word_distr[1] = L_MAX;

      multinomial_distribution_log<Real> w_mult(word_distr);
      WordId word = w_mult(eng);

      Real wordp = word_distr[word];
      parser.shift(word);
      parser.add_particle_weight(wordp);
      if (!config_->root_first && (word == 1)) terminate_shift = true;
    }

  } while (!parser.inTerminalConfiguration());

  std::cout << std::endl;
  return parser;
}

template <class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentence(
    const ParsedSentence& sent,
    const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardLabelledParser parse = staticGoldParseSentence(sent);
  parse.set_id(sent.id());
  parse.extractExamples(examples);
}

template <class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentence(
    const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
    const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardLabelledParser parse =
      beamParticleGoldParseSentence(sent, weights, config_->num_particles);
  parse.set_id(sent.id());
  parse.extractExamples(examples);
}

template <class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentence(
    const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
    MT19937& eng, const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardLabelledParser parse =
      particleGoldParseSentence(sent, weights, eng, config_->num_particles);
  parse.set_id(sent.id());
  parse.extractExamples(examples);
}

template <class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentenceUnsupervised(
    const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
    MT19937& eng, const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardLabelledParser parse = particleParseSentence(
      sent, weights, eng, config_->num_particles, examples);
  parse.set_id(sent.id());
  parse.extractExamples(examples);
}

template <class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentenceUnsupervised(
    const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
    const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardLabelledParser parse = beamParticleParseSentence(
      sent, weights, config_->num_particles, examples);
  parse.set_id(sent.id());
  parse.extractExamples(examples);
}

template <class ParsedWeights>
Parser ArcStandardLabelledParseModel<ParsedWeights>::evaluateSentence(
    const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights,
    const boost::shared_ptr<AccuracyCounts>& acc_counts, bool acc,
    size_t beam_size) {
  ArcStandardLabelledParser parse(config_);
  boost::shared_ptr<ParseDataSet> examples = boost::make_shared<ParseDataSet>();
  if (beam_size == 0) {
    parse = greedyParseSentence(sent, weights);
  } else {
    parse = beamParticleParseSentence(sent, weights, beam_size, examples);
  }

  if (acc) {
    acc_counts->countAccuracy(parse, sent);
    ArcStandardLabelledParser gold_parse =
        staticGoldParseSentence(sent, weights);
    acc_counts->countGoldLikelihood(parse.weight(), gold_parse.weight());
  }

  parse.set_id(sent.id());
  return parse;
}

template class ArcStandardLabelledParseModel<
    ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardLabelledParseModel<
    ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardLabelledParseModel<ParsedFactoredWeights>;
template class ArcStandardLabelledParseModel<TaggedParsedFactoredWeights>;

}  // namespace oxlm

