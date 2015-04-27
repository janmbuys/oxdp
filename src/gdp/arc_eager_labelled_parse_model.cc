#include "gdp/arc_eager_labelled_parse_model.h"

namespace oxlm {

template<class ParsedWeights>
ArcEagerLabelledParseModel<ParsedWeights>::ArcEagerLabelledParseModel(boost::shared_ptr<ModelConfig> config):
  config_(config)
{  
}

template<class ParsedWeights>
void ArcEagerLabelledParseModel<ParsedWeights>::reallocateParticles(AelParserList* beam_stack, unsigned num_particles) {
  std::vector<Real> weight(beam_stack->size(), L_MAX);
  Real sum = beam_stack->at(0)->weighted_particle_weight();
  for (unsigned i = 0; i < beam_stack->size(); ++i) 
    if (beam_stack->at(i)->num_particles() > 0) {
      weight[i] = beam_stack->at(i)->weighted_particle_weight();
      if (i > 0)
        sum = neg_log_sum_exp(sum, weight[i]);
    }

  std::vector<int> sample_counts(beam_stack->size(), 0);
  unsigned best_i = 0;
  Real best_weight = beam_stack->at(0)->particle_weight();

  for (unsigned i = 0; i < beam_stack->size(); ++i) 
    if (beam_stack->at(i)->num_particles() > 0) {
      sample_counts[i] = (int)(std::exp(sum - beam_stack->at(i)->weighted_particle_weight())*num_particles);
      if (beam_stack->at(i)->particle_weight() < best_weight) {
        best_weight = beam_stack->at(i)->particle_weight();
        best_i = i;
      }
    }
  
  for (unsigned i = 0; i < beam_stack->size(); ++i) {
    if (beam_stack->at(i)->num_particles() > 0) 
      beam_stack->at(i)->set_num_particles(sample_counts[i]);
  }
}

template<class ParsedWeights>
void ArcEagerLabelledParseModel<ParsedWeights>::resampleParticleParticles(AelParserList* beam_stack, MT19937& eng,
        unsigned num_particles) {
  std::vector<Real> importance_w(beam_stack->size(), L_MAX); 
  bool valid_sample = false;
  for (unsigned i = 0; i < importance_w.size(); ++i) 
    if (beam_stack->at(i)->num_particles() > 0) {
      importance_w[i] = beam_stack->at(i)->weighted_particle_weight();
      valid_sample = true;
    }

  //resample according to particle weight
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

template<class ParsedWeights>
void ArcEagerLabelledParseModel<ParsedWeights>::resampleParticles(AelParserList* beam_stack, MT19937& eng,
        unsigned num_particles) {
  std::vector<Real> importance_w(beam_stack->size(), L_MAX); 
  bool valid_sample = false;

  for (unsigned i = 0; i < importance_w.size(); ++i) 
    if (beam_stack->at(i)->num_particles() > 0) {
      importance_w[i] = beam_stack->at(i)->weighted_importance_weight();
      valid_sample = true;
    }

  //resample according to importance weight
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

template<class ParsedWeights>
ArcEagerLabelledParser ArcEagerLabelledParseModel<ParsedWeights>::greedyParseSentence(const ParsedSentence& sent, 
                           const boost::shared_ptr<ParsedWeights>& weights) {
  ArcEagerLabelledParser parser(static_cast<TaggedSentence>(sent), config_);

  while (!parser.buffer_empty()) {
    Reals action_probs = weights->predictAction(parser.actionContext());
    if (!parser.left_arc_valid()) 
      for (unsigned i = 1; i <= config_->num_labels; ++i)
        action_probs[i] = L_MAX;
    if (!parser.reduce_valid())
      action_probs[action_probs.size()-1] = L_MAX;

    WordIndex pred = arg_min(action_probs, 0);

    if (parser.stack_empty())
      pred = 0; 
    //for root last, enforce reduces so that the parse forms a tree
    else if (!config_->root_first && (parser.buffer_next() == 0)) {
      //left-arc or reduce allowed
      if (parser.left_arc_valid())
        pred = arg_min(action_probs, 1, config_->num_labels + 1);
      else if (parser.reduce_valid())
        pred = action_probs.size() - 1;
      else 
        pred = 0; //cannot form a tree
    } else if (pred == 0)
      parser.add_particle_weight(action_probs[pred]);

    //reduce until a shift action is chosen
    while (pred > 0) {
      kAction re_act = parser.lookup_action(pred);
      WordId re_label = parser.lookup_label(pred);
      if (re_act == kAction::la) 
        parser.leftArc(re_label);
      else if (re_act == kAction::ra) 
        parser.rightArc(re_label);
      else
        parser.reduce();
      parser.add_particle_weight(action_probs[pred]);

      action_probs = weights->predictAction(parser.actionContext());
      if (!parser.left_arc_valid()) 
        for (unsigned i = 1; i <= config_->num_labels; ++i)
          action_probs[i] = L_MAX;
      if (!parser.reduce_valid())
        action_probs[action_probs.size()-1] = L_MAX;

      pred = arg_min(action_probs, 0);

      if ((re_act == kAction::ra) || parser.stack_empty())
        pred = 0; 
      else if (!config_->root_first && (parser.buffer_next() == 0)) {
      if (parser.left_arc_valid())
        pred = arg_min(action_probs, 1, config_->num_labels + 1);
      else if (parser.reduce_valid())
        pred = action_probs.size() - 1;
      else 
        pred = 0; //cannot form a tree
      } else if (pred == 0)
        parser.add_particle_weight(action_probs[pred]);
    }

    //shift    
    Real tagp = weights->predictTag(parser.next_tag(), parser.tagContext());
    Real wordp = weights->predictWord(parser.next_word(), parser.wordContext());
    parser.shift();
    parser.add_particle_weight(tagp);
    parser.add_particle_weight(wordp);
  }

  //no final reduce 
  //parser.print_actions();
  return parser;
}

template<class ParsedWeights>
ArcEagerLabelledParser ArcEagerLabelledParseModel<ParsedWeights>::beamDiscriminativeParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, unsigned beam_size) {
  AelParserList beam_stack; 
  //TODO

  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
  return ArcEagerLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else {
    return ArcEagerLabelledParser(*beam_stack[0]); 
  }

}

template<class ParsedWeights>
ArcEagerLabelledParser ArcEagerLabelledParseModel<ParsedWeights>::beamParseSentence(const ParsedSentence& sent, 
                           const boost::shared_ptr<ParsedWeights>& weights, unsigned beam_size) {
  AelParserList beam_stack; 

  beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(static_cast<TaggedSentence>(sent), config_)); 

  for (unsigned i = 0; (i < sent.size()); ++i) {
    //TODO test if worst_weight helps  -> causes a bug here
    double worst_weight = beam_stack[beam_stack.size() - 1]->particle_weight();

    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      //reduce actions are direction deterministic
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      Real shiftp = action_probs[0];
      if (beam_stack[j]->num_particles() == 0)
        continue;

      if ((!beam_stack[j]->stack_empty()) && (beam_stack[j]->last_action() != kAction::ra)) {
         // && ((j < beam_size) || (!config_->root_first && (i == sent.size() -1)) || (beam_stack[j]->particle_weight() < worst_weight))) { 
        //add reduce actions
        if (beam_stack[j]->reduce_valid()) {
          Real reducep = action_probs.back();
          beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));
          beam_stack.back()->reduce();
          beam_stack.back()->add_particle_weight(reducep);
        } else if (beam_stack[j]->left_arc_valid()) {
          //sort to find the best left-arc actions
          std::vector<int> indices(config_->num_labels);
          std::iota(indices.begin(), indices.end(), 1);
          std::sort(indices.begin(), indices.end(), [&action_probs](const int i, const int j) 
                {return (action_probs[i] < action_probs[j]);});
          for (unsigned l = 0; (l < beam_size && l < config_->max_beam_increment); ++l) {
            WordIndex reduce_pred = indices[l];
            Real reducep = action_probs[reduce_pred];
            WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
            //don't add hypothesis guaranteed to be off the beam
            //if ((j < beam_size) || (beam_stack[j]->particle_weight() + reducep < worst_weight)) {
              beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));
              beam_stack.back()->leftArc(re_label);
              beam_stack.back()->add_particle_weight(reducep);
            //}
          }
        } 

        //right-arcs added and shifted when reached later in loop
        if (config_->root_first || (i < sent.size() - 1)) {
          std::vector<int> indices(config_->num_labels);
          std::iota(indices.begin(), indices.end(), config_->num_labels + 1);
          std::sort(indices.begin(), indices.end(), [&action_probs](const int i, const int j) 
              {return (action_probs[i] < action_probs[j]);});
          for (unsigned l = 0; (l < beam_size && l < config_->max_beam_increment); ++l) {
            WordIndex reduce_pred = indices[l];
            Real reducep = action_probs[reduce_pred];
            WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
            //don't add hypothesis guaranteed to be off the beam
            //if ((j < beam_size) || (beam_stack[j]->particle_weight() + reducep < worst_weight)) {
              beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));
              beam_stack.back()->rightArc(re_label);
              beam_stack.back()->add_particle_weight(reducep);
            //}
          }
        }
      } else {
        shiftp = 0;
      }

      if ((beam_stack[j]->num_particles() > 0) && (beam_stack[j]->particle_weight() > worst_weight))
        worst_weight = beam_stack[j]->particle_weight();

      if (config_->root_first || !config_->complete_parse ||  (i < sent.size() - 1) || beam_stack[j]->stack_empty()) {
        //shift
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());
        //std::cout << "(" <<  beam_stack[j]->next_tag() << ": " << tagp << ", " 
        //       <<  beam_stack[j]->next_word() << ": " << wordp << ") ";

        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
      } else {
        beam_stack[j]->set_num_particles(0);
      }
    }

    //prune the beam
    if (beam_stack.size() > beam_size) {
      std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
      for (int j = beam_stack.size()- 1; ((j >= beam_size) || ((j >= 0) && (beam_stack[j]->num_particles() == 0))); --j)
        beam_stack.pop_back();
    }
   
    //std::cout << std::endl;
    //for (unsigned j = 0; j < beam_stack.size(); ++j)  
    //  std::cout << beam_stack[j]->particle_weight() << " ";
    //std::cout << std::endl;      
   }

  //sum over identical parses in final beam 
  vector<bool> duplicate(beam_stack.size(), false);
  if (config_->sum_over_beam) {
    for (unsigned i = 0; (i < beam_stack.size()-1); ++i) {
      if (!duplicate[i])
        for (unsigned j = i + 1; (j < beam_stack.size()); ++j) {
          if (ParsedSentence::eq_arcs(beam_stack[i], beam_stack[j])) {
            beam_stack[i]->add_log_particle_weight(beam_stack[j]->particle_weight());          
            duplicate[j] = true;
          }
        }
    } 
  }

  //eliminate incomplete parses in root-first parser
  if (config_->root_first) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      while ((beam_stack[j]->num_particles() > 0) && (beam_stack[j]->stack_depth() >= 2)) {
        if (beam_stack[j]->reduce_valid()) {
          Real reducep = weights->predictAction(config_->numActions()-1, beam_stack[j]->actionContext());
          beam_stack[j]->reduce();
          beam_stack[j]->add_particle_weight(reducep);
        } else {
          beam_stack[j]->set_num_particles(0);
        }
      }
    }
  }

  if (beam_stack.size() > 0)
    std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 

  //compute beam weight
  for (unsigned i = 0; (i < beam_stack.size()); ++i)
    if (!duplicate[i] && (beam_stack[i]->num_particles() > 0)) 
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight()); 

  if (beam_stack.size()==0 || beam_stack[0]->num_particles() == 0) {
    std::cout << "no parse found" << std::endl;
    return ArcEagerLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else {
    return ArcEagerLabelledParser(*beam_stack[0]); 
  }
}

template<class ParsedWeights>
ArcEagerLabelledParser ArcEagerLabelledParseModel<ParsedWeights>::beamParticleParseSentence(
        const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights, unsigned num_particles) {
  AelParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(static_cast<TaggedSentence>(sent),
            static_cast<int>(num_particles), config_)); 

  for (unsigned i = 0; (i < sent.size()); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      int num_samples = beam_stack[j]->num_particles();
      if (num_samples == 0)
        continue;

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int shift_count = 0;
      Real shiftp = action_probs[0];

      //least effort now to affort to leak some prob mass to simplify things
      if ((!beam_stack[j]->stack_empty()) && (beam_stack[j]->last_action() != kAction::ra)) {
        //0 prob to invalid transitions
        if (!beam_stack[j]->reduce_valid())
          action_probs.back() = L_MAX;
        if (!beam_stack[j]->left_arc_valid()) {
          for (unsigned l = 0; l < config_->num_labels; ++l)
            action_probs[l+1] = L_MAX;
        }
        if (!config_->root_first && (i == sent.size() - 1)) {
          for (unsigned l = 0; l < config_->num_labels; ++l)
            action_probs[config_->num_labels+l+1] = L_MAX;
        }

        Real reducep = L_MAX;
        int reduce_count = 0;

        std::vector<int> action_counts(action_probs.size(), 0); 
        if (beam_stack[j]->left_arc_valid()) {
          for (unsigned l = 1; l < config_->num_labels + 1; ++l) {
            action_counts[l] = std::floor(std::exp(-action_probs[l])*num_samples);
            reducep = neg_log_sum_exp(reducep, action_probs[l]);
            reduce_count += action_counts[l];
          }
        }
        if (beam_stack[j]->reduce_valid()) {
          action_counts.back() = std::floor(std::exp(-action_probs.back())*num_samples);
          reducep = neg_log_sum_exp(reducep, action_probs.back());
          reduce_count += action_counts.back();
        }
        
        int res_reduce_count = std::round(std::exp(-reducep)*num_samples) - reduce_count;

        //only reduce
        if (!config_->root_first && (i == sent.size() - 1) && config_->complete_parse) {
          res_reduce_count = num_samples - reduce_count;
        }

        if (res_reduce_count > 0) {
          if (beam_stack[j]->left_arc_valid()) {
            unsigned max_la = arg_max(action_probs, 1, config_->num_labels + 1);
            action_counts[max_la] += res_reduce_count;
          } else if (beam_stack[j]->reduce_valid()) {
            action_counts.back() += res_reduce_count;
          }
        }

        //right arc
        if (config_->root_first || (i < sent.size() - 1) || !config_->complete_parse) {
          for (unsigned l = config_->num_labels + 1; l < 2*config_->num_labels + 1; ++l)
            action_counts[l] = std::floor(std::exp(-action_probs[l])*num_samples);
        }
        shift_count = num_samples - std::accumulate(action_counts.begin(), action_counts.end(), 0);
        //std::cout << num_samples << "," << shift_count << " ";

        //add reduce actions
        if (action_counts.back() > 0) {
          Real reducep = action_probs.back();
          beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));
          beam_stack.back()->reduce();
          beam_stack.back()->add_particle_weight(reducep);
          beam_stack.back()->set_num_particles(action_counts.back());
        } 
        
        //add left-arcs
        for (unsigned l = 1; l < config_->num_labels + 1; ++l) {
          if (action_counts[l] > 0) {
            Real reducep = action_probs[l];
            WordId re_label = beam_stack[j]->lookup_label(l);
            beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));
            beam_stack.back()->leftArc(re_label);
            beam_stack.back()->add_particle_weight(reducep);
            beam_stack.back()->set_num_particles(action_counts[l]);
          }
        }

        //right-arcs added and shifted when reached later in loop
        for (unsigned l = config_->num_labels + 1; l < 2*config_->num_labels + 1; ++l) {
          if (action_counts[l] > 0) {
            Real reducep = action_probs[l];
            WordId re_label = beam_stack[j]->lookup_label(l);
            beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));
            beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(reducep);
            beam_stack.back()->set_num_particles(action_counts[l]);
          }
        }
      } else {
        shift_count = num_samples;
        shiftp = 0;
      }

      if ((shift_count > 0) && 
               (config_->root_first || !config_->complete_parse || (i < sent.size() - 1) || beam_stack[j]->stack_empty())) {
        //shift
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->set_num_particles(shift_count);
      } else {
        beam_stack[j]->set_num_particles(0);
        //std::cout << j << " ";
      }
    }

    reallocateParticles(&beam_stack, num_particles);
    //std::cout << std::endl;
  }

  //eliminate incomplete parses in root-first parser
  if (config_->root_first && config_->complete_parse) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      while ((beam_stack[j]->num_particles() > 0) && (beam_stack[j]->stack_depth() >= 2)) {
        if (beam_stack[j]->reduce_valid()) {
          Real reducep = weights->predictAction(config_->numActions()-1, beam_stack[j]->actionContext());
          beam_stack[j]->reduce();
          beam_stack[j]->add_particle_weight(reducep);
        } else {
          beam_stack[j]->set_num_particles(0);
        }
      }
    }
  }

  //sum over identical parses in final beam 
  vector<bool> duplicate(beam_stack.size(), false);
  if (config_->sum_over_beam) {
    for (unsigned i = 0; (i < beam_stack.size()-1); ++i) {
      if (!duplicate[i])
        for (unsigned j = i + 1; (j < beam_stack.size()); ++j) {
          if (ParsedSentence::eq_arcs(beam_stack[i], beam_stack[j])) {
            beam_stack[i]->add_log_particle_weight(beam_stack[j]->particle_weight());          
            duplicate[j] = true;
          }
        }
    } 
  }

  if (beam_stack.size() > 0)
    std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 

  //compute beam weight
  //std::cout << "===\n";
  for (unsigned i = 0; (i < beam_stack.size()); ++i) {
    //std::cout << beam_stack[i]->num_particles() << " ";
    if (!duplicate[i] && (beam_stack[i]->num_particles() > 0)) 
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight()); 
  }
  //std::cout << std::endl;

  if (beam_stack.size()==0 || beam_stack[0]->num_particles() == 0) {
    //std::cout << "no parse found" << std::endl;
    return ArcEagerLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else {
    return ArcEagerLabelledParser(*beam_stack[0]); 
  }

  if (beam_stack.size()==0) {
    //std::cout << "no parse found" << std::endl;
  return ArcEagerLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else {
    return ArcEagerLabelledParser(*beam_stack[0]); 
  }
}

template<class ParsedWeights>
ArcEagerLabelledParser ArcEagerLabelledParseModel<ParsedWeights>::staticGoldParseSentence(const ParsedSentence& sent,
                                        const boost::shared_ptr<ParsedWeights>& weights) {
  ArcEagerLabelledParser parser(static_cast<TaggedSentence>(sent), config_);

  kAction a = kAction::sh;
  while (!parser.buffer_empty()) {
    a = parser.oracleNext(sent);
    WordId lab = parser.oracleNextLabel(sent);
    WordId la = parser.convert_action(a, lab);
    Real actionp = weights->predictAction(la, parser.actionContext());
    parser.add_particle_weight(actionp);

    if (a != kAction::sh) 
      parser.executeAction(a, lab); 

    if (a == kAction::sh || a == kAction::ra) {
      Real tagp = weights->predictTag(parser.next_tag(), parser.tagContext());
      Real wordp = weights->predictWord(parser.next_word(), parser.wordContext());
      parser.shift();
      parser.add_particle_weight(tagp);
      parser.add_particle_weight(wordp);
    }
      
  }

  return parser;
}

template<class ParsedWeights>
ArcEagerLabelledParser ArcEagerLabelledParseModel<ParsedWeights>::staticGoldParseSentence(const ParsedSentence& sent) {
  ArcEagerLabelledParser parser(static_cast<TaggedSentence>(sent), config_);
  kAction a = kAction::sh;
  while (!parser.buffer_empty()) {
    a = parser.oracleNext(sent);
    WordId lab = parser.oracleNextLabel(sent);
    if (a != kAction::sh) 
      parser.executeAction(a, lab); 
    if (a == kAction::sh || a == kAction::ra) 
      parser.shift();
  }

  return parser;
}

template<class ParsedWeights>
ArcEagerLabelledParser ArcEagerLabelledParseModel<ParsedWeights>::generateSentence(const boost::shared_ptr<ParsedWeights>& weights, 
        MT19937& eng) {
  ArcEagerLabelledParser parser(config_);
/*  unsigned sent_limit = 100;
  bool terminate_shift = false;
  //bool need_shift = false;
  parser.push_tag(0);
  parser.shift(0);
    
  do {
    kAction a = kAction::sh; //placeholder action
    if (parser.size() >= sent_limit) {
        // check to upper bound sentence length
        //if (!terminate_shift)
        //  std::cout << " LENGTH LIMITED ";
        terminate_shift = true;
        a = kAction::re;
    } else {
      Words r_ctx = parser.actionContext();
      Real shiftp = weights->predictAction(static_cast<WordId>(kAction::sh), r_ctx);
      Real leftarcreducep = weights->predictAction(static_cast<WordId>(kAction::la), r_ctx);
      Real rightarcshiftp = weights->predictAction(static_cast<WordId>(kAction::ra), r_ctx);
      Real reducep = weights->predictAction(static_cast<WordId>(kAction::re), r_ctx); 

      if (!parser.left_arc_valid())
        leftarcreducep = L_MAX;
      if (!parser.reduce_valid())
        reducep = L_MAX;
      //if (leftarcreducep==0 && reducep==0)
      //  std::cout << "[ra:" << rightarcshiftp << " sh:" << shiftp << "] ";

      //sample an action
      std::vector<Real> distr = {shiftp, leftarcreducep, rightarcshiftp, reducep};
      multinomial_distribution_log<Real> mult(distr); 
      WordId act = mult(eng);
      //std::cout << "(" << parser.stack_depth() << ") ";
      //std::cout << act << " ";
      parser.add_particle_weight(distr[act]);
      
      if (act==0) {
        a = kAction::sh;
      } else if (act==1) {
        a = kAction::la; 
        parser.leftArc(0);
        //need_shift = true;
      } else if (act==2) {
        a = kAction::ra;
      } else {
        a = kAction::re;
        parser.reduce();
      }
    } 

    if ((a == kAction::sh) || (a == kAction::ra)) {
      //need_shift = false;
      //sample a tag - disallow root tag
      Words t_ctx = parser.tagContext(a);
      std::vector<Real> t_distr(weights->numTags() - 1, L_MAX);
      for (WordId w = 1; w < weights->numTags(); ++w) 
        t_distr[w-1] = weights->predictTag(w, t_ctx); 
      multinomial_distribution_log<Real> t_mult(t_distr);
      WordId tag = t_mult(eng) + 1;

      Real tagp = weights->predictTag(tag, t_ctx); 
      parser.push_tag(tag);
      parser.add_particle_weight(tagp);

      //sample a word 
      Words w_ctx = parser.wordContext();
      std::vector<Real> w_distr(weights->numWords(), 0);

      w_distr[0] = weights->predictWord(-1, w_ctx); //unk probability
      for (WordId w = 1; w < weights->numWords(); ++w) 
        w_distr[w] = weights->predictWord(w, w_ctx); 
      multinomial_distribution_log<Real> w_mult(w_distr);
      WordId word = w_mult(eng);
      if (word==0)
        word = -1;

      Real wordp = weights->predictWord(word, w_ctx); 
      parser.add_particle_weight(wordp);
      
      //perform action
      if (a == kAction::ra) {
        parser.rightArc(word);
      } else {
        parser.shift(word);
      }
    } //extra condition that left-arc needs a shift following
    //if (parser.inTerminalConfiguration())
    //  std::cout << ":" << parser.stack_depth() << " ";

  } while (!parser.inTerminalConfiguration() && !terminate_shift);
  //std::cout << std::endl;
 */
  return parser;
}

template<class ParsedWeights>
void ArcEagerLabelledParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcEagerLabelledParser parse = staticGoldParseSentence(sent); 
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcEagerLabelledParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcEagerLabelledParser parse = staticGoldParseSentence(sent, weights); 
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcEagerLabelledParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng,
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcEagerLabelledParser parse = staticGoldParseSentence(sent, weights); 
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcEagerLabelledParseModel<ParsedWeights>::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng,
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcEagerLabelledParser parse = beamParticleParseSentence(sent, weights, config_->num_particles);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcEagerLabelledParseModel<ParsedWeights>::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcEagerLabelledParser parse = beamParticleParseSentence(sent, weights, config_->num_particles);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
Parser ArcEagerLabelledParseModel<ParsedWeights>::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          bool acc,
          size_t beam_size) {
  ArcEagerLabelledParser parse(config_);
  if (beam_size == 0)
    parse = greedyParseSentence(sent, weights);
  else
    parse = beamParticleParseSentence(sent, weights, beam_size);
    //parse = beamParseSentence(sent, weights, beam_size);
    //parse = beamParticleParseSentence(sent, weights, beam_size);
  //std::cout << parse.size() << std::endl;
  //parse.print_arcs();
  //parse.print_actions();

  //resimulate the computation, recompute weight
/*  ArcEagerLabelledParser simul(static_cast<TaggedSentence>(parse), config_); 
  Real particle_weight = 0;

  for (unsigned i = 0; i < parse.actions().size(); ++i) {
    kAction a = parse.actions().at(i);
    WordId lab = parse.action_label_at(i);
    WordId lab_act = parse.convert_action(a, lab);
    particle_weight += weights->predictAction(lab_act, simul.actionContext());
    
    if (a == kAction::sh) {
      particle_weight += weights->predictTag(simul.next_tag(), simul.tagContext());
      particle_weight += weights->predictWord(simul.next_word(), simul.wordContext());
    }

    //kAction next = simul.oracleNext(gold_parse);
    //WordId nextLabel = simul.oracleNextLabel(gold_parse);
    simul.executeAction(a, lab);
  }
  //std::cout << parse.weight() << ": " << particle_weight << std::endl;
*/
  if (acc) {
    acc_counts->countAccuracy(parse, sent);
    ArcEagerLabelledParser gold_parse = staticGoldParseSentence(sent, weights);
  
    acc_counts->countGoldLikelihood(parse.weight(), gold_parse.weight());
  }
  return parse;
}

template<class ParsedWeights>
Parser ArcEagerLabelledParseModel<ParsedWeights>::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) {
  ArcEagerLabelledParser parse(config_);
  if (beam_size == 0)
    parse = greedyParseSentence(sent, weights);
  else
    parse = beamParticleParseSentence(sent, weights, beam_size);
  acc_counts->countAccuracy(parse, sent);
  ArcEagerLabelledParser gold_parse = staticGoldParseSentence(sent, weights);
  
  acc_counts->countGoldLikelihood(parse.weight(), gold_parse.weight());
  return parse;
}

/*
template<class ParsedWeights>
ArcEagerLabelledParser ArcEagerLabelledParseModel<ParsedWeights>::particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles,
        bool resample) {
//TODO
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states

  AelParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    unsigned shift_beam_size = beam_stack.size();
    //std::cout << i << ": " << shift_beam_size << " ";
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if ((beam_stack[j]->num_particles()==0) || 
          ((j >= shift_beam_size) && (beam_stack[j]->last_action() == kAction::ra)))
        continue;
       
      //sample a sequence of possible actions leading up to the next shift or ra

      int num_samples = beam_stack[j]->num_particles();

      Words r_ctx = beam_stack[j]->actionContext();
      Real shiftp = weights->predictAction(static_cast<WordId>(kAction::sh), r_ctx);
      Real leftarcreducep = weights->predictAction(static_cast<WordId>(kAction::la), r_ctx);
      Real rightarcshiftp = weights->predictAction(static_cast<WordId>(kAction::ra), r_ctx);
      Real reducep = weights->predictAction(static_cast<WordId>(kAction::re), r_ctx); 
      //Real totalreducep = neg_log_sum_exp(reducep, leftarcreducep);

      std::vector<int> sample_counts = {0, 0, 0, 0}; //shift, la, ra, re
      
      if (!beam_stack[j]->left_arc_valid())
        leftarcreducep = L_MAX;
      if (!beam_stack[j]->reduce_valid())
        reducep = L_MAX;

      std::vector<Real> distr = {shiftp, leftarcreducep, rightarcshiftp, reducep};
      multinomial_distribution_log<Real> mult(distr); 
      for (int k = 0; k < num_samples; k++) {
        WordId act = mult(eng);
        ++sample_counts[act];
      }
      
      //for reduce actions: 
      if (sample_counts[1] > 0) {
        beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));
        beam_stack.back()->leftArc(0);
        beam_stack.back()->add_particle_weight(leftarcreducep); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      if (sample_counts[3] > 0) {
        beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));
        beam_stack.back()->reduce();
        beam_stack.back()->add_particle_weight(reducep); 
        beam_stack.back()->set_num_particles(sample_counts[3]); 
      }

      //for shift actions
      if (sample_counts[2] > 0) {
        beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));

        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), 
                    beam_stack[j]->tagContext(kAction::ra));
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

        beam_stack.back()->rightArc(0);
        beam_stack.back()->add_importance_weight(wordp); 
        beam_stack.back()->add_importance_weight(tagp); 
        beam_stack.back()->add_particle_weight(rightarcshiftp); 
        beam_stack.back()->add_particle_weight(wordp); 
        beam_stack.back()->add_particle_weight(tagp); 
        beam_stack.back()->set_num_particles(sample_counts[2]);
      } 
      
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), 
                    beam_stack[j]->tagContext(kAction::sh));
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      if (beam_stack.size() > 0)
        resampleParticles(&beam_stack, eng, num_particles);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
     
  ///completion
  bool has_more_states = true;
  //std::cout << " compl " << std::endl;

  while (has_more_states) {
    has_more_states = false;
    unsigned cur_beam_size = beam_stack.size();
    //std::cout << cur_beam_size << ": ";

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->inTerminalConfiguration()) {
        //add paths for reduce actions
        //std::cout << beam_stack[j]->stack_depth() << "," << beam_stack[j]->num_particles() << " ";
        has_more_states = true; 
        Real reducep = weights->predictAction(static_cast<WordId>(kAction::re), beam_stack[j]->actionContext());
        
        if (beam_stack[j]->reduce_valid()) {
          beam_stack[j]->reduce();
          beam_stack[j]->add_particle_weight(reducep); 
          beam_stack[j]->add_importance_weight(reducep); 
        } else {
          beam_stack[j]->set_num_particles(0);
        }
      }
    }
    //std::cout << std::endl;

    //no point in resampling here
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cout << "Final beam size: " << beam_stack.size() << std::endl;

  if (beam_stack.size() > 0) {
    //just take 1 sample
    resampleParticles(&beam_stack, eng, 1);
    for (unsigned i = 0; i < beam_stack.size(); ++i) {
      if (beam_stack[i]->num_particles() == 1) {
        //beam_stack[i]->print_arcs();
        //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
        //std::cout << "  Dir Accuracy: " << dir_acc;
        //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;
        return ArcEagerLabelledParser(*beam_stack[i], config_->num_labels); 
      }
    }
  }

  //std::cout << "no parse found" << std::endl;
  return ArcEagerLabelledParser(static_cast<TaggedSentence>(sent), config_->num_labels);  
}

//4-way decisions
template<class ParsedWeights>
ArcEagerLabelledParser ArcEagerLabelledParseModel<ParsedWeights>::particleGoldParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, 
          unsigned num_particles, bool resample) {
  //TODO
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states

  AelParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift

      int num_samples = beam_stack[j]->num_particles();

      Words r_ctx = beam_stack[j]->actionContext();
      Real shiftp = weights->predictAction(static_cast<WordId>(kAction::sh), r_ctx);
      Real leftarcreducep = weights->predictAction(static_cast<WordId>(kAction::la), r_ctx);
      Real rightarcshiftp = weights->predictAction(static_cast<WordId>(kAction::ra), r_ctx);
      Real reducep = weights->predictAction(static_cast<WordId>(kAction::re), r_ctx); 
      Real noarcp = neg_log_sum_exp(reducep, shiftp);
      //Real reducep = neg_log_sum_exp(reduceleftarcp, reducerightarcp); 
      
      kAction oracle_next = beam_stack[j]->oracleNext(sent);
      //only ambiguity is if oracle_next==re - sh also allowed

      if (oracle_next==kAction::la) {
        beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));
        beam_stack.back()->leftArc(0);
        beam_stack.back()->add_particle_weight(leftarcreducep); 
        beam_stack.back()->add_importance_weight(leftarcreducep); 
        beam_stack[j]->set_num_particles(0);
      } else if (oracle_next==kAction::ra) {
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), 
                beam_stack[j]->tagContext(kAction::ra));
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());
        
        beam_stack[j]->rightArc(0);
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(rightarcshiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
      } else {
        if (oracle_next==kAction::re) {
          //enforce at least 1 particle to reduce
          std::vector<int> sample_counts = {0, 1}; //shift, reduce
          std::vector<Real> distr = {shiftp, reducep};
          multinomial_distribution_log<Real> mult(distr); 
          for (int k = 1; k < num_samples; k++) {
            WordId act = mult(eng);
            ++sample_counts[act];
          }

          if (sample_counts[1] > 0) {
            beam_stack.push_back(boost::make_shared<ArcEagerLabelledParser>(*beam_stack[j]));
            beam_stack.back()->reduce();
            beam_stack.back()->add_particle_weight(reducep); 
            beam_stack.back()->add_importance_weight(reducep - noarcp);
            beam_stack.back()->set_num_particles(sample_counts[1]); 
            beam_stack[j]->set_num_particles(sample_counts[0]);
            beam_stack[j]->add_importance_weight(shiftp - noarcp);
          }  
          
        }

        //shift allowed
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), 
                beam_stack[j]->tagContext(kAction::sh));
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp);
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
      }
    }
       
    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      if (beam_stack.size() > 0)
        resampleParticles(&beam_stack, eng, num_particles);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //   ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }
  
  //std::cout << "Beam size: " << beam_stack.size();
  int active_particle_count = 0;
  for (int j = 0; j < beam_stack.size(); ++j)
    if (beam_stack[j]->num_particles() > 0)
      ++active_particle_count;
  //std::cout << " -> " << active_particle_count << " without null \n";

  ///completion
  bool has_more_states = true;

  while (has_more_states) {
    has_more_states = false;
    //unsigned cur_beam_size = beam_stack.size();
    //std::cout << cur_beam_size << ": ";

    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->inTerminalConfiguration()) {
        //add paths for reduce actions
        has_more_states = true; 
        //include or not?
        Real reducep = weights->predictAction(static_cast<WordId>(kAction::re), beam_stack[j]->actionContext());
       
        kAction oracle_next = beam_stack[j]->oracleNext(sent);
        if (oracle_next==kAction::re) {
          beam_stack[j]->reduce();
          //beam_stack[j]->add_particle_weight(reducep); 
          //beam_stack[j]->add_importance_weight(reducep); 
        } else {
          beam_stack[j]->set_num_particles(0);
        }
      }
    }
    //std::cerr << std::endl;
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cerr << beam_stack.size() << " ";

  //just take 1 sample
  if (beam_stack.size() > 0)
    resampleParticles(&beam_stack, eng, 1);
  for (unsigned i = 0; i < beam_stack.size(); ++i) {
    if (beam_stack[i]->num_particles() == 1) {
      //beam_stack[i]->print_arcs();
      //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
      //std::cout << "  Dir Accuracy: " << dir_acc;
      //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;

      return ArcEagerLabelledParser(*beam_stack[i], config_->num_labels); 
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcEagerLabelledParser(static_cast<TaggedSentence>(sent), config_->num_labels);  
}
*/

template class ArcEagerLabelledParseModel<ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>;
template class ArcEagerLabelledParseModel<ParsedChLexPypWeights<wordLMOrderAE, charLMOrder, tagLMOrderAE, actionLMOrderAE>>;
template class ArcEagerLabelledParseModel<ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>;
template class ArcEagerLabelledParseModel<ParsedFactoredWeights>;
template class ArcEagerLabelledParseModel<ParsedWeights>;

}

