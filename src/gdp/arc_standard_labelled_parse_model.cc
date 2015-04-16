#include "gdp/arc_standard_labelled_parse_model.h"

namespace oxlm {

template<class ParsedWeights>
ArcStandardLabelledParseModel<ParsedWeights>::ArcStandardLabelledParseModel(boost::shared_ptr<ModelConfig> config):
  config_(config)
{
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::reallocateParticles(AslParserList* beam_stack, unsigned num_particles) {
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
  
  //TODO test with(out) this 
  //sample_counts[best_i] += num_particles - std::accumulate(sample_counts.begin(), sample_counts.end(), 0);

  for (unsigned i = 0; i < beam_stack->size(); ++i) {
    if (beam_stack->at(i)->num_particles() > 0) 
      beam_stack->at(i)->set_num_particles(sample_counts[i]);
  }
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::resampleParticleParticles(AslParserList* beam_stack, MT19937& eng,
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
void ArcStandardLabelledParseModel<ParsedWeights>::resampleParticles(AslParserList* beam_stack, MT19937& eng,
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
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::greedyParseSentence(
        const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights) {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(sent), config_);

  //for (unsigned k = 0; k < sent.size(); ++k) {
  while (!parser.buffer_empty()) {
    Reals action_probs = weights->predictAction(parser.actionContext());
    WordIndex pred = arg_min(action_probs, 0);
    if ((parser.stack_depth() < 2) || (config_->root_first && (parser.stack_depth() == 2)))
      pred = 0; 
    //for root last, enforce reduces so that the parse forms a tree
    else if (!config_->root_first && (parser.buffer_next() == 0)) 
      pred = arg_min(action_probs, 1);
    else if (pred == 0)
      parser.add_particle_weight(action_probs[pred]);

    //reduce until a shift action is chosen
    while (pred > 0) {
      kAction re_act = parser.lookup_action(pred);
      WordId re_label = parser.lookup_label(pred);
      if (re_act == kAction::la) 
        parser.leftArc(re_label);
	  else
        parser.rightArc(re_label);
      parser.add_particle_weight(action_probs[pred]);

      action_probs = weights->predictAction(parser.actionContext());
      pred = arg_min(action_probs, 0);
      if ((parser.stack_depth() < 2) || (config_->root_first && (parser.stack_depth() == 2)))
        pred = 0;
      else if (!config_->root_first && (parser.buffer_next() == 0)) 
        pred = arg_min(action_probs, 1);
      else if (pred == 0)
        parser.add_particle_weight(action_probs[pred]);
    }
    
    //shift    
    Real tagp = weights->predictTag(parser.next_tag(), parser.tagContext());
    Real wordp = weights->predictWord(parser.next_word(), parser.wordContext());
    parser.shift();
    parser.add_particle_weight(tagp);
    parser.add_particle_weight(wordp);
  }

  //final reduce actions
  while (!parser.inTerminalConfiguration()) {
    Reals action_probs = weights->predictAction(parser.actionContext());
    WordIndex pred = arg_min(action_probs, 1);
    if (!parser.left_arc_valid()) 
      pred = arg_min(action_probs, config_->num_labels + 1);
    kAction re_act = parser.lookup_action(pred);
    WordId re_label = parser.lookup_label(pred);
    
    if (re_act == kAction::la) 
      parser.leftArc(re_label);
	else
      parser.rightArc(re_label);
    parser.add_particle_weight(action_probs[pred]);
  }

  return parser;
}

template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::beamDiscriminativeParseSentence(
      const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights, unsigned beam_size) {
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), config_)); 

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 0; k <= 2*(sent.size()-1); ++k) {
    //prune if size exceeds beam_size
    if (beam_stack.size() > beam_size) {
      std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
      for (int j = beam_stack.size()- 1; ((j >= beam_size) || (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
    }
      
    //for every item in the beam, add some valid actions 
    unsigned stack_size = beam_stack.size();
    for (unsigned j = 0; (j < stack_size); ++j) {
      if (beam_stack[j]->num_particles() == 0)
        continue;
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      Real shiftp = action_probs[0];
      
      if ((beam_stack[j]->stack_depth() < 2) || 
             (config_->root_first && (beam_stack[j]->stack_depth() == 2) 
                && !beam_stack[j]->buffer_empty())) {
        //only shift is valid
        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp); 
      } else if (config_->direction_deterministic || !beam_stack[j]->left_arc_valid()) {  
        WordIndex reduce_pred = arg_min(action_probs, 1);
	    if (!beam_stack[j]->left_arc_valid()) 
          reduce_pred = arg_min(action_probs, config_->num_labels + 1);

        Real reducep = action_probs[reduce_pred];
        kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
        WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
      
        //add the best reduce action 
        beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
        if (re_act == kAction::la) 
          beam_stack.back()->leftArc(re_label);
	    else
          beam_stack.back()->rightArc(re_label);
        beam_stack.back()->add_particle_weight(reducep);

        //take shift action (if possible)
        if (!beam_stack[j]->buffer_empty() && 
                 (config_->root_first || (beam_stack[j]->buffer_next() != 0))) {
          beam_stack[j]->shift();
          beam_stack[j]->add_particle_weight(shiftp); 
        } else {
          beam_stack[j]->set_num_particles(0);
        }
      } else {
        //sort to find n-best actions
        std::vector<int> indices(action_probs.size());
        std::iota(indices.begin(), indices.end(), 0);
        bool has_shift = false;
               
        std::sort(indices.begin(), indices.end(), [&action_probs](const int i, const int j) 
                {return (action_probs[i] < action_probs[j]);});
        for (unsigned l = 0; (l < beam_size && l < config_->max_beam_increment); ++l) {
          WordIndex reduce_pred = indices[l];
          Real reducep = action_probs[reduce_pred];
          kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
          WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
          if (re_act == kAction::sh)
            has_shift = true;  
          else {
            //add reduce action
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (re_act == kAction::la) 
              beam_stack.back()->leftArc(re_label);
	        else
              beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(reducep);
          }
        }
        //take shift action if chosen and valid
        if (has_shift && !beam_stack[j]->buffer_empty()
               && (config_->root_first || (beam_stack[j]->buffer_next() != 0))) {
          beam_stack[j]->shift();
          beam_stack[j]->add_particle_weight(shiftp); 
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

  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
  
  //compute beam weight
  for (unsigned i = 0; (i < beam_stack.size()); ++i)
    if (!duplicate[i])
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight()); 

  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else
    return ArcStandardLabelledParser(*beam_stack[0]); 
}

template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::beamParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, unsigned beam_size) {
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), 
              config_)); 

  for (unsigned i = 0; ((config_->root_first && (i < sent.size())) 
                         || (!config_->root_first && (i < sent.size() - 1))); ++i) {
    //TODO test if worst_weight helps
    double worst_weight = beam_stack[beam_stack.size() - 1]->particle_weight();

    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      //reduce actions are direction deterministic
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      Real shiftp = action_probs[0];
      
      if ((beam_stack[j]->stack_depth() >= 2) && 
          ((j < beam_size) || (beam_stack[j]->particle_weight() < worst_weight))) { 
        //add best reduce action
        WordIndex reduce_pred = arg_min(action_probs, 1);
	    if (!beam_stack[j]->left_arc_valid()) 
          reduce_pred = arg_min(action_probs, config_->num_labels + 1);

        Real reducep = action_probs[reduce_pred];
        kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
        WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
      
        //don't add hypothesis guaranteed to be off the beam
        if ((j < beam_size) || (beam_stack[j]->particle_weight() + reducep < worst_weight)) {
          beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
          if (re_act == kAction::la) 
            beam_stack.back()->leftArc(re_label);
	      else
            beam_stack.back()->rightArc(re_label);
          beam_stack.back()->add_particle_weight(reducep);
        }
      } else {
        shiftp = 0;
      }

      if (beam_stack[j]->particle_weight() > worst_weight)
        worst_weight = beam_stack[j]->particle_weight();

      //shift
      Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
      Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

      beam_stack[j]->shift();
      beam_stack[j]->add_particle_weight(shiftp); 
      beam_stack[j]->add_particle_weight(wordp); 
      beam_stack[j]->add_particle_weight(tagp); 
      beam_stack[j]->add_importance_weight(wordp); 
      beam_stack[j]->add_importance_weight(tagp); 
    }

    //prune the beam
    if (beam_stack.size() > beam_size) {
      std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
      for (int j = beam_stack.size()- 1; (j >= beam_size); --j)
        beam_stack.pop_back();
    }
  }
    
  //completion: greedily reduce each item
  for (unsigned j = 0; (config_->complete_parse && (j < beam_stack.size())); ++j) { 
    while (beam_stack[j]->stack_depth() >= 2) {
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      WordIndex reduce_pred = arg_min(action_probs, 1);
      if (!beam_stack[j]->left_arc_valid()) 
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);

      Real reducep = action_probs[reduce_pred];
      kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
      
      if (re_act == kAction::la) 
        beam_stack[j]->leftArc(re_label);
	  else
        beam_stack[j]->rightArc(re_label);
      beam_stack[j]->add_particle_weight(reducep);
    }
    
    //for root-last: final shift and reduce
    if (!config_->root_first) {
      Real shiftp = weights->predictAction(0, beam_stack[j]->actionContext());
      Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
      Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

      beam_stack[j]->shift();
      beam_stack[j]->add_particle_weight(shiftp); 
      beam_stack[j]->add_particle_weight(wordp); 
      beam_stack[j]->add_particle_weight(tagp); 
      beam_stack[j]->add_importance_weight(wordp); 
      beam_stack[j]->add_importance_weight(tagp); 

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      WordId reduce_pred = arg_min(action_probs, 1, config_->num_labels + 1); //only left-arc valid
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
    
      beam_stack[j]->leftArc(re_label);
      beam_stack[j]->add_particle_weight(action_probs[reduce_pred]);
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

  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
 
  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else
    return ArcStandardLabelledParser(*beam_stack[0]); 
}

//modified to extract training examples during decoding
template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::beamParticleParseSentence(
        const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights, unsigned num_particles, const boost::shared_ptr<ParseDataSet>& examples) {
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles), config_)); 

  for (unsigned i = 0; ((i < sent.size() - 1)  
             || ((config_->root_first || !config_->complete_parse) && (i == sent.size() - 1))); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      int num_samples = beam_stack[j]->num_particles();
      //std::cout << i << ", " << j << " " << num_samples << std::endl;
      if (num_samples == 0)
        continue;
      else if (num_samples < 0) {
        beam_stack[j]->set_num_particles(-num_samples);
        continue;
      }
      
      //extract training example
      //kAction gold_a = beam_stack[j]->oracleNext(sent);
      //WordId gold_lab = beam_stack[j]->oracleNextLabel(sent);

//      if (gold_a == kAction::sh) {
//        examples->add_tag_example(DataPoint(beam_stack[j]->next_tag(), beam_stack[j]->tagContext()));  
//        examples->add_word_example(DataPoint(beam_stack[j]->next_word(), beam_stack[j]->wordContext()));  
//      } 

//      if (gold_a != kAction::re) 
//        examples->add_action_example(DataPoint(beam_stack[j]->convert_action(gold_a, gold_lab), beam_stack[j]->actionContext()));  

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int shift_count = 0;
      int reduce_count = 0;

      Real shiftp = action_probs[0];
      Real tot_reducep = log_one_min(shiftp);

      if ((beam_stack[j]->stack_depth() < 2) || 
          (config_->root_first && config_->complete_parse && (beam_stack[j]->stack_depth() == 2))) {
        shift_count = num_samples;
        shiftp = 0;
      } else { 
        shift_count = std::round(std::exp(-shiftp)*num_samples); 
        reduce_count = num_samples - shift_count; 

        if (config_->direction_deterministic) {
          WordIndex reduce_pred = arg_min(action_probs, 1, 2*config_->num_labels +1);
          if (config_->parser_type == ParserType::arcstandard2)
            reduce_pred = arg_min(action_probs, 1);

          if (!beam_stack[j]->left_arc2_valid() && (reduce_pred <= 3*config_->num_labels)) 
            reduce_pred = arg_min(action_probs, 1, 2*config_->num_labels + 1);
          if (!beam_stack[j]->left_arc_valid()) 
            reduce_pred = arg_min(action_probs, config_->num_labels + 1);
        
          if (reduce_count > 0) {
            kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
            WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (re_act == kAction::la) 
              beam_stack.back()->leftArc(re_label);
  	        else if (re_act == kAction::ra) {
              //std::cout << beam_stack.back()->stack_depth() << std::endl;
              beam_stack.back()->rightArc(re_label);
            } else if (re_act == kAction::la2) 
              beam_stack.back()->leftArc2(re_label);
  	        else if (re_act == kAction::ra2) 
              beam_stack.back()->rightArc2(re_label);
            //else
            //  std::cout << "bla" << std::endl;
            beam_stack.back()->add_particle_weight(action_probs[reduce_pred]);
            beam_stack.back()->set_num_particles(reduce_count); 
          }
        } else {
          if (config_->parser_type == ParserType::arcstandard2) {
            WordIndex left_reduce2_pred = arg_min(action_probs, 2*config_->num_labels + 1, 
                                                                3*config_->num_labels + 1);
            WordIndex right_reduce2_pred = arg_min(action_probs, 3*config_->num_labels + 1, 
                                                               4*config_->num_labels + 1);
            Real left_reduce2p = L_MAX;
            for (unsigned l = 2*config_->num_labels + 1; l < 3*config_->num_labels + 1; ++l) 
              left_reduce2p = neg_log_sum_exp(left_reduce2p, action_probs[l]);
            Real right_reduce2p = L_MAX;
            for (unsigned l = 3*config_->num_labels + 1; l < 4*config_->num_labels + 1; ++l) 
              right_reduce2p = neg_log_sum_exp(right_reduce2p, action_probs[l]);

            int left_reduce2_count = std::round(std::exp(-left_reduce2p)*num_samples); 
            if (!beam_stack[j]->left_arc2_valid()) 
              left_reduce2_count = 0;
            int right_reduce2_count = std::round(std::exp(-right_reduce2p)*num_samples); 
            reduce_count = reduce_count - left_reduce2_count - right_reduce2_count;
          
            if (left_reduce2_count > 0) {
              //std::cout << std::exp(-tot_reducep) << ";" << std::exp(-left_reduce2p) << "," << std::exp(-right_reduce2p) << " ";
              WordId re_label = beam_stack[j]->lookup_label(left_reduce2_pred);

              beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
              beam_stack.back()->leftArc2(re_label);
              beam_stack.back()->add_particle_weight(action_probs[left_reduce2_pred]);
              beam_stack.back()->set_num_particles(left_reduce2_count); 
            }

            if (right_reduce2_count > 0) {
              //std::cout << right_reduce2_count << " ";
              WordId re_label = beam_stack[j]->lookup_label(right_reduce2_pred);

              beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
              beam_stack.back()->rightArc2(re_label);
              beam_stack.back()->add_particle_weight(action_probs[right_reduce2_pred]);
              beam_stack.back()->set_num_particles(right_reduce2_count); 
            }
          }  

          WordIndex left_reduce_pred = arg_min(action_probs, 1, config_->num_labels + 1);
          WordIndex right_reduce_pred = arg_min(action_probs, config_->num_labels + 1, 
                                                              2*config_->num_labels + 1);
          Real left_reducep = L_MAX;
          for (unsigned l = 1; l < config_->num_labels + 1; ++l) 
            left_reducep = neg_log_sum_exp(left_reducep, action_probs[l]);
          Real right_reducep = L_MAX;
          for (unsigned l = config_->num_labels + 1; l < 2*config_->num_labels + 1; ++l) 
            right_reducep = neg_log_sum_exp(right_reducep, action_probs[l]);

          int left_reduce_count = std::round(std::exp(-left_reducep)*num_samples); 
          if (!beam_stack[j]->left_arc_valid()) 
            left_reduce_count = 0;
          int right_reduce_count = reduce_count - left_reduce_count;
          
          if (left_reduce_count > 0) {
            WordId re_label = beam_stack[j]->lookup_label(left_reduce_pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
              beam_stack.back()->leftArc(re_label);
            beam_stack.back()->add_particle_weight(action_probs[left_reduce_pred]);
            beam_stack.back()->set_num_particles(left_reduce_count); 
          }

          if (right_reduce_count > 0) {
            WordId re_label = beam_stack[j]->lookup_label(right_reduce_pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            //std::cout << beam_stack.back()->stack_depth() << std::endl;
            beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(action_probs[right_reduce_pred]);
            beam_stack.back()->set_num_particles(right_reduce_count); 
          }
        }
      } 

      if (shift_count == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        if (config_->predict_pos) {
          Reals tag_probs = weights->predictTag(beam_stack[j]->tagContext());
          Reals word_probs = weights->predictWordOverTags(beam_stack[j]->next_word(), beam_stack[j]->wordContext());
          for (unsigned k = 0; k < tag_probs.size(); ++k)
            tag_probs[k] += word_probs[k];

          //sort tag+word prob
          std::vector<int> indices(tag_probs.size());
          std::iota(indices.begin(), indices.end(), 0);
          std::sort(indices.begin(), indices.end(), [&tag_probs](const int i, const int j) 
                {return (tag_probs[i] < tag_probs[j]);});

          //most likely tag
          WordIndex tag_pred = indices[0]; // arg_min(tag_probs, 0);
          beam_stack[j]->set_tag_at(i, tag_pred);

          //add second most likely tag if prob is high enough 
          WordIndex tag2_pred = indices[1]; 
          WordIndex tag3_pred = indices[2]; 
          //int shift2_count = std::floor(std::exp(-tag_probs[tag2_pred] 
          //                   + neg_log_sum_exp(tag_probs[tag_pred], tag_probs[tag2_pred]))*shift_count);
          double pred_sum = neg_log_sum_exp(neg_log_sum_exp(tag_probs[tag_pred], tag_probs[tag2_pred]),
                                 tag_probs[tag3_pred]);

          int shift2_count = std::floor(std::exp(-tag_probs[tag2_pred] + pred_sum)*shift_count);
          int shift3_count = std::floor(std::exp(-tag_probs[tag3_pred] + pred_sum)*shift_count);
          
          if (shift2_count > 0) {
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->set_tag_at(i, tag2_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp); 
            beam_stack.back()->set_num_particles(-shift2_count); //hack to communicate it should be skipped
        
            beam_stack.back()->add_particle_weight(tag_probs[tag2_pred]); 
            beam_stack.back()->add_importance_weight(tag_probs[tag2_pred]); 
            shift_count -= shift2_count;    
          } 

          //add third most likely tag if prob is high enough 
         if (shift3_count > 0) {
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->set_tag_at(i, tag3_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp); 
            beam_stack.back()->set_num_particles(-shift3_count); //hack to communicate it should be skipped
        
            beam_stack.back()->add_particle_weight(tag_probs[tag3_pred]); 
            beam_stack.back()->add_importance_weight(tag_probs[tag3_pred]); 
            shift_count -= shift3_count;    
          } 
        } 

        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

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

  //std::cout << "completion" << std::endl;
  //completion: greedily reduce each item
  for (unsigned j = 0; (j < beam_stack.size()); ++j) { 
    while ((beam_stack[j]->num_particles() > 0) && (beam_stack[j]->stack_depth() >= 2)) {  //&& config_->complete_parse
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();
      //std::cout << j << " " << beam_stack[j]->stack_depth() << std::endl;
      
      //extract training example
      //kAction gold_a = beam_stack[j]->oracleNext(sent);
      //WordId gold_lab = beam_stack[j]->oracleNextLabel(sent);
      //if (gold_a == kAction::la || gold_a == kAction::ra) 
      //  examples->add_action_example(DataPoint(beam_stack[j]->convert_action(gold_a, gold_lab), beam_stack[j]->actionContext()));  
      
      WordIndex reduce_pred = arg_min(action_probs, 1, 2*config_->num_labels +1);
      if (config_->parser_type == ParserType::arcstandard2)
        reduce_pred = arg_min(action_probs, 1);

      if (!beam_stack[j]->left_arc2_valid() && (reduce_pred <= 3*config_->num_labels)) 
        reduce_pred = arg_min(action_probs, 1, 2*config_->num_labels + 1);
      if (!beam_stack[j]->left_arc_valid()) 
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);

      Real reducep = action_probs[reduce_pred];
      kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

      if (re_act == kAction::la) 
        beam_stack[j]->leftArc(re_label);
  	  else if (re_act == kAction::ra) {
        //std::cout << beam_stack.back()->stack_depth() << std::endl;
        beam_stack[j]->rightArc(re_label);
      } else if (re_act == kAction::la2) 
        beam_stack[j]->leftArc2(re_label);
  	  else if (re_act == kAction::ra2) 
        beam_stack[j]->rightArc2(re_label);
      //else
      //  std::cout << "bla" << std::endl;

      beam_stack[j]->add_particle_weight(reducep);
      beam_stack[j]->set_num_particles(num_samples);
    } 

    //for root-last: final shift and reduce
    if (config_->complete_parse && !config_->root_first) {
      Real shiftp = weights->predictAction(0, beam_stack[j]->actionContext());
      Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
      Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

      beam_stack[j]->shift();
      beam_stack[j]->add_particle_weight(shiftp); 
      beam_stack[j]->add_particle_weight(wordp); 
      beam_stack[j]->add_particle_weight(tagp); 
      beam_stack[j]->add_importance_weight(wordp); 
      beam_stack[j]->add_importance_weight(tagp); 

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      WordId reduce_pred = arg_min(action_probs, 1, config_->num_labels + 1); //only left-arc valid
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
    
      beam_stack[j]->leftArc(re_label);
      beam_stack[j]->add_particle_weight(action_probs[reduce_pred]);
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

  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 

  //compute beam weight
  for (unsigned i = 0; (i < beam_stack.size()); ++i)
    if (!duplicate[i] && (beam_stack[i]->num_particles() > 0)) 
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight()); 

  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else
    return ArcStandardLabelledParser(*beam_stack[0]); 
} 

template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::particleParseSentence(
        const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights, 
        MT19937& eng, unsigned num_particles, const boost::shared_ptr<ParseDataSet>& examples) {
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles), config_)); 

  for (unsigned i = 0; ((i < sent.size() - 1)  
             || ((config_->root_first || !config_->complete_parse) && (i == sent.size() - 1))); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      int num_samples = beam_stack[j]->num_particles();
      //std::cout << i << ", " << j << " " << num_samples << std::endl;
      if (num_samples == 0)
        continue;
      else if (num_samples < 0) {
        beam_stack[j]->set_num_particles(-num_samples);
        continue;
      }
      
      //extract training example
      //kAction gold_a = beam_stack[j]->oracleNext(sent);
      //WordId gold_lab = beam_stack[j]->oracleNextLabel(sent);

//      if (gold_a == kAction::sh) {
//        examples->add_tag_example(DataPoint(beam_stack[j]->next_tag(), beam_stack[j]->tagContext()));  
//        examples->add_word_example(DataPoint(beam_stack[j]->next_word(), beam_stack[j]->wordContext()));  
//      } 

//      if (gold_a != kAction::re) 
//        examples->add_action_example(DataPoint(beam_stack[j]->convert_action(gold_a, gold_lab), beam_stack[j]->actionContext()));  

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int shift_count = 0;
      int reduce_count = 0;

      Real shiftp = action_probs[0];
      Real tot_reducep = log_one_min(shiftp);

      if ((beam_stack[j]->stack_depth() < 2) || 
          (config_->root_first && config_->complete_parse && (beam_stack[j]->stack_depth() == 2))) {
        shift_count = num_samples;
        shiftp = 0;
      } else { 
        shift_count = std::round(std::exp(-shiftp)*num_samples); 
        reduce_count = num_samples - shift_count; 

        if (config_->direction_deterministic) {
          WordIndex reduce_pred = arg_min(action_probs, 1, 2*config_->num_labels +1);
          if (config_->parser_type == ParserType::arcstandard2)
            reduce_pred = arg_min(action_probs, 1);

          if (!beam_stack[j]->left_arc2_valid() && (reduce_pred <= 3*config_->num_labels)) 
            reduce_pred = arg_min(action_probs, 1, 2*config_->num_labels + 1);
          if (!beam_stack[j]->left_arc_valid()) 
            reduce_pred = arg_min(action_probs, config_->num_labels + 1);
        
          if (reduce_count > 0) {
            kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
            WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (re_act == kAction::la) 
              beam_stack.back()->leftArc(re_label);
  	        else if (re_act == kAction::ra) {
              //std::cout << beam_stack.back()->stack_depth() << std::endl;
              beam_stack.back()->rightArc(re_label);
            } else if (re_act == kAction::la2) 
              beam_stack.back()->leftArc2(re_label);
  	        else if (re_act == kAction::ra2) 
              beam_stack.back()->rightArc2(re_label);
            //else
            //  std::cout << "bla" << std::endl;
            beam_stack.back()->add_particle_weight(action_probs[reduce_pred]);
            beam_stack.back()->set_num_particles(reduce_count); 
          }
        } else {
          if (config_->parser_type == ParserType::arcstandard2) {
            WordIndex left_reduce2_pred = arg_min(action_probs, 2*config_->num_labels + 1, 
                                                                3*config_->num_labels + 1);
            WordIndex right_reduce2_pred = arg_min(action_probs, 3*config_->num_labels + 1, 
                                                               4*config_->num_labels + 1);
            Real left_reduce2p = L_MAX;
            for (unsigned l = 2*config_->num_labels + 1; l < 3*config_->num_labels + 1; ++l) 
              left_reduce2p = neg_log_sum_exp(left_reduce2p, action_probs[l]);
            Real right_reduce2p = L_MAX;
            for (unsigned l = 3*config_->num_labels + 1; l < 4*config_->num_labels + 1; ++l) 
              right_reduce2p = neg_log_sum_exp(right_reduce2p, action_probs[l]);

            int left_reduce2_count = std::round(std::exp(-left_reduce2p)*num_samples); 
            if (!beam_stack[j]->left_arc2_valid()) 
              left_reduce2_count = 0;
            int right_reduce2_count = std::round(std::exp(-right_reduce2p)*num_samples); 
            reduce_count = reduce_count - left_reduce2_count - right_reduce2_count;
          
            if (left_reduce2_count > 0) {
              //std::cout << std::exp(-tot_reducep) << ";" << std::exp(-left_reduce2p) << "," << std::exp(-right_reduce2p) << " ";
              WordId re_label = beam_stack[j]->lookup_label(left_reduce2_pred);

              beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
              beam_stack.back()->leftArc2(re_label);
              beam_stack.back()->add_particle_weight(action_probs[left_reduce2_pred]);
              beam_stack.back()->set_num_particles(left_reduce2_count); 
            }

            if (right_reduce2_count > 0) {
              //std::cout << right_reduce2_count << " ";
              WordId re_label = beam_stack[j]->lookup_label(right_reduce2_pred);

              beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
              beam_stack.back()->rightArc2(re_label);
              beam_stack.back()->add_particle_weight(action_probs[right_reduce2_pred]);
              beam_stack.back()->set_num_particles(right_reduce2_count); 
            }
          }  

          WordIndex left_reduce_pred = arg_min(action_probs, 1, config_->num_labels + 1);
          WordIndex right_reduce_pred = arg_min(action_probs, config_->num_labels + 1, 
                                                              2*config_->num_labels + 1);
          Real left_reducep = L_MAX;
          for (unsigned l = 1; l < config_->num_labels + 1; ++l) 
            left_reducep = neg_log_sum_exp(left_reducep, action_probs[l]);
          Real right_reducep = L_MAX;
          for (unsigned l = config_->num_labels + 1; l < 2*config_->num_labels + 1; ++l) 
            right_reducep = neg_log_sum_exp(right_reducep, action_probs[l]);

          int left_reduce_count = std::round(std::exp(-left_reducep)*num_samples); 
          if (!beam_stack[j]->left_arc_valid()) 
            left_reduce_count = 0;
          int right_reduce_count = reduce_count - left_reduce_count;
          
          if (left_reduce_count > 0) {
            WordId re_label = beam_stack[j]->lookup_label(left_reduce_pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
              beam_stack.back()->leftArc(re_label);
            beam_stack.back()->add_particle_weight(action_probs[left_reduce_pred]);
            beam_stack.back()->set_num_particles(left_reduce_count); 
          }

          if (right_reduce_count > 0) {
            WordId re_label = beam_stack[j]->lookup_label(right_reduce_pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            //std::cout << beam_stack.back()->stack_depth() << std::endl;
            beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(action_probs[right_reduce_pred]);
            beam_stack.back()->set_num_particles(right_reduce_count); 
          }
        }
      } 

      if (shift_count == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        if (config_->predict_pos) {
          Reals tag_probs = weights->predictTag(beam_stack[j]->tagContext());
          Reals word_probs = weights->predictWordOverTags(beam_stack[j]->next_word(), beam_stack[j]->wordContext());
          for (unsigned k = 0; k < tag_probs.size(); ++k)
            tag_probs[k] += word_probs[k];

          //sort tag+word prob
          std::vector<int> indices(tag_probs.size());
          std::iota(indices.begin(), indices.end(), 0);
          std::sort(indices.begin(), indices.end(), [&tag_probs](const int i, const int j) 
                {return (tag_probs[i] < tag_probs[j]);});

          //most likely tag
          WordIndex tag_pred = indices[0]; // arg_min(tag_probs, 0);
          beam_stack[j]->set_tag_at(i, tag_pred);

          //add second most likely tag if prob is high enough 
          WordIndex tag2_pred = indices[1]; 
          WordIndex tag3_pred = indices[2]; 
          //int shift2_count = std::floor(std::exp(-tag_probs[tag2_pred] 
          //                   + neg_log_sum_exp(tag_probs[tag_pred], tag_probs[tag2_pred]))*shift_count);
          double pred_sum = neg_log_sum_exp(neg_log_sum_exp(tag_probs[tag_pred], tag_probs[tag2_pred]),
                                 tag_probs[tag3_pred]);

          int shift2_count = std::floor(std::exp(-tag_probs[tag2_pred] + pred_sum)*shift_count);
          int shift3_count = std::floor(std::exp(-tag_probs[tag3_pred] + pred_sum)*shift_count);
          
          if (shift2_count > 0) {
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->set_tag_at(i, tag2_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp); 
            beam_stack.back()->set_num_particles(-shift2_count); //hack to communicate it should be skipped
        
            beam_stack.back()->add_particle_weight(tag_probs[tag2_pred]); 
            beam_stack.back()->add_importance_weight(tag_probs[tag2_pred]); 
            shift_count -= shift2_count;    
          } 

          //add third most likely tag if prob is high enough 
         if (shift3_count > 0) {
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->set_tag_at(i, tag3_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp); 
            beam_stack.back()->set_num_particles(-shift3_count); //hack to communicate it should be skipped
        
            beam_stack.back()->add_particle_weight(tag_probs[tag3_pred]); 
            beam_stack.back()->add_importance_weight(tag_probs[tag3_pred]); 
            shift_count -= shift3_count;    
          } 
        } 

        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

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

  //std::cout << "completion" << std::endl;
  //completion: greedily reduce each item
  for (unsigned j = 0; (j < beam_stack.size()); ++j) { 
    while ((beam_stack[j]->num_particles() > 0) && (beam_stack[j]->stack_depth() >= 2)) {  //&& config_->complete_parse
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();
      //std::cout << j << " " << beam_stack[j]->stack_depth() << std::endl;
      
      //extract training example
      //kAction gold_a = beam_stack[j]->oracleNext(sent);
      //WordId gold_lab = beam_stack[j]->oracleNextLabel(sent);
      //if (gold_a == kAction::la || gold_a == kAction::ra) 
      //  examples->add_action_example(DataPoint(beam_stack[j]->convert_action(gold_a, gold_lab), beam_stack[j]->actionContext()));  
      
      WordIndex reduce_pred = arg_min(action_probs, 1, 2*config_->num_labels +1);
      if (config_->parser_type == ParserType::arcstandard2)
        reduce_pred = arg_min(action_probs, 1);

      if (!beam_stack[j]->left_arc2_valid() && (reduce_pred <= 3*config_->num_labels)) 
        reduce_pred = arg_min(action_probs, 1, 2*config_->num_labels + 1);
      if (!beam_stack[j]->left_arc_valid()) 
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);

      Real reducep = action_probs[reduce_pred];
      kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

      if (re_act == kAction::la) 
        beam_stack[j]->leftArc(re_label);
  	  else if (re_act == kAction::ra) {
        //std::cout << beam_stack.back()->stack_depth() << std::endl;
        beam_stack[j]->rightArc(re_label);
      } else if (re_act == kAction::la2) 
        beam_stack[j]->leftArc2(re_label);
  	  else if (re_act == kAction::ra2) 
        beam_stack[j]->rightArc2(re_label);
      //else
      //  std::cout << "bla" << std::endl;

      beam_stack[j]->add_particle_weight(reducep);
      beam_stack[j]->set_num_particles(num_samples);
    } 

    //for root-last: final shift and reduce
    if (config_->complete_parse && !config_->root_first) {
      Real shiftp = weights->predictAction(0, beam_stack[j]->actionContext());
      Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
      Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

      beam_stack[j]->shift();
      beam_stack[j]->add_particle_weight(shiftp); 
      beam_stack[j]->add_particle_weight(wordp); 
      beam_stack[j]->add_particle_weight(tagp); 
      beam_stack[j]->add_importance_weight(wordp); 
      beam_stack[j]->add_importance_weight(tagp); 

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      WordId reduce_pred = arg_min(action_probs, 1, config_->num_labels + 1); //only left-arc valid
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
    
      beam_stack[j]->leftArc(re_label);
      beam_stack[j]->add_particle_weight(action_probs[reduce_pred]);
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

  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 

  //compute beam weight
  for (unsigned i = 0; (i < beam_stack.size()); ++i)
    if (!duplicate[i] && (beam_stack[i]->num_particles() > 0)) 
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight()); 

  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else
    return ArcStandardLabelledParser(*beam_stack[0]); 
} 



//find highest-scoring parse consistent with gold-parse
template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::beamParticleGoldParseSentence(
        const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights, unsigned num_particles) {
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles), config_)); 

  for (unsigned i = 0; (i < sent.size()); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      int num_samples = beam_stack[j]->num_particles();
      if (num_samples == 0)
        continue;
      else if (num_samples < 0) {
        beam_stack[j]->set_num_particles(-num_samples);
        continue;
      }

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
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
          WordIndex reduce_pred = beam_stack[j]->convert_action(gold_act, gold_label);
          shift_count = std::round(std::exp(-shiftp)*num_samples); 
          if (gold_act == kAction::ra)
            shift_count = 0;
          else {
            //shift invalid if s0 has all its right children
            int st = beam_stack[j]->stack_top();
            bool has_right_children = true;
            for (WordIndex l = st + 1; l < beam_stack[j]->size(); ++l) {
              if (sent.has_arc(l, st) && !beam_stack[j]->has_arc(l, st)) {
                has_right_children = false;
                break;
              }
            }
            if (has_right_children)
              shift_count = 0;
          } 

          reduce_count = num_samples - shift_count; 
          
          if (reduce_count > 0) {
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (gold_act == kAction::la) 
              beam_stack.back()->leftArc(gold_label);
	        else
              beam_stack.back()->rightArc(gold_label);
            beam_stack.back()->add_particle_weight(action_probs[reduce_pred]);
            beam_stack.back()->set_num_particles(reduce_count); 
          }
        } else {
          shift_count = num_samples;
        }
         
      }

      if (shift_count == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        if (config_->predict_pos) {
          Reals tag_probs = weights->predictTag(beam_stack[j]->tagContext());
          Reals word_probs = weights->predictWordOverTags(beam_stack[j]->next_word(), beam_stack[j]->wordContext());
          for (unsigned k = 0; k < tag_probs.size(); ++k)
            tag_probs[k] += word_probs[k];

          //sort tag+word prob
          std::vector<int> indices(tag_probs.size());
          std::iota(indices.begin(), indices.end(), 0);
          std::sort(indices.begin(), indices.end(), [&tag_probs](const int i, const int j) 
                {return (tag_probs[i] < tag_probs[j]);});

          //most likely tag
          WordIndex tag_pred = indices[0]; // arg_min(tag_probs, 0);
          beam_stack[j]->set_tag_at(i, tag_pred);

          //add second most likely tag if prob is high enough 
          WordIndex tag2_pred = indices[1]; 
          WordIndex tag3_pred = indices[2]; 
          //int shift2_count = std::floor(std::exp(-tag_probs[tag2_pred] 
          //                   + neg_log_sum_exp(tag_probs[tag_pred], tag_probs[tag2_pred]))*shift_count);
          double pred_sum = neg_log_sum_exp(neg_log_sum_exp(tag_probs[tag_pred], tag_probs[tag2_pred]),
                                 tag_probs[tag3_pred]);

          int shift2_count = std::floor(std::exp(-tag_probs[tag2_pred] + pred_sum)*shift_count);
          int shift3_count = std::floor(std::exp(-tag_probs[tag3_pred] + pred_sum)*shift_count);
          
          if (shift2_count > 0) {
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->set_tag_at(i, tag2_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp); 
            beam_stack.back()->set_num_particles(-shift2_count); //hack to communicate it should be skipped
        
            beam_stack.back()->add_particle_weight(tag_probs[tag2_pred]); 
            beam_stack.back()->add_importance_weight(tag_probs[tag2_pred]); 
            shift_count -= shift2_count;    
          } 

          //add third most likely tag if prob is high enough 
         if (shift3_count > 0) {
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->set_tag_at(i, tag3_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp); 
            beam_stack.back()->set_num_particles(-shift3_count); //hack to communicate it should be skipped
        
            beam_stack.back()->add_particle_weight(tag_probs[tag3_pred]); 
            beam_stack.back()->add_importance_weight(tag_probs[tag3_pred]); 
            shift_count -= shift3_count;    
          } 
        } 

        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

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

  //completion
  for (unsigned j = 0; (j < beam_stack.size()); ++j) { 
    while ((beam_stack[j]->num_particles() > 0) && (beam_stack[j]->stack_depth() >= 2)) {  
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();

      kAction gold_act = beam_stack[j]->oracleNext(sent);
      WordId gold_label = beam_stack[j]->oracleNextLabel(sent);

      if (gold_act == kAction::la || gold_act == kAction::ra) {
        WordIndex reduce_pred = beam_stack[j]->convert_action(gold_act, gold_label);
        //if (!beam_stack[j]->left_arc_valid()) 
        //  reduce_pred = arg_min(action_probs, config_->num_labels + 1);
        Real reducep = action_probs[reduce_pred];
      
        if (gold_act == kAction::la) 
          beam_stack[j]->leftArc(gold_label);
        else
          beam_stack[j]->rightArc(gold_label);
        beam_stack[j]->add_particle_weight(reducep);
        beam_stack[j]->set_num_particles(num_samples);
      } else {
        //beam_stack[j]->set_num_particles(0);
        break; //assume gold does not want a complete parse
      }
    } 
  }

  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 

  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else
    return ArcStandardLabelledParser(*beam_stack[0]); 
}

//sample parse consistent with gold-parse
template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::particleGoldParseSentence(
        const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles) {
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles), config_)); 

  for (unsigned i = 0; (i < sent.size()); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      int num_samples = beam_stack[j]->num_particles();
      if (num_samples == 0)
        continue;
      else if (num_samples < 0) {
        beam_stack[j]->set_num_particles(-num_samples);
        continue;
      }

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
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
          WordIndex reduce_pred = beam_stack[j]->convert_action(gold_act, gold_label);
          shift_count = std::round(std::exp(-shiftp)*num_samples); 
          if (gold_act == kAction::ra)
            shift_count = 0;
          else {
            //shift invalid if s0 has all its right children
            int st = beam_stack[j]->stack_top();
            bool has_right_children = true;
            for (WordIndex l = st + 1; l < beam_stack[j]->size(); ++l) {
              if (sent.has_arc(l, st) && !beam_stack[j]->has_arc(l, st)) {
                has_right_children = false;
                break;
              }
            }
            if (has_right_children)
              shift_count = 0;
          } 

          reduce_count = num_samples - shift_count; 
          
          if (reduce_count > 0) {
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (gold_act == kAction::la) 
              beam_stack.back()->leftArc(gold_label);
	        else
              beam_stack.back()->rightArc(gold_label);
            beam_stack.back()->add_particle_weight(action_probs[reduce_pred]);
            beam_stack.back()->set_num_particles(reduce_count); 
          }
        } else {
          shift_count = num_samples;
        }
         
      }

      if (shift_count == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        if (config_->predict_pos) {
          Reals tag_probs = weights->predictTag(beam_stack[j]->tagContext());
          Reals word_probs = weights->predictWordOverTags(beam_stack[j]->next_word(), beam_stack[j]->wordContext());
          for (unsigned k = 0; k < tag_probs.size(); ++k)
            tag_probs[k] += word_probs[k];

          //sort tag+word prob
          std::vector<int> indices(tag_probs.size());
          std::iota(indices.begin(), indices.end(), 0);
          std::sort(indices.begin(), indices.end(), [&tag_probs](const int i, const int j) 
                {return (tag_probs[i] < tag_probs[j]);});

          //most likely tag
          WordIndex tag_pred = indices[0]; // arg_min(tag_probs, 0);
          beam_stack[j]->set_tag_at(i, tag_pred);

          //add second most likely tag if prob is high enough 
          WordIndex tag2_pred = indices[1]; 
          WordIndex tag3_pred = indices[2]; 
          //int shift2_count = std::floor(std::exp(-tag_probs[tag2_pred] 
          //                   + neg_log_sum_exp(tag_probs[tag_pred], tag_probs[tag2_pred]))*shift_count);
          double pred_sum = neg_log_sum_exp(neg_log_sum_exp(tag_probs[tag_pred], tag_probs[tag2_pred]),
                                 tag_probs[tag3_pred]);

          int shift2_count = std::floor(std::exp(-tag_probs[tag2_pred] + pred_sum)*shift_count);
          int shift3_count = std::floor(std::exp(-tag_probs[tag3_pred] + pred_sum)*shift_count);
          
          if (shift2_count > 0) {
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->set_tag_at(i, tag2_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp); 
            beam_stack.back()->set_num_particles(-shift2_count); //hack to communicate it should be skipped
        
            beam_stack.back()->add_particle_weight(tag_probs[tag2_pred]); 
            beam_stack.back()->add_importance_weight(tag_probs[tag2_pred]); 
            shift_count -= shift2_count;    
          } 

          //add third most likely tag if prob is high enough 
         if (shift3_count > 0) {
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            beam_stack.back()->set_tag_at(i, tag3_pred);
            beam_stack.back()->shift();
            beam_stack.back()->add_particle_weight(shiftp); 
            beam_stack.back()->set_num_particles(-shift3_count); //hack to communicate it should be skipped
        
            beam_stack.back()->add_particle_weight(tag_probs[tag3_pred]); 
            beam_stack.back()->add_importance_weight(tag_probs[tag3_pred]); 
            shift_count -= shift3_count;    
          } 
        } 

        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

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

  //completion
  for (unsigned j = 0; (j < beam_stack.size()); ++j) { 
    while ((beam_stack[j]->num_particles() > 0) && (beam_stack[j]->stack_depth() >= 2)) {  
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();

      kAction gold_act = beam_stack[j]->oracleNext(sent);
      WordId gold_label = beam_stack[j]->oracleNextLabel(sent);

      if (gold_act == kAction::la || gold_act == kAction::ra) {
        WordIndex reduce_pred = beam_stack[j]->convert_action(gold_act, gold_label);
        //if (!beam_stack[j]->left_arc_valid()) 
        //  reduce_pred = arg_min(action_probs, config_->num_labels + 1);
        Real reducep = action_probs[reduce_pred];
      
        if (gold_act == kAction::la) 
          beam_stack[j]->leftArc(gold_label);
        else
          beam_stack[j]->rightArc(gold_label);
        beam_stack[j]->add_particle_weight(reducep);
        beam_stack[j]->set_num_particles(num_samples);
      } else {
        //beam_stack[j]->set_num_particles(0);
        break; //assume gold does not want a complete parse
      }
    } 
  }

  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 

  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else
    return ArcStandardLabelledParser(*beam_stack[0]); 
}



/*
template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::beamParticleParseSentence(
        const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights, unsigned num_particles) {
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles), config_)); 

  //slightly cheating for root-list incomplete parse, but that's just for comparison
  for (unsigned i = 0; (((config_->root_first || !config_->complete_parse) && (i < sent.size())) 
                         || (i < sent.size() - 1)); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      int num_samples = beam_stack[j]->num_particles();
      if (num_samples == 0)
        continue;

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int shift_count = 0;
      int reduce_count = 0;

      Real shiftp = action_probs[0];
      Real tot_reducep = log_one_min(shiftp);

      if (beam_stack[j]->stack_depth() < 2) {
        shift_count = num_samples;
        shiftp = 0;
      } else { 
        WordIndex reduce_pred = arg_min(action_probs, 1);
        if (!beam_stack[j]->left_arc_valid()) 
          reduce_pred = arg_min(action_probs, config_->num_labels + 1);

        shift_count = std::round(std::exp(-shiftp)*num_samples); 
        reduce_count = num_samples - shift_count; 
        
        if (reduce_count > 0) {
          kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
          WordId re_label = beam_stack[j]->lookup_label(reduce_pred);

          beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
          if (re_act == kAction::la) 
            beam_stack.back()->leftArc(re_label);
	      else
            beam_stack.back()->rightArc(re_label);
          beam_stack.back()->add_particle_weight(action_probs[reduce_pred]);
          beam_stack.back()->set_num_particles(reduce_count); 
        }
      } 

      if (shift_count == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

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

  //completion: greedily reduce each item
  for (unsigned j = 0; (config_->complete_parse && (j < beam_stack.size())); ++j) { 
    while ((beam_stack[j]->num_particles() > 0) && (beam_stack[j]->stack_depth() >= 2)) {
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();

      WordIndex reduce_pred = arg_min(action_probs, 1);
      if (!beam_stack[j]->left_arc_valid()) 
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);

      Real reducep = action_probs[reduce_pred];
      kAction act = beam_stack[j]->lookup_action(reduce_pred);
      WordId label = beam_stack[j]->lookup_label(reduce_pred);
      
      if (act == kAction::la) 
        beam_stack[j]->leftArc(label);
      else
        beam_stack[j]->rightArc(label);
      beam_stack[j]->add_particle_weight(reducep);
      beam_stack[j]->set_num_particles(num_samples);
    } 

    //for root-last: final shift and reduce
    if (!config_->root_first) {
      Real shiftp = weights->predictAction(0, beam_stack[j]->actionContext());
      Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
      Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

      beam_stack[j]->shift();
      beam_stack[j]->add_particle_weight(shiftp); 
      beam_stack[j]->add_particle_weight(wordp); 
      beam_stack[j]->add_particle_weight(tagp); 
      beam_stack[j]->add_importance_weight(wordp); 
      beam_stack[j]->add_importance_weight(tagp); 

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      WordId reduce_pred = arg_min(action_probs, 1, config_->num_labels + 1); //only left-arc valid
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
    
      beam_stack[j]->leftArc(re_label);
      beam_stack[j]->add_particle_weight(action_probs[reduce_pred]);
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

  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 

  //compute beam weight
  for (unsigned i = 0; (i < beam_stack.size()); ++i)
    if (!duplicate[i] && (beam_stack[i]->num_particles() > 0)) 
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight()); 

  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_);  
  } else
    return ArcStandardLabelledParser(*beam_stack[0]); 
} */

template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::staticGoldParseSentence(
        const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights) {
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
        Real wordp = weights->predictWord(parser.next_word(), parser.wordContext());
        parser.add_particle_weight(tagp);
        parser.add_particle_weight(wordp);
      }

      parser.executeAction(a, lab);
    } 
  }

  return parser;
}
    
template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::staticGoldParseSentence(const ParsedSentence& sent) {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(sent), config_);
  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && (a != kAction::re)) {
    a = parser.oracleNext(sent);  
    WordId lab = parser.oracleNextLabel(sent);
    //if (a == kAction::la2 || a == kAction::ra2)
    //  std::cout << lab << " ";
    if (a != kAction::re) 
      parser.executeAction(a, lab);
    //else {
    //  sent.print_arcs();
    //  parser.print_arcs();
    //}
  }

  return parser;
}

template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::generateSentence(const boost::shared_ptr<ParsedWeights>& weights, 
        MT19937& eng) {
  unsigned sent_limit = 100;
  ArcStandardLabelledParser parser(config_);
  bool terminate_shift = false;
  //if (config_->root_first) {
  parser.push_tag(1);
  parser.shift(1);
  //} 
  /* else {
    //sample a tag
    Reals tag_distr = weights->predictTag(parser.tagContext());
    tag_distr[0] = L_MAX;  
    tag_distr[1] = L_MAX;  

    multinomial_distribution_log<Real> t_mult(tag_distr);
    WordId tag = t_mult(eng);
    Real tagp = tag_distr[tag];
    parser.push_tag(tag);
    parser.add_particle_weight(tagp);

    //sample a word 
    Reals word_distr = weights->predictWord(parser.wordContext());
    word_distr[0] = L_MAX;  
    word_distr[1] = L_MAX;  

    multinomial_distribution_log<Real> w_mult(word_distr);
    WordId word = w_mult(eng);

    Real wordp = word_distr[word];
    parser.shift(word);
    parser.add_particle_weight(wordp);
  } */
   
  do {
    WordId pred = 0; //placeholder action (shift)
    
    if (parser.stack_depth() >= 2) {
      if (parser.size() == sent_limit) {
        std::cout << " SENTENCE LIMITED ";
        terminate_shift = true;
      }

      Reals action_probs = weights->predictAction(parser.actionContext());
  
      if (terminate_shift)
         action_probs[0] = L_MAX;
      if (!parser.left_arc_valid()) {
        for (int k = 0; k < config_->num_labels; ++k)
          action_probs[k+1] = L_MAX;
      }

      //sample an action
      multinomial_distribution_log<Real> mult(action_probs); 
      pred = mult(eng);
      parser.add_particle_weight(action_probs[pred]);
      //for (auto p: action_probs)
      //  std::cout << std::exp(-p) << " ";
      //std::cout << ": " << pred << std::endl;
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
      //sample a tag
      Reals tag_distr = weights->predictTag(parser.tagContext());
      tag_distr[0] = L_MAX;  
      if (config_->root_first || (parser.size() < 2))
        tag_distr[1] = L_MAX;  

      multinomial_distribution_log<Real> t_mult(tag_distr);
      WordId tag = t_mult(eng);
      Real tagp = tag_distr[tag];
      parser.push_tag(tag);
      parser.add_particle_weight(tagp);

      //sample a word 
      Reals word_distr = weights->predictWord(parser.wordContext());
      word_distr[0] = L_MAX;  
      if (config_->root_first || (parser.size() <= 2))
        word_distr[1] = L_MAX;  

      multinomial_distribution_log<Real> w_mult(word_distr);
      WordId word = w_mult(eng);

      Real wordp = word_distr[word];
      parser.shift(word);
      parser.add_particle_weight(wordp);
      if (!config_->root_first && (word == 1))
        terminate_shift = true;
    }
    
  } while (!parser.inTerminalConfiguration()); 
  
  std::cout << std::endl;
  return parser;
} 

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardLabelledParser parse = staticGoldParseSentence(sent); 
  parse.extractExamples(examples);
  //parse.print_actions();
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  //beamParticleParseSentence(sent, weights, config_->num_particles, examples);

   
  /*ArcStandardLabelledParser parse = staticGoldParseSentence(sent, weights);
  parse.extractExamples(examples);
  ArcStandardLabelledParser beam_parse = beamParticleParseSentence(sent, weights, config_->num_particles);
  beam_parse.extractExamples(examples, sent); */

  ArcStandardLabelledParser beam_parse = beamParticleGoldParseSentence(sent, weights, config_->num_particles);
  beam_parse.extractExamples(examples); 
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng,
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardLabelledParser parse = particleGoldParseSentence(sent, weights, eng, config_->num_particles);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng,
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardLabelledParser parse = particleParseSentence(sent, weights, eng, config_->num_particles, examples);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardLabelledParser parse = beamParticleParseSentence(sent, weights, config_->num_particles, examples);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
Parser ArcStandardLabelledParseModel<ParsedWeights>::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) {
  ArcStandardLabelledParser parse(config_);
  boost::shared_ptr<ParseDataSet> examples = boost::make_shared<ParseDataSet>();
  if (beam_size == 0)
    parse = greedyParseSentence(sent, weights);
  else if (config_->discriminative)
    parse = beamDiscriminativeParseSentence(sent, weights, beam_size);
  else
    parse = beamParticleParseSentence(sent, weights, beam_size, examples);
    //parse = beamParseSentence(sent, weights, beam_size);

  acc_counts->countAccuracy(parse, sent);
  ArcStandardLabelledParser gold_parse = staticGoldParseSentence(sent, weights);

  acc_counts->countGoldLikelihood(parse.weight(), gold_parse.weight());
  return parse;
}

//don't use unless neccessary
template<class ParsedWeights>
Parser ArcStandardLabelledParseModel<ParsedWeights>::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) {
  ArcStandardLabelledParser parse(config_);
  boost::shared_ptr<ParseDataSet> examples = boost::make_shared<ParseDataSet>();
  if (beam_size == 0)
    parse = greedyParseSentence(sent, weights);
  else if (config_->discriminative)
    parse = beamDiscriminativeParseSentence(sent, weights, beam_size);
  else 
    parse = beamParticleParseSentence(sent, weights, beam_size, examples); 
  acc_counts->countAccuracy(parse, sent);
  ArcStandardLabelledParser gold_parse = staticGoldParseSentence(sent, weights);
  
  acc_counts->countGoldLikelihood(parse.weight(), gold_parse.weight());
  return parse;
}

/*
 
//Quadratic beam search no longer used
template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::beamParseSentence(const ParsedSentence& sent, 
                               const boost::shared_ptr<ParsedWeights>& weights, unsigned beam_size) {
  //bool direction_deterministic = false;
    
  std::vector<AslParserList> beam_chart; 
  beam_chart.push_back(AslParserList());
  beam_chart[0].push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), config_->num_labels)); 

  //std::cout << "gold arcs: ";
  //sent.print_arcs();

  //shift ROOT symbol (probability 1)
  beam_chart[0][0]->shift(); 
  //std::cout << "shifted" << std::endl;

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 1; k <= sent.size(); ++k) {
    //there are k beam lists. perform reduces down to list 1

    for (unsigned i = k - 1; i > 0; --i) { 
      //prune if size exceeds beam_size
      if (beam_chart[i].size() > beam_size) {
        std::sort(beam_chart[i].begin(), beam_chart[i].end(), TransitionParser::cmp_particle_weights); 
        //remove items with worst scores
        for (unsigned j = beam_chart[i].size(); j > beam_size; --j)
          beam_chart[i].pop_back();
      }
      //std::cout << i << std::endl;
      //for every item in the list, add valid reduce actions to list i - 1 
      for (unsigned j = 0; (j < beam_chart[i].size()); ++j) {
	  //TODO find max labelled action
        //std::cerr << "predicting action" << std::endl;
        Reals action_probs = weights->predictAction(beam_chart[i][j]->actionContext());
        Real tot_reducep = log_one_min(action_probs.at(0));
        
        if (config_->direction_deterministic) {
          WordIndex reduce_pred = arg_min(action_probs, 1);
          //left arc invalid for stack size 2
	      if (i == 1) 
            reduce_pred = arg_min(action_probs, config_->num_labels + 1);

          Real reducep = action_probs[reduce_pred];
          kAction re_act = beam_chart[i][j]->lookup_action(reduce_pred);
          WordId re_label = beam_chart[i][j]->lookup_label(reduce_pred);
      
          //take best reduce action 
          beam_chart[i-1].push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_chart[i][j]));
          if (re_act == kAction::la) 
            beam_chart[i-1].back()->leftArc(re_label);
	      else
            beam_chart[i-1].back()->rightArc(re_label);
          beam_chart[i-1].back()->add_particle_weight(reducep);
          //if (k == sent.size())   
          //  beam_chart[i-1].back()->add_importance_weight(tot_reducep); 
        } else {
          //sort to find n-best reduce actions
          std::vector<int> indices(action_probs.size()-1);
          std::iota(indices.begin(), indices.end(), 1);
          //left arc invalid for stack size 2
          if (i == 1) {
            indices.resize(config_->num_labels);
            std::iota(indices.begin(), indices.end(), config_->num_labels + 1);
          }
          
          std::sort(indices.begin(), indices.end(), [&action_probs](const int i, const int j) 
                  {return (action_probs[i] < action_probs[j]);});
          for (unsigned l = 0; ((l < beam_size) && (l < 4)); ++l) {
            WordIndex reduce_pred = indices[l];

            Real reducep = action_probs[reduce_pred];
            kAction re_act = beam_chart[i][j]->lookup_action(reduce_pred);
            WordId re_label = beam_chart[i][j]->lookup_label(reduce_pred);
    
            //add action
            beam_chart[i-1].push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_chart[i][j]));
            if (re_act == kAction::la) 
              beam_chart[i-1].back()->leftArc(re_label);
	        else
              beam_chart[i-1].back()->rightArc(re_label);
            beam_chart[i-1].back()->add_particle_weight(reducep);
            //if (k == sent.size())   
            //  beam_chart[i-1].back()->add_importance_weight(tot_reducep); 
          }
        }
      }
    }

    if ((beam_chart[0].size() > beam_size) || (k == sent.size())) {
        std::sort(beam_chart[0].begin(), beam_chart[0].end(), TransitionParser::cmp_particle_weights); 
        //remove items with worst scores
        for (unsigned j = beam_chart[0].size(); j > beam_size; --j)
          beam_chart[0].pop_back();
    }

    //perform shifts
    if (k < sent.size()) {
      for (unsigned i = 0; (i < k); ++i) { 
        for (unsigned j = 0; j < beam_chart[i].size(); ++j) {
          Real shiftp = weights->predictAction(static_cast<WordId>(kAction::sh), 
                                                      beam_chart[i][j]->actionContext());
          Real tagp = weights->predictTag(beam_chart[i][j]->next_tag(), 
                                           beam_chart[i][j]->tagContext());
          Real wordp = weights->predictWord(beam_chart[i][j]->next_word(), 
                                             beam_chart[i][j]->wordContext());

          beam_chart[i][j]->shift();
          beam_chart[i][j]->add_particle_weight(shiftp); 
          beam_chart[i][j]->add_importance_weight(tagp); 
          beam_chart[i][j]->add_importance_weight(wordp); 
          beam_chart[i][j]->add_particle_weight(tagp); 
          beam_chart[i][j]->add_particle_weight(wordp); 
        }
      }
      //insert new beam_chart[0] to increment indexes
      beam_chart.insert(beam_chart.begin(), AslParserList());
    } 
    //std::cout << std::endl; 
  }
 
  unsigned n = 0; //index to final beam

  //sum over identical parses in final beam 
  vector<bool> duplicate(beam_chart[n].size(), false);

  if (config_->sum_over_beam) {
    for (unsigned i = 0; (i < beam_chart[n].size()-1); ++i) {
      if (!duplicate[i])
        for (unsigned j = i + 1; (j < beam_chart[n].size()); ++j) {
          if (ParsedSentence::eq_arcs(beam_chart[n][i], beam_chart[n][j])) {
            beam_chart[n][i]->add_log_particle_weight(beam_chart[n][j]->particle_weight());          
            duplicate[j] = true;
          }
        }
    } 
  
    std::sort(beam_chart[n].begin(), beam_chart[n].end(), TransitionParser::cmp_particle_weights); 
    //for (unsigned i = 0; i < duplicate.size(); ++i)
    //  std::cout << duplicate[i] << " ";
    //std::cout << std::endl; 
  }

  for (unsigned i = 0; (i < beam_chart[n].size()); ++i)
    if (!duplicate[i])
      beam_chart[n][0]->add_beam_weight(beam_chart[n][i]->particle_weight()); 

  //print parses
  //add verbose option?
  //for (unsigned i = 0; (i < 5) && (i < beam_chart[n].size()); ++i) {
    //std::cout << beam_chart[n][i]->particle_weight() << " ";
    //beam_chart[n][i]->print_arcs();
    //beam_chart[n][i]->print_actions();

    //can't do this now, but add if needed later
    //float dir_acc = (beam_chart[n][i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
    //std::cout << "  Dir Accuracy: " << dir_acc;
  //} 

  if (beam_chart[n].size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_->num_labels);  
  } else
    return ArcStandardLabelledParser(*beam_chart[n][0]); 
}



template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::particlePosteriorParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles,
        bool resample) {
    //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states

  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles), config_->num_labels)); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift
      int num_samples = beam_stack[j]->num_particles();

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      Real shiftp = action_probs[0];
      Real tot_reducep = log_one_min(shiftp);

      std::vector<int> sample_counts(config_->numActions(), 0);

      if (beam_stack[j]->stack_depth() <= 2) {
        //only shift is allowed
        sample_counts[0] += num_samples;
      } else {
        multinomial_distribution_log<Real> mult(action_probs); 
        for (int k = 0; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }
      }        
     
      //reduce actions
      for (WordId pred = 1; pred < sample_counts.size(); ++pred) {
        if (sample_counts[pred] > 0) {
          kAction re_act = beam_stack[j]->lookup_action(pred);
          WordId re_label = beam_stack[j]->lookup_label(pred);
          beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));

          if (re_act == kAction::la) 
            beam_stack.back()->leftArc(re_label);
          else
            beam_stack.back()->rightArc(re_label);

          beam_stack.back()->add_particle_weight(action_probs[pred]);
          beam_stack.back()->set_num_particles(sample_counts[pred]); 
        }
      } 

      //perform shift if > 0 samples
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      resampleParticles(&beam_stack, eng, num_particles);
      //sort and remove 
      //std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
      //for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
      //  beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
    }
  }
     
  //completion
 for (unsigned j = 0; j < beam_stack.size(); ++j) { 
    while ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->inTerminalConfiguration()) {
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      Real tot_reducep = log_one_min(action_probs[0]);
      int num_samples = beam_stack[j]->num_particles();
      
      //sample first reduce action
      action_probs[0] = L_MAX;
      multinomial_distribution_log<Real> mult(action_probs); 
      WordIndex reduce_pred = mult(eng);
      //for stack depth 2, deterministically take best action
      if (beam_stack[j]->stack_depth() == 2) 
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);

      Real reducep = action_probs[reduce_pred];
      kAction act = beam_stack[j]->lookup_action(reduce_pred);
      WordId label = beam_stack[j]->lookup_label(reduce_pred);
      
      if (act == kAction::la) 
        beam_stack[j]->leftArc(label);
      else
        beam_stack[j]->rightArc(label);
      beam_stack[j]->add_particle_weight(reducep);
      beam_stack.back()->add_importance_weight(tot_reducep); 
      beam_stack[j]->set_num_particles(num_samples);
 
      //sample more reduce actions 
	  if ((beam_stack[j]->stack_depth() > 2) && (num_samples > 1)) {
        std::vector<int> sample_counts(config_->numActions(), 0);
        for (int k = 1; k < num_samples; k++) {
          WordId action = mult(eng);
          ++sample_counts[action];
        }
        
        for (WordId pred = 1; pred < sample_counts.size(); ++pred) {
          if (pred == reduce_pred) {
            beam_stack[j]->set_num_particles(sample_counts[pred]+1);
          } else if (sample_counts[pred] > 0) {
            kAction re_act = beam_stack[j]->lookup_action(pred);
            WordId re_label = beam_stack[j]->lookup_label(pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (re_act == kAction::la) 
              beam_stack.back()->leftArc(re_label);
            else
              beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(action_probs[pred]);
            beam_stack.back()->add_importance_weight(tot_reducep); 
            beam_stack.back()->set_num_particles(sample_counts[pred]); 
          }
        }
      }  
    } 
  }
 
  if (resample) 
    resampleParticles(&beam_stack, eng, num_particles);

  std::vector<Reals> post_counts; //(sent.size(), Reals(sent.size(), 0));
  for (unsigned i = 0; i < sent.size(); ++i)
    post_counts.push_back(Reals(sent.size(), 0.0));

  //for each particle, add arcs to count
  //else fractional counts based on importance weights
  for (unsigned i = 0; i < beam_stack.size(); ++i) {
    if (beam_stack[i]->num_particles() == 0)
      continue;
    for (unsigned j = 1; j < beam_stack[i]->size(); ++j) {
      if (beam_stack[i]->arc_at(j) >= 0)
        post_counts[j][beam_stack[i]->arc_at(j)] += beam_stack[i]->num_particles();
    }
  } 
  
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(sent), 
          static_cast<int>(num_particles), 1); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    unsigned j = arg_max(post_counts[i], 0);
    parser.set_arc(i, j);
  } 

  return parser;

  //just take 1 sample
  //resampleParticles(&beam_stack, eng, 1);
  //for (unsigned i = 0; i < beam_stack.size(); ++i) 
  //  if (beam_stack[i]->num_particles() == 1) 
   //   return ArcStandardLabelledParser(*beam_stack[i]); 
  //std::cout << "no parse found" << std::endl;
  //return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_->num_labels);  
}



template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles,
        bool resample) {
    //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states

  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles), config_->num_labels)); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift
      int num_samples = beam_stack[j]->num_particles();

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      Real shiftp = action_probs[0];
      Real tot_reducep = log_one_min(shiftp);

      std::vector<int> sample_counts(config_->numActions(), 0);

      if (beam_stack[j]->stack_depth() <= 2) {
        //only shift is allowed
        sample_counts[0] += num_samples;
      } else {
        multinomial_distribution_log<Real> mult(action_probs); 
        for (int k = 0; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }
      }        
     
      //reduce actions
      for (WordId pred = 1; pred < sample_counts.size(); ++pred) {
        if (sample_counts[pred] > 0) {
          kAction re_act = beam_stack[j]->lookup_action(pred);
          WordId re_label = beam_stack[j]->lookup_label(pred);
          beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));

          if (re_act == kAction::la) 
            beam_stack.back()->leftArc(re_label);
          else
            beam_stack.back()->rightArc(re_label);

          beam_stack.back()->add_particle_weight(action_probs[pred]);
          beam_stack.back()->set_num_particles(sample_counts[pred]); 
        }
      } 

      //perform shift if > 0 samples
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    if (resample) {
      resampleParticles(&beam_stack, eng, num_particles);
      //sort and remove 
      //std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
      //for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
      //  beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
    }
  }
     
  //completion
 for (unsigned j = 0; j < beam_stack.size(); ++j) { 
    while ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->inTerminalConfiguration()) {
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      Real tot_reducep = log_one_min(action_probs[0]);
      int num_samples = beam_stack[j]->num_particles();
      
      //sample first reduce action
      action_probs[0] = L_MAX;
      multinomial_distribution_log<Real> mult(action_probs); 
      WordIndex reduce_pred = mult(eng);
      //for stack depth 2, deterministically take best action
      if (beam_stack[j]->stack_depth() == 2) 
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);

      Real reducep = action_probs[reduce_pred];
      kAction act = beam_stack[j]->lookup_action(reduce_pred);
      WordId label = beam_stack[j]->lookup_label(reduce_pred);
      
      if (act == kAction::la) 
        beam_stack[j]->leftArc(label);
      else
        beam_stack[j]->rightArc(label);
      beam_stack[j]->add_particle_weight(reducep);
      beam_stack.back()->add_importance_weight(tot_reducep); 
      beam_stack[j]->set_num_particles(num_samples);
 
      //sample more reduce actions 
	  if ((beam_stack[j]->stack_depth() > 2) && (num_samples > 1)) {
        std::vector<int> sample_counts(config_->numActions(), 0);
        for (int k = 1; k < num_samples; k++) {
          WordId action = mult(eng);
          ++sample_counts[action];
        }
        
        for (WordId pred = 1; pred < sample_counts.size(); ++pred) {
          if (pred == reduce_pred) {
            beam_stack[j]->set_num_particles(sample_counts[pred]+1);
          } else if (sample_counts[pred] > 0) {
            kAction re_act = beam_stack[j]->lookup_action(pred);
            WordId re_label = beam_stack[j]->lookup_label(pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (re_act == kAction::la) 
              beam_stack.back()->leftArc(re_label);
            else
              beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(action_probs[pred]);
            beam_stack.back()->add_importance_weight(tot_reducep); 
            beam_stack.back()->set_num_particles(sample_counts[pred]); 
          }
        }
      }  
    } 
  }
  
  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  //std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
  //for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
  //  beam_stack.pop_back();
  //std::cout << "Final beam size: " << beam_stack.size();
  
  //resampleParticles(&beam_stack, eng, num_particles);
  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 

  for (unsigned i = 0; (i < beam_stack.size()); ++i)
    if (beam_stack[i]->num_particles() > 0) {
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight()); 
      std::cout <<  beam_stack[i]->particle_weight() << "," << beam_stack[0]->beam_weight() << " ";
}
  std::cout << std::endl;
  //std::cout << beam_stack[0]->size() << " " << beam_stack[0]->particle_weight() << " " << beam_stack[0]->beam_weight() << std::endl;

  //just take 1 sample
  //resampleParticles(&beam_stack, eng, 1);
  //for (unsigned i = 0; i < beam_stack.size(); ++i) 
    if (beam_stack[0]->num_particles() > 0) 
      return ArcStandardLabelledParser(*beam_stack[0]); 

  std::cout << "no parse found" << std::endl;
  return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_->num_labels);  
}

//sample a derivation for the gold parse, given the current model
//three-way decisions
template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::particleGoldParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles, bool resample) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
  
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles), config_->num_labels)); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    //sample a sequence of possible actions leading up to the next shift
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      //std::cout << i << " " << j << std::endl;
      int num_samples = beam_stack[j]->num_particles();
      if (num_samples == 0)
        continue;

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int shift_count = 0;
      int reduce_count = 0;
      Real shiftp = action_probs[0];

      kAction oracle_next = beam_stack[j]->oracleNext(sent);
      WordId lab_next = beam_stack[j]->oracleNextLabel(sent); 
      WordId pred = beam_stack[j]->convert_action(oracle_next, lab_next);
      Real tot_reducep = log_one_min(shiftp);      

      if (oracle_next==kAction::sh) {
        //only shift is allowed
        shift_count = num_samples;
        beam_stack[j]->add_importance_weight(shiftp);  
      } else if (num_samples == 1) {
        reduce_count = num_samples;
      } else {
        multinomial_distribution_log<Real> mult(action_probs); 
        for (int k = 0; k < num_samples; k++) {
          WordId act = mult(eng);
          if (act == 0)
            ++shift_count;
          else 
            ++reduce_count;
        }
                    
        if (reduce_count > 0) {
          kAction re_act = beam_stack[j]->lookup_action(pred);
          WordId re_label = beam_stack[j]->lookup_label(pred);

          beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
          if (re_act == kAction::la) 
            beam_stack.back()->leftArc(re_label);
	      else
            beam_stack.back()->rightArc(re_label);
          beam_stack.back()->add_particle_weight(action_probs[pred]);
          beam_stack.back()->set_num_particles(reduce_count); 
          beam_stack.back()->add_importance_weight(tot_reducep - action_probs[pred]); 
        }
      } 

      //perform shift if > 0 samples
      if (shift_count == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(shift_count);
      }
    }
 
    if (resample)
      resampleParticles(&beam_stack, eng, num_particles);
    //sort to remove 0's
    //std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
    //for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    //  beam_stack.pop_back();
  }
  //std::cout << std::endl;   

  //std::cout << "completion" << std::endl;
  //completion
  for (unsigned j = 0; j < beam_stack.size(); ++j) { 
    while ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->inTerminalConfiguration()) {
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();

      kAction oracle_next = beam_stack[j]->oracleNext(sent);
      WordId label_next = beam_stack[j]->oracleNextLabel(sent); 
      WordId reduce_pred = beam_stack[j]->convert_action(oracle_next, label_next);
      //std::cout << j << " " << beam_stack[j]->stack_depth() << " " << reduce_pred << std::endl;
      Real tot_reducep = log_one_min(action_probs[0]);      

      action_probs[0] = L_MAX;
      multinomial_distribution_log<Real> mult(action_probs); 

      if ((beam_stack[j]->stack_depth() == 2) && (oracle_next == kAction::re)) {
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);
        beam_stack[j]->add_importance_weight(tot_reducep); 
      } else if (oracle_next == kAction::re) {
        beam_stack[j]->set_num_particles(0);
        continue;
        //reduce_pred = mult(eng);
        //beam_stack[j]->add_importance_weight(tot_reducep); 
        //hack to penalize inconsistent reduce decisions
        //imagine actually making a shift decision
        //beam_stack[j]->add_importance_weight(action_probs[0]); 
      } else {
        beam_stack[j]->add_importance_weight(action_probs[reduce_pred]); 
      }

      Real reducep = action_probs[reduce_pred];
      kAction act = beam_stack[j]->lookup_action(reduce_pred);
      WordId label = beam_stack[j]->lookup_label(reduce_pred);
      //std::cout << reduce_pred << std::endl;

      //take oracle or sampled action
      if (act == kAction::la) 
        beam_stack[j]->leftArc(label);
      else
        beam_stack[j]->rightArc(label);
      beam_stack[j]->add_particle_weight(reducep);
      beam_stack[j]->set_num_particles(num_samples);

      if ((beam_stack[j]->stack_depth() > 2) && (oracle_next == kAction::sh) && (num_samples > 1)) {
        //sample more actions
        std::vector<int> sample_counts(config_->num_actions, 0);
        for (int k = 1; k < num_samples; k++) {
          WordId action = mult(eng);
          ++sample_counts[action];
        }
        
        for (int pred = 1; pred < sample_counts.size(); ++pred) {
          if (pred == reduce_pred) {
            beam_stack[j]->set_num_particles(sample_counts[pred]+1);
          } else if (sample_counts[pred] > 0) {
            kAction re_act = beam_stack[j]->lookup_action(pred);
            WordId re_label = beam_stack[j]->lookup_label(pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (re_act == kAction::la) 
              beam_stack.back()->leftArc(re_label);
            else
              beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(action_probs[pred]);
            beam_stack.back()->set_num_particles(sample_counts[pred]); 
          }
        }
      }   
    } 
  } 

  //just take 1 sample
  resampleParticles(&beam_stack, eng, 1);
  for (unsigned i = 0; i < beam_stack.size(); ++i) 
    if (beam_stack[i]->num_particles() == 1) 
      return ArcStandardLabelledParser(*beam_stack[i]); 

  std::cout << "no parse found" << std::endl;
  return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_->num_labels);  
}

template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::particleMaxParseSentence(
        const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, 
        unsigned num_particles) {
  //perform sampling and resampling to update these counts, and remove 0 count states
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles), config_->num_labels)); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    //sample a sequence of possible actions leading up to the next shift
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      int num_samples = beam_stack[j]->num_particles();
      if (num_samples == 0)
        continue;

      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      std::vector<int> sample_counts(config_->numActions(), 0);
      Real shiftp = action_probs[0];

      if (beam_stack[j]->stack_depth() <= 2) {
        sample_counts[0] = num_samples;
      } else { 
        //let the first sampled reduce action always be the best one
        WordIndex reduce_pred = arg_min(action_probs, 1);
        bool reduce_fixed = false;

        multinomial_distribution_log<Real> mult(action_probs); 
        for (int k = 0; k < num_samples; k++) {
          WordId act = mult(eng);
          if (act == 0)
            ++sample_counts[0];
          else if (reduce_fixed) {
            ++sample_counts[reduce_pred];
            if (!config_->direction_deterministic)
              reduce_fixed = false;
          } else
            ++sample_counts[act];
        }
                    
        for (int pred = 1; pred < sample_counts.size(); ++pred) {
          if (sample_counts[pred] > 0) {
            kAction re_act = beam_stack[j]->lookup_action(pred);
            WordId re_label = beam_stack[j]->lookup_label(pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (re_act == kAction::la) 
              beam_stack.back()->leftArc(re_label);
	        else
              beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(action_probs[pred]);
            beam_stack.back()->set_num_particles(sample_counts[pred]); 
          }
        }
      } 

      //perform shift if > 0 samples
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp); 
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(tagp); 
        beam_stack[j]->set_num_particles(sample_counts[0]);
      }
    }
 
    resampleParticleParticles(&beam_stack, eng, num_particles);
    //sort to remove 0's
    //std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
    //for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    //  beam_stack.pop_back();
  }
  //std::cout << std::endl;   

  //completion
  for (unsigned j = 0; j < beam_stack.size(); ++j) { 
    while ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->inTerminalConfiguration()) {
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      int num_samples = beam_stack[j]->num_particles();

      //greedily reduce each item
      WordIndex reduce_pred = arg_min(action_probs, 1);
      //left arc invalid for stack size 2
      if (beam_stack[j]->stack_depth() == 2) 
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);

      Real reducep = action_probs[reduce_pred];
      kAction act = beam_stack[j]->lookup_action(reduce_pred);
      WordId label = beam_stack[j]->lookup_label(reduce_pred);
      
      if (act == kAction::la) 
        beam_stack[j]->leftArc(label);
      else
        beam_stack[j]->rightArc(label);
      beam_stack[j]->add_particle_weight(reducep);
      beam_stack[j]->set_num_particles(num_samples);

	  if (config_->direction_deterministic && (beam_stack[j]->stack_depth() > 2) && (num_samples > 1)) {
        action_probs[0] = L_MAX;
        std::vector<int> sample_counts(config_->numActions(), 0);
        multinomial_distribution_log<Real> mult(action_probs); 
        for (int k = 1; k < num_samples; k++) {
          WordId action = mult(eng);
          ++sample_counts[action];
        }
        
        for (int pred = 1; pred < sample_counts.size(); ++pred) {
          if (pred == reduce_pred) {
            beam_stack[j]->set_num_particles(sample_counts[pred]+1);
          } else if (sample_counts[pred] > 0) {
            kAction re_act = beam_stack[j]->lookup_action(pred);
            WordId re_label = beam_stack[j]->lookup_label(pred);

            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
            if (re_act == kAction::la) 
              beam_stack.back()->leftArc(re_label);
            else
              beam_stack.back()->rightArc(re_label);
            beam_stack.back()->add_particle_weight(action_probs[pred]);
            beam_stack.back()->set_num_particles(sample_counts[pred]); 
          }
        }
      }  
    } 
  } 

  unsigned n = 0; //index to final beam

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

  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 

  for (unsigned i = 0; (i < beam_stack.size()); ++i)
    if (!duplicate[i] && (beam_stack[i]->num_particles() > 0)) 
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight()); 
  
  //std::cout << beam_stack[0]->size() << " " << beam_stack[0]->particle_weight() << " " << beam_stack[0]->beam_weight() << std::endl;

  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_->num_labels);  
  } else
    return ArcStandardLabelledParser(*beam_stack[0]); 
}

*/

template class ArcStandardLabelledParseModel<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardLabelledParseModel<ParsedCALexPypWeights<wordLMOrderAS, wordTagLMOrderAS, tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardLabelledParseModel<ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardLabelledParseModel<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardLabelledParseModel<ParsedFactoredWeights>;
template class ArcStandardLabelledParseModel<DiscriminativeWeights>;
template class ArcStandardLabelledParseModel<ParsedWeights>;

}

