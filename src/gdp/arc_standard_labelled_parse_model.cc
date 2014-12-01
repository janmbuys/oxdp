#include "gdp/arc_standard_labelled_parse_model.h"

namespace oxlm {

//TODO update constructors, actions, etc
template<class ParsedWeights>
ArcStandardLabelledParseModel<ParsedWeights>::ArcStandardLabelledParseModel(boost::shared_ptr<ModelConfig> config):
  config_(config)
{
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::resampleParticleParticles(AslParserList* beam_stack, MT19937& eng,
        unsigned num_particles) {
  std::vector<Real> importance_w(beam_stack->size(), L_MAX); 
  for (unsigned i = 0; i < importance_w.size(); ++i) 
    if (beam_stack->at(i)->num_particles() > 0)
      importance_w[i] = beam_stack->at(i)->weighted_particle_weight();

  //resample according to importance weight
  multinomial_distribution_log<Real> part_mult(importance_w); 
  std::vector<int> sample_counts(beam_stack->size(), 0);
  for (unsigned i = 0; i < num_particles;) {
    unsigned pi = part_mult(eng);
    ++sample_counts[pi];
    ++i;
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
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::greedyParseSentence(const ParsedSentence& sent, 
                               const boost::shared_ptr<ParsedWeights>& weights) {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(sent), config_->num_labels);
  parser.shift(); 

  //if greedy, we are effectively ignoring word predictions
  for (unsigned k = 1; k < sent.size(); ++k) {
    Reals action_probs = weights->predictAction(parser.actionContext());
    WordIndex pred = arg_min(action_probs, 0);
    if (parser.stack_depth() <= 2) //don't want to add root before the end
      pred = 0;
    //std::cout << pred << "," << action_probs[pred] << " ";
    
    //reduce until shift action is chosen
    while (pred > 0) {
      //std::cout << "re ";
      kAction re_act = parser.lookup_action(pred);
      WordId re_label = parser.lookup_label(pred);
      if (re_act == kAction::la) 
        parser.leftArc(re_label);
	  else
        parser.rightArc(re_label);
      parser.add_particle_weight(action_probs[pred]);

      action_probs = weights->predictAction(parser.actionContext());
      pred = arg_min(action_probs, 0);
      if (parser.stack_depth() <= 2) 
        pred = 0;
      //std::cout << pred << "," << action_probs[pred] << " ";
    }
    
    //shift    
    if (k < sent.size()) {
      //std::cout << "sh ";
      //Real tagp = weights->predictTag(parser.next_tag(), parser.tagContext());
      //Real wordp = weights->predictWord(parser.next_word(), parser.wordContext());
      parser.shift();
      //parser.add_particle_weight(tagp);
      //parser.add_particle_weight(wordp);
    }
  }

  //completion
  while (!parser.inTerminalConfiguration()) {
    //std::cout << "re ";
    Reals action_probs = weights->predictAction(parser.actionContext());
    WordIndex pred = arg_min(action_probs, 1);
    if (parser.stack_depth() == 2) 
      pred = arg_min(action_probs, config_->num_labels + 1);
    //std::cout << pred << "," << action_probs[pred] << " ";
    kAction re_act = parser.lookup_action(pred);
    WordId re_label = parser.lookup_label(pred);
    
    if (re_act == kAction::la) 
      parser.leftArc(re_label);
	else
      parser.rightArc(re_label);
    parser.add_particle_weight(action_probs[pred]);
  }
  
  //std::cout << std::endl;

  return parser;
}

template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::beamDiscriminativeParseSentence(const ParsedSentence& sent, 
                               const boost::shared_ptr<ParsedWeights>& weights, unsigned beam_size) {
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), config_->num_labels)); 
  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  //keep beam per actions

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 1; k <= 2*(sent.size()-1); ++k) {
    //prune if size exceeds beam_size
    if (beam_stack.size() > beam_size) {
      std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
      //remove items with worst scores
      for (int j = beam_stack.size()- 1; ((j >= beam_size) || (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
    }
      
    //for every item in the beam, add some valid actions 
    unsigned stack_size = beam_stack.size();
    for (unsigned j = 0; (j < stack_size); ++j) {
      //std::cout << k << " " << stack_size << " " << j << " " << beam_stack[j]->stack_depth() << std::endl;
      if (beam_stack[j]->num_particles() == 0)
        continue;
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      Real shiftp = action_probs[0];
        
      if ((beam_stack[j]->buffer_empty() && beam_stack[j]->stack_depth() == 2) || ((beam_stack[j]->stack_depth() >= 3) && config_->direction_deterministic)) {
        WordIndex reduce_pred = arg_min(action_probs, 1);
        //left arc invalid for stack size 2
	    if (beam_stack[j]->stack_depth() == 2) 
          reduce_pred = arg_min(action_probs, config_->num_labels + 1);

        Real reducep = action_probs[reduce_pred];
        kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
        WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
      
        //take best reduce action 
        //if (reducep < shiftp) {
        //if (shiftp < -std::log(0.5)) {
          beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
          if (re_act == kAction::la) 
            beam_stack.back()->leftArc(re_label);
	      else
            beam_stack.back()->rightArc(re_label);
          //std::cout << "re ";
          beam_stack.back()->add_particle_weight(reducep);
        //}

        //take shift action if possible
        if (!beam_stack[j]->buffer_empty()) {
          beam_stack[j]->shift();
          beam_stack[j]->add_particle_weight(shiftp); 
          //std::cout << "sh ";
        } else {
          beam_stack[j]->set_num_particles(0);
        }
      } else if (beam_stack[j]->stack_depth() >= 3) {
        //sort to find n-best actions
        std::vector<int> indices(action_probs.size()-1);
        std::iota(indices.begin(), indices.end(), 0);
        bool has_shift = false;   
               
        std::sort(indices.begin(), indices.end(), [&action_probs](const int i, const int j) 
                {return (action_probs[i] < action_probs[j]);});
        for (unsigned l = 0; ((l < beam_size) && (l < 4)); ++l) {
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
        //take shift action if prefered
        if (has_shift && !beam_stack[j]->buffer_empty()) {
          beam_stack[j]->shift();
          beam_stack[j]->add_particle_weight(shiftp); 
        } else {
          beam_stack[j]->set_num_particles(0);
        }
      } else {
        //only shift valid
        //std::cout << "sh1 ";
        beam_stack[j]->shift();
        beam_stack[j]->add_particle_weight(shiftp); 
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

  for (unsigned i = 0; (i < beam_stack.size()); ++i)
    if (!duplicate[i])
      beam_stack[0]->add_beam_weight(beam_stack[i]->particle_weight()); 

  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
  //print parses
  //add verbose option?
  //for (unsigned i = 0; (i < 5) && (i < beam_stack.size()); ++i) {
    //std::cout << beam_stack[i]->particle_weight() << " ";
    //beam_stack[i]->print_arcs();
    //beam_stack[i]->print_actions();

    //can't do this now, but add if needed later
    //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
    //std::cout << "  Dir Accuracy: " << dir_acc;
  //} 

  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_->num_labels);  
  } else
    return ArcStandardLabelledParser(*beam_stack[0]); 
}


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
          if (k == sent.size())   
            beam_chart[i-1].back()->add_importance_weight(tot_reducep); 
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
            if (k == sent.size())   
              beam_chart[i-1].back()->add_importance_weight(tot_reducep); 
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
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::beamLinearParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, unsigned beam_size) {
  AslParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(static_cast<TaggedSentence>(sent), 
              1, config_->num_labels)); 
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    if (beam_stack.size() > beam_size) {
      //std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
      std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
      for (int j = beam_stack.size()- 1; (j >= beam_size); --j)
        beam_stack.pop_back();
    }

    double worst_weight = beam_stack[0]->particle_weight();

    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      Real shiftp = action_probs[0];
             
	  if (beam_stack[j]->stack_depth() >= 3) { //&& (shiftp > -std::log(0.5))) {
             //&& ((j < beam_size) || (beam_stack[j]->particle_weight() < worst_weight))) { 
        //add best reduce action
        WordIndex reduce_pred = arg_min(action_probs, 1);

        Real reducep = action_probs[reduce_pred];
        kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
        WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
      
        if ((j < beam_size) || (beam_stack[j]->particle_weight() + reducep < worst_weight)) {
          beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
          if (re_act == kAction::la) 
            beam_stack.back()->leftArc(re_label);
	      else
            beam_stack.back()->rightArc(re_label);
          beam_stack.back()->add_particle_weight(reducep);
        }
      }

      if (beam_stack[j]->particle_weight() > worst_weight)
        worst_weight = beam_stack[j]->particle_weight();

      //shift
      Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
      Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

      beam_stack[j]->shift();
      beam_stack[j]->add_particle_weight(shiftp); 
      beam_stack[j]->add_importance_weight(wordp); 
      beam_stack[j]->add_particle_weight(wordp); 
      beam_stack[j]->add_importance_weight(tagp); 
      beam_stack[j]->add_particle_weight(tagp); 
    }
  }
    
  //completion
  for (unsigned j = 0; j < beam_stack.size(); ++j) { 
    //greedily reduce each item
    while (!beam_stack[j]->inTerminalConfiguration()) {
      Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
      WordIndex reduce_pred = arg_min(action_probs, 1);
      //left arc invalid for stack size 2
	  if (beam_stack[j]->stack_depth() == 2) 
        reduce_pred = arg_min(action_probs, config_->num_labels + 1);

      Real reducep = action_probs[reduce_pred];
      kAction re_act = beam_stack[j]->lookup_action(reduce_pred);
      WordId re_label = beam_stack[j]->lookup_label(reduce_pred);
      
      if (re_act == kAction::la) 
        beam_stack[j]->leftArc(re_label);
	  else
        beam_stack[j]->rightArc(re_label);
      beam_stack.back()->add_particle_weight(reducep);
    }
  } 
  
  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
 
  if (beam_stack.size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardLabelledParser(static_cast<TaggedSentence>(sent), config_->num_labels);  
  } else
    return ArcStandardLabelledParser(*beam_stack[0]); 
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
      std::vector<int> sample_counts(config_->num_actions, 0);
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
        std::vector<int> sample_counts(config_->num_actions, 0);
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

      std::vector<int> sample_counts(config_->num_actions, 0);

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
      /* int active_particle_count = 0;
       for (int j = 0; j < beam_stack.size(); ++j)
        if (beam_stack[j]->num_particles() > 0)
         ++active_particle_count;
      std::cout << " -> " << active_particle_count << " without null \n"; */
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
        std::vector<int> sample_counts(config_->num_actions, 0);
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
  
  /*
  AslParserList final_beam; 
  bool has_more_states = true;

  while (has_more_states) {
    has_more_states = false;
    unsigned cur_beam_size = beam_stack.size();

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->inTerminalConfiguration()) {
        //add paths for reduce actions
        has_more_states = true; 

      	Reals action_probs = weights->predictAction(beam_stack[j]->actionContext());
        Real tot_reducep = log_one_min(action_probs[0]);

        int num_samples = beam_stack[j]->num_particles();
        std::vector<int> sample_counts(config_->num_actions, 0);
       
        //disallow shift
        action_probs[0] = L_MAX;

        if (beam_stack[j]->stack_depth() == 2) {
          //left arcs disallowed
       	  for (int k = 0; k < config_->num_labels; ++k)
            action_probs[k+1] = L_MAX;
        }

        multinomial_distribution_log<Real> mult(action_probs); 
        for (int k = 0; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }

        //reduce actions
        for (int k = 1; k < sample_counts.size(); ++k) {
          if (sample_counts[k] > 0) {
    //check that this isn't wrong for some obscure reason
            beam_stack.push_back(boost::make_shared<ArcStandardLabelledParser>(*beam_stack[j]));
          if (k <= config_->num_labels) 
            beam_stack.back()->leftArc(k-1);
          else
            beam_stack.back()->rightArc(k-config_->num_labels-1);

            beam_stack.back()->add_particle_weight(action_probs[k]);
            beam_stack.back()->set_num_particles(sample_counts[k]); 
            beam_stack.back()->add_importance_weight(tot_reducep); 
          }
	    }
      } 
    }

    if (resample) {
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, eng, num_particles);
      //int active_particle_count = 0;
      //for (int j = 0; j < beam_stack.size(); ++j)
      //  if (beam_stack[j]->num_particles() > 0)
      //    ++active_particle_count;
      //std::cout << " -> " << active_particle_count << " without null \n";
    }
  }  */

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  //std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
  //for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
  //  beam_stack.pop_back();
  //std::cout << "Final beam size: " << beam_stack.size();

  /* if ((beam_stack.size() > 0) && take_max) {
    //resampleParticles(&beam_stack, eng, num_particles);
    std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
    return ArcStandardLabelledParser(*beam_stack[0]);
  } */
  
 //just take 1 sample
  resampleParticles(&beam_stack, eng, 1);
  for (unsigned i = 0; i < beam_stack.size(); ++i) 
    if (beam_stack[i]->num_particles() == 1) 
      return ArcStandardLabelledParser(*beam_stack[i]); 

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
      resampleParticleParticles(&beam_stack, eng, num_particles);
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

    /*  if ((beam_stack[j]->stack_depth() > 2) && (oracle_next == kAction::sh) && (num_samples > 1)) {
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
      } */  
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
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::staticGoldParseSentence(const ParsedSentence& sent, 
                                    const boost::shared_ptr<ParsedWeights>& weights) {
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(sent), config_->num_labels);
  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && (a != kAction::re)) {
    a = parser.oracleNext(sent);  
    WordId lab = parser.oracleNextLabel(sent);
    if (a != kAction::re) {
      //update particle weight
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
  ArcStandardLabelledParser parser(static_cast<TaggedSentence>(sent), config_->num_labels);
  //std::cout << "Oracle: "; 
  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && (a != kAction::re)) {
    a = parser.oracleNext(sent);  
    WordId lab = parser.oracleNextLabel(sent);
    //std::cout << static_cast<WordId>(a) << "," << lab << " ";
    if (a != kAction::re) 
      parser.executeAction(a, lab);
  }
  //std::cout << std::endl;

  return parser;
}

//generate a sentence: ternary decisions
template<class ParsedWeights>
ArcStandardLabelledParser ArcStandardLabelledParseModel<ParsedWeights>::generateSentence(const boost::shared_ptr<ParsedWeights>& weights, 
        MT19937& eng) {
  unsigned sent_limit = 100;
  ArcStandardLabelledParser parser(config_->num_labels);
  bool terminate_shift = false;
  parser.push_tag(0);
  parser.shift(1);
    
  do {
    WordId pred = 0; //placeholder action (shift)
    //std::cerr << "arcs: " << parser.arcs().size() << std::endl;
    
    if (parser.stack_depth() >= 2) {
      if (parser.size() == sent_limit) {
        std::cout << "  LL ";
        terminate_shift = true;
      }

      Reals action_probs = weights->predictAction(parser.actionContext());
  
      if (terminate_shift)
         action_probs[0] = L_MAX;
      if (parser.stack_depth() == 2) {
        for (int k = 0; k < config_->num_labels; ++k)
          action_probs[k+1] = L_MAX;
      }

      //sample an action
      multinomial_distribution_log<Real> mult(action_probs); 
      pred = mult(eng);
      parser.add_particle_weight(action_probs[pred]);
      
      //std::cout << "(" << parser.stack_depth() << ") ";
      //std::cout << pred << ":";
    }
      
    kAction act = parser.lookup_action(pred);
    WordId lab = parser.lookup_label(pred);
    //std::cout << pred << "," << static_cast<int>(act) << " ";

    if (act == kAction::la) {
      parser.leftArc(lab);
      //std::cout << "la ";
    } else if (act == kAction::ra) {
      parser.rightArc(lab);
      //std::cout << "ra ";
    } else if (act == kAction::sh) {
      //sample a tag - disallow root tag
      Reals tag_distr = weights->predictTag(parser.tagContext());
      tag_distr[0] = L_MAX;  

      multinomial_distribution_log<Real> t_mult(tag_distr);
      WordId tag = t_mult(eng);
      Real tagp = tag_distr[tag];
      parser.push_tag(tag);
      parser.add_particle_weight(tagp);

      //sample a word 
      Reals word_distr = weights->predictWord(parser.wordContext());
      word_distr[0] = L_MAX;  

      multinomial_distribution_log<Real> w_mult(word_distr);
      WordId word = w_mult(eng);

      Real wordp = word_distr[word];
      parser.shift(word);
      parser.add_particle_weight(wordp);
      //std::cout << "sh ";
      
      //terminate generation if word is EOS punctuation
      if (word == config_->stop_id || word == config_-> ques_id)
        terminate_shift = true;
    }
    
  } while ((parser.stack_depth() > 1)); // && !terminate_shift);

  //std::cout << std::endl;
  return parser;
} 

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardLabelledParser parse = staticGoldParseSentence(sent); 
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  unsigned beam_size = 8;
  ArcStandardLabelledParser parse = staticGoldParseSentence(sent, weights);
  //ArcStandardLabelledParser parse = beamParseSentence(sent, weights, beam_size);
  //std::cout << "Gold actions: ";
  //parse.print_actions();
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng,
          const boost::shared_ptr<ParseDataSet>& examples) {
  //unsigned num_particles = 100;
  //bool resample = true;

  //ArcStandardLabelledParser parse = staticGoldParseSentence(sent, weights);
  //ArcStandardLabelledParser parse = particleGoldParseSentence(sent, weights, eng, config_->num_particles, config_->resample);
  ArcStandardLabelledParser  parse = particleMaxParseSentence(sent, weights, eng, 256);
  //parse.print_actions();
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng,
          const boost::shared_ptr<ParseDataSet>& examples) {
  //ArcStandardLabelledParser  parse = particleMaxParseSentence(sent, weights, eng, 32);
  ArcStandardLabelledParser parse = particleParseSentence(sent, weights, eng, config_->num_particles, config_->resample);
  //parse.print_sentence();
  //parse.print_arcs();
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcStandardLabelledParseModel<ParsedWeights>::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  unsigned beam_size = 8;
  ArcStandardLabelledParser parse = beamLinearParseSentence(sent, weights, beam_size);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
Parser ArcStandardLabelledParseModel<ParsedWeights>::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) {
  Words ctx(7, 0);
  
  ArcStandardLabelledParser parse(config_->num_labels);
  if (beam_size == 0)
    parse = greedyParseSentence(sent, weights);
  else
    //parse = beamParseSentence(sent, weights, beam_size);
    parse = beamDiscriminativeParseSentence(sent, weights, beam_size);
    //parse = beamLinearParseSentence(sent, weights, beam_size);

  acc_counts->countAccuracy(parse, sent);
  ArcStandardLabelledParser gold_parse = staticGoldParseSentence(sent, weights);
  //parse.print_actions();
  //parse.print_arcs();
  //parse.print_labels();

  acc_counts->countLikelihood(parse.weight(), gold_parse.weight());
  return parse;
}

template<class ParsedWeights>
Parser ArcStandardLabelledParseModel<ParsedWeights>::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) {
  bool resample = true;

  //eval with particle parse
  //ArcStandardLabelledParser parse = particleParseSentence(sent, weights, eng, beam_size, resample);
  ArcStandardLabelledParser parse(config_->num_labels);
  if (beam_size == 0)
    parse = greedyParseSentence(sent, weights);
  else 
    parse = particleMaxParseSentence(sent, weights, eng, beam_size);
  acc_counts->countAccuracy(parse, sent);
  ArcStandardLabelledParser gold_parse = staticGoldParseSentence(sent, weights);
  
  acc_counts->countLikelihood(parse.weight(), gold_parse.weight());
  return parse;
}

template class ArcStandardLabelledParseModel<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardLabelledParseModel<ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardLabelledParseModel<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardLabelledParseModel<ParsedFactoredWeights>;

}

