#include "gdp/arc_standard_parse_model.h"

namespace oxlm {


template<class ParsedWeights>
ArcStandardParseModel<ParsedWeights>::ArcStandardParseModel(boost::shared_ptr<ModelConfig> config):
  config_(config)
{
}

template<class ParsedWeights>
void ArcStandardParseModel<ParsedWeights>::resampleParticles(AsParserList* beam_stack, MT19937& eng,
        unsigned num_particles) {
  //assume (beam_stack->at(pi)->num_particles() > 0)
  std::vector<Real> importance_w(beam_stack->size(), L_MAX); 
  for (unsigned i = 0; i < importance_w.size(); ++i) 
    importance_w[i] = beam_stack->at(i)->weighted_importance_weight();

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
    beam_stack->at(i)->reset_importance_weight();
  }
}

template<class ParsedWeights>
ArcStandardParser ArcStandardParseModel<ParsedWeights>::greedyParseSentence(const ParsedSentence& sent, 
                               const boost::shared_ptr<ParsedWeights>& weights) {
  ArcStandardParser parser(static_cast<TaggedSentence>(sent), config_->num_labels);
  
  parser.shift(); 

  //if greedy, we are effectively ignoring word predictions
  for (unsigned k = 1; k < sent.size(); ++k) {
    Reals action_probs = weights->predictAction(parser.actionContext());
    WordIndex pred = arg_min(action_probs, 0);
    if (parser.stack_depth() <= 2) //don't want to add root before the end
      pred = 0;
    
    //reduce until shift action is chosen
    while (pred > 0) {
      //std::cout << "re ";
      if (pred == 1)
        parser.leftArc();
	  else
        parser.rightArc();
      parser.add_particle_weight(action_probs[pred]);

      action_probs = weights->predictAction(parser.actionContext());
      pred = arg_min(action_probs, 0);
      if (parser.stack_depth() <= 2) 
        pred = 0;
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
      pred = 2; 
    
    if (pred == 1) 
      parser.leftArc();
	else
      parser.rightArc();
    parser.add_particle_weight(action_probs[pred]);
  }
  
  //std::cout << std::endl;

  return parser;
}

template<class ParsedWeights>
ArcStandardParser ArcStandardParseModel<ParsedWeights>::beamParseSentence(const ParsedSentence& sent, 
                               const boost::shared_ptr<ParsedWeights>& weights, unsigned beam_size) {
  //bool direction_deterministic = false;
    
  std::vector<AsParserList> beam_chart; 
  beam_chart.push_back(AsParserList());
  beam_chart[0].push_back(boost::make_shared<ArcStandardParser>(static_cast<TaggedSentence>(sent))); 

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
        Real reduceleftarcp = weights->predictAction(static_cast<WordId>(kAction::la), beam_chart[i][j]->actionContext());
        Real reducerightarcp = weights->predictAction(static_cast<WordId>(kAction::ra), beam_chart[i][j]->actionContext());
        //std::cout << j << " (la: " << reduceleftarcp << ", ra: " << reducerightarcp << ")" << " ";
        Real reducep = neg_log_sum_exp(reduceleftarcp, reducerightarcp);
       
        //TODO so adding both to the same list is giving an issue

        //left arc only invalid when stack size is 2 **
        if ((i > 1) && (!config_->direction_deterministic || (reduceleftarcp < reducerightarcp))) { 
          beam_chart[i-1].push_back(boost::make_shared<ArcStandardParser>(*beam_chart[i][j]));
          beam_chart[i-1].back()->leftArc();
          beam_chart[i-1].back()->add_particle_weight(reduceleftarcp);
          if (k == sent.size())   
            beam_chart[i-1].back()->add_importance_weight(reducep); 
        } 
        
        //right arc 
        if ((i==1) || (!config_->direction_deterministic || (reducerightarcp <= reduceleftarcp))) {
          beam_chart[i-1].push_back(boost::make_shared<ArcStandardParser>(*beam_chart[i][j]));
          beam_chart[i-1].back()->rightArc();
          beam_chart[i-1].back()->add_particle_weight(reducerightarcp); 
          if (k == sent.size()) 
            beam_chart[i-1].back()->add_importance_weight(reducep); 
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
      beam_chart.insert(beam_chart.begin(), AsParserList());
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
    return ArcStandardParser(static_cast<TaggedSentence>(sent));  
  } else
    return ArcStandardParser(*beam_chart[n][0]); 
}

template<class ParsedWeights>
ArcStandardParser ArcStandardParseModel<ParsedWeights>::particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles,
        bool resample) {
    //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states

  AsParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles))); 

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
      Real reduceleftarcp = weights->predictAction(static_cast<WordId>(kAction::la), r_ctx);
      Real reducerightarcp = weights->predictAction(static_cast<WordId>(kAction::ra), r_ctx);
      Real reducep = neg_log_sum_exp(reduceleftarcp, reducerightarcp); 

      std::vector<int> sample_counts = {0, 0, 0}; //shift, reduceleftarc, reducerightarc

      if (beam_stack[j]->stack_depth() < 2) {
        //only shift is allowed
        sample_counts[0] += num_samples;
      } else {
        if (beam_stack[j]->stack_depth() == 2) {
          //left arc disallowed
          reduceleftarcp = L_MAX;
          reducerightarcp = reducep;
        }

        std::vector<Real> distr = {shiftp, reduceleftarcp, reducerightarcp};
        multinomial_distribution_log<Real> mult(distr); 
        for (int k = 0; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }
      }        
     
      if ((sample_counts[1] > 0) && (sample_counts[2] > 0)) {
        beam_stack.push_back(boost::make_shared<ArcStandardParser>(*beam_stack[j]));
        beam_stack.push_back(boost::make_shared<ArcStandardParser>(*beam_stack[j]));

        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp);
        beam_stack.back()->set_num_particles(sample_counts[1]); 

        beam_stack.rbegin()[1]->rightArc();
        beam_stack.rbegin()[1]->add_particle_weight(reducerightarcp); 
        beam_stack.rbegin()[1]->set_num_particles(sample_counts[2]); 

      } else if (sample_counts[2] > 0) {
        beam_stack.push_back(boost::make_shared<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->rightArc();
        beam_stack.back()->add_particle_weight(reducerightarcp); 
        beam_stack.back()->set_num_particles(sample_counts[2]); 
      } else if (sample_counts[1] > 0) {
        beam_stack.push_back(boost::make_shared<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
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
      //sort and remove 
      std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
      for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
        beam_stack.pop_back();
      //std::cout << "Beam size: " << beam_stack.size();
      resampleParticles(&beam_stack, eng, num_particles);
      /* int active_particle_count = 0;
       for (int j = 0; j < beam_stack.size(); ++j)
        if (beam_stack[j]->num_particles() > 0)
         ++active_particle_count;
      std::cout << " -> " << active_particle_count << " without null \n"; */
    }
  }
     
  ///completion
  AsParserList final_beam; 
  bool has_more_states = true;

  while (has_more_states) {
    has_more_states = false;
    unsigned cur_beam_size = beam_stack.size();

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->inTerminalConfiguration()) {
        //add paths for reduce actions
        has_more_states = true; 
        Words r_ctx = beam_stack[j]->actionContext();
        Real reduceleftarcp = weights->predictAction(static_cast<WordId>(kAction::la), r_ctx);
        Real reducerightarcp = weights->predictAction(static_cast<WordId>(kAction::ra), r_ctx);
        Real reducep = neg_log_sum_exp(reduceleftarcp, reducerightarcp); 
        
        int num_samples = beam_stack[j]->num_particles();
        std::vector<int> sample_counts = {0, 0}; //reduceleftarc, reducerightarc

        if (beam_stack[j]->stack_depth() == 2) {
          //only allow right arc
          sample_counts[1] = num_samples;
        } else {
          std::vector<Real> distr = {reduceleftarcp, reducerightarcp};
          multinomial_distribution_log<Real> mult(distr); 
          for (int k = 0; k < num_samples; k++) {
            WordId act = mult(eng);
            ++sample_counts[act];
          }
        }

        if ((sample_counts[0] > 0) && (sample_counts[1] > 0)) {
          beam_stack.push_back(boost::make_shared<ArcStandardParser>(*beam_stack[j]));

          beam_stack.back()->leftArc();
          beam_stack.back()->add_particle_weight(reduceleftarcp);
          beam_stack.back()->set_num_particles(sample_counts[0]); 
          beam_stack.back()->add_importance_weight(reducep); 

          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->set_num_particles(sample_counts[1]); 
          beam_stack[j]->add_importance_weight(reducep); 
          
        } else if (sample_counts[1] > 0) {
          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->set_num_particles(sample_counts[1]); 
          beam_stack[j]->add_importance_weight(reducep); 

        } else if (sample_counts[0] > 0) {
          beam_stack[j]->leftArc();
          beam_stack[j]->add_particle_weight(reduceleftarcp); 
          beam_stack[j]->set_num_particles(sample_counts[0]); 
          beam_stack[j]->add_importance_weight(reducep); 
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
  }

  //alternatively, sort according to particle weight 
  //std::sort(final_beam.begin(), final_beam.end(), cmp_particle_ptr_weights); //handle pointers
 
  std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_weighted_importance_weights); 
  for (int j = beam_stack.size()- 1; ((j >= 0) && (beam_stack[j]->num_particles() == 0)); --j)
    beam_stack.pop_back();
  //std::cout << "Final beam size: " << beam_stack.size();

  /* if ((beam_stack.size() > 0) && take_max) {
    //resampleParticles(&beam_stack, eng, num_particles);
    std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
    return ArcStandardParser(*beam_stack[0]);
  } */
  if (beam_stack.size() > 0) {
    //just take 1 sample
    resampleParticles(&beam_stack, eng, 1);
    for (unsigned i = 0; i < beam_stack.size(); ++i) {
      if (beam_stack[i]->num_particles() == 1) {
        //beam_stack[i]->print_arcs();
        //float dir_acc = (beam_stack[i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
        //std::cout << "  Dir Accuracy: " << dir_acc;
        //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;
        return ArcStandardParser(*beam_stack[i]); 
      }
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcStandardParser(static_cast<TaggedSentence>(sent));  
}

//sample a derivation for the gold parse, given the current model
//three-way decisions
template<class ParsedWeights>
ArcStandardParser ArcStandardParseModel<ParsedWeights>::particleGoldParseSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, MT19937& eng, unsigned num_particles, bool resample) {
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states
  
  AsParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcStandardParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles))); 

  //shift ROOT symbol (probability 1)
  beam_stack[0]->shift(); 

  for (unsigned i = 1; i < sent.size(); ++i) {
    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if (beam_stack[j]->num_particles()==0)
        continue;
       
      //sample a sequence of possible actions leading up to the next shift

      int num_samples = beam_stack[j]->num_particles();
      std::vector<int> sample_counts = {0, 0, 0}; //shift, reduceleftarc, reducerightarc

      Words r_ctx = beam_stack[j]->actionContext();
      Real shiftp = weights->predictAction(static_cast<WordId>(kAction::sh), r_ctx);
      Real reduceleftarcp = weights->predictAction(static_cast<WordId>(kAction::la), r_ctx);
      Real reducerightarcp = weights->predictAction(static_cast<WordId>(kAction::ra), r_ctx);
      Real reducep = neg_log_sum_exp(reduceleftarcp, reducerightarcp); 
      
      kAction oracle_next = beam_stack[j]->oracleNext(sent);

      if (oracle_next==kAction::sh) {
        //only shift is allowed
        sample_counts[0] += num_samples;
        if (beam_stack[j]->stack_depth() >= 2)
          beam_stack[j]->add_importance_weight(shiftp);  
      } else {
        //enforce at least one particle to reduce
        std::vector<Real> distr; //= {shiftp, reduceleftarcp, reducerightarcp};
        if (oracle_next==kAction::la) {
          distr = {shiftp, reducep, L_MAX};
          sample_counts[1] =  1;            
        }
        if (oracle_next==kAction::ra) {
          distr = {shiftp, L_MAX, reducep};
          sample_counts[2] =  1;            
        }

        multinomial_distribution_log<Real> mult(distr); 
        for (int k = 1; k < num_samples; k++) {
          WordId act = mult(eng);
          ++sample_counts[act];
        }
      }
      
     if (sample_counts[2] > 0) {
        beam_stack.push_back(boost::make_shared<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->rightArc();
        beam_stack.back()->add_particle_weight(reducerightarcp); 
        beam_stack.back()->add_importance_weight(reducep - reducerightarcp); 
        beam_stack.back()->set_num_particles(sample_counts[2]); 
      } else if (sample_counts[1] > 0) {
        beam_stack.push_back(boost::make_shared<ArcStandardParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(reduceleftarcp); 
        beam_stack.back()->add_importance_weight(reducep - reduceleftarcp); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      //perform shift if > 0 samples
      if (sample_counts[0] == 0)
        beam_stack[j]->set_num_particles(0);
      else {
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), beam_stack[j]->tagContext());
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

  while (has_more_states) {
    has_more_states = false;
    //unsigned cur_beam_size = beam_stack.size();
    //std::cerr << cur_beam_size << ": ";

    for (unsigned j = 0; j < beam_stack.size(); ++j) { 
      if ((beam_stack[j]->num_particles() > 0) && !beam_stack[j]->inTerminalConfiguration()) {
        //add paths for reduce actions
        has_more_states = true; 
        Words r_ctx = beam_stack[j]->actionContext();
        Real reduceleftarcp = weights->predictAction(static_cast<WordId>(kAction::la), r_ctx);
        Real reducerightarcp = weights->predictAction(static_cast<WordId>(kAction::ra), r_ctx);
        
        kAction oracle_next = beam_stack[j]->oracleNext(sent);
        //std::cerr << " (" << beam_stack[j]->num_particles() << ") " << static_cast<WordId>(oracle_next);
        if (oracle_next==kAction::re) {
          //invalid, so let particles die (else all particles are moved on)
          beam_stack[j]->set_num_particles(0);
        } else if (oracle_next == kAction::ra) {
          beam_stack[j]->rightArc();
          beam_stack[j]->add_particle_weight(reducerightarcp); 
          beam_stack[j]->add_importance_weight(reducerightarcp); 
        } else if (oracle_next == kAction::la) {
          beam_stack[j]->leftArc();
          beam_stack[j]->add_particle_weight(reduceleftarcp); 
          beam_stack[j]->add_importance_weight(reduceleftarcp); 
        }
      }
    }
    //std::cerr << std::endl;

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
      //float dir_acc = (beam_stack[i]->directed_accuracy_count(sent) + 0.0)/(sent.size()-1);
      //std::cout << "  Dir Accuracy: " << dir_acc;
      //std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;

      return ArcStandardParser(*beam_stack[i]); 
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcStandardParser(static_cast<TaggedSentence>(sent));  
}


template<class ParsedWeights>
ArcStandardParser ArcStandardParseModel<ParsedWeights>::staticGoldParseSentence(const ParsedSentence& sent, 
                                    const boost::shared_ptr<ParsedWeights>& weights) {
  ArcStandardParser parser(static_cast<TaggedSentence>(sent));
  
  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && (a != kAction::re)) {
    a = parser.oracleNext(sent);  
    if (a != kAction::re) {
      //update particle weight
      Real actionp = weights->predictAction(static_cast<WordId>(a), parser.actionContext());
      parser.add_particle_weight(actionp);

      if (a == kAction::sh) {
        Real tagp = weights->predictTag(parser.next_tag(), parser.tagContext());
        Real wordp = weights->predictWord(parser.next_word(), parser.wordContext());
        parser.add_particle_weight(tagp);
        parser.add_particle_weight(wordp);
      }

      parser.executeAction(a);
    } 
  }

  return parser;
}
    
template<class ParsedWeights>
ArcStandardParser ArcStandardParseModel<ParsedWeights>::staticGoldParseSentence(const ParsedSentence& sent) {
  ArcStandardParser parser(static_cast<TaggedSentence>(sent));
  
  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && (a != kAction::re)) {
    a = parser.oracleNext(sent);  
    if (a != kAction::re) 
      parser.executeAction(a);
  }

  return parser;
}

//generate a sentence: ternary decisions
template<class ParsedWeights>
ArcStandardParser ArcStandardParseModel<ParsedWeights>::generateSentence(const boost::shared_ptr<ParsedWeights>& weights, 
        MT19937& eng) {
  unsigned sent_limit = 100;
  ArcStandardParser parser;
  bool terminate_shift = false;
  parser.push_tag(0);
  parser.shift(0);
    
  do {
    kAction a = kAction::sh; //placeholder action
    //std::cerr << "arcs: " << parser.arcs().size() << std::endl;
    
    if (parser.stack_depth() < 2) {
      a = kAction::sh;
    } else if (parser.size() >= sent_limit) {
        // check to upper bound sentence length
        //if (!terminate_shift)
        //  std::cout << " LENGTH LIMITED ";
        terminate_shift = true;
        a = kAction::re;
    } else {
      Words r_ctx = parser.actionContext();
      Real shiftp = weights->predictAction(static_cast<WordId>(kAction::sh), r_ctx);
      Real leftarcreducep = weights->predictAction(static_cast<WordId>(kAction::la), r_ctx);
      Real rightarcreducep = weights->predictAction(static_cast<WordId>(kAction::ra), r_ctx);

      if (parser.stack_depth() == 2)
        leftarcreducep = L_MAX;

      //sample an action
      Reals distr = {shiftp, leftarcreducep, rightarcreducep};
      multinomial_distribution_log<Real> mult(distr); 
      WordId act = mult(eng);
      //std::cout << "(" << parser.stack_depth() << ") ";
      //std::cout << act << " ";
      parser.add_particle_weight(distr[act]);
      
      if (act==0) {
        a = kAction::sh;
      } else if (act==1) {
        a = kAction::la; 
        parser.leftArc();
      } else {
        a = kAction::ra;
        parser.rightArc();
      }
    } 

    if (a == kAction::sh) {
      //sample a tag - disallow root tag
      Words t_ctx = parser.tagContext();
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
      parser.shift(word);
      parser.add_particle_weight(wordp);
    }
  } while (!parser.inTerminalConfiguration() && !terminate_shift);
  //std::cout << std::endl;

  //std::cout << std::endl;
  return parser;
}

template<class ParsedWeights>
void ArcStandardParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcStandardParser parse = staticGoldParseSentence(sent); 
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcStandardParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  unsigned beam_size = 8;
  //ArcStandardParser parse = staticGoldParseSentence(sent, weights);
  ArcStandardParser parse = beamParseSentence(sent, weights, beam_size);
  //std::cout << "Gold actions: ";
  //parse.print_actions();
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcStandardParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng,
          const boost::shared_ptr<ParseDataSet>& examples) {
  //unsigned num_particles = 100;
  //bool resample = true;

  ArcStandardParser parse = particleGoldParseSentence(sent, weights, eng, config_->num_particles, config_->resample);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcStandardParseModel<ParsedWeights>::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng,
          const boost::shared_ptr<ParseDataSet>& examples) {
  //unsigned num_particles = 100;
  //bool resample = true;

  ArcStandardParser parse = particleParseSentence(sent, weights, eng, config_->num_particles, config_->resample);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void ArcStandardParseModel<ParsedWeights>::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  unsigned beam_size = 8;
  ArcStandardParser parse = beamParseSentence(sent, weights, beam_size);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
Parser ArcStandardParseModel<ParsedWeights>::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) {
  Words ctx(7, 0);
  ArcStandardParser parse;
  if (beam_size == 0)
    parse = greedyParseSentence(sent, weights);
  else
    parse = beamParseSentence(sent, weights, beam_size);
  acc_counts->countAccuracy(parse, sent);
  ArcStandardParser gold_parse = staticGoldParseSentence(sent, weights);
  
  acc_counts->countLikelihood(parse.weight(), gold_parse.weight());
  return parse;
}

template<class ParsedWeights>
Parser ArcStandardParseModel<ParsedWeights>::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) {
  bool resample = true;

  //eval with particle parse
  ArcStandardParser parse = particleParseSentence(sent, weights, eng, beam_size, resample);
  acc_counts->countAccuracy(parse, sent);
  ArcStandardParser gold_parse = staticGoldParseSentence(sent, weights);
  
  acc_counts->countLikelihood(parse.weight(), gold_parse.weight());
  return parse;
}

template class ArcStandardParseModel<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardParseModel<ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardParseModel<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>;
template class ArcStandardParseModel<ParsedFactoredWeights>;
template class ArcStandardParseModel<ParsedWeights>;

}

