#include "gdp/arc_eager_parse_model.h"

namespace oxlm {

void ArcEagerParseModel::resampleParticles(AeParserList* beam_stack, MT19937& eng,
        unsigned num_particles) {
  //assume (beam_stack->at(pi)->num_particles() > 0)
  std::vector<double> importance_w(beam_stack->size(), L_MAX); 
  for (unsigned i = 0; i < importance_w.size(); ++i) 
    importance_w[i] = beam_stack->at(i)->weighted_importance_weight();

  //resample according to importance weight
  multinomial_distribution_log part_mult(importance_w); 
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

ArcEagerParser ArcEagerParseModel::beamParseSentence(const ParsedSentence& sent, 
                           const boost::shared_ptr<ParsedWeightsInterface>& weights, unsigned beam_size) {
  //index in beam_chart is depth-of-stack - 1
  std::vector<AeParserList> beam_chart; 
  beam_chart.push_back(AeParserList());
  beam_chart[0].push_back(boost::make_shared<ArcEagerParser>(static_cast<TaggedSentence>(sent))); 

  //shift ROOT symbol (probability 1)
  beam_chart[0][0]->shift(); 

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 1; k < sent.size(); ++k) {
    //there are k beam lists. perform reduces down to list 1
    for (unsigned i = k - 1; i > 0; --i) { 
      //prune if size exceeds beam_size
      if (beam_chart[i].size() > beam_size) {
        std::sort(beam_chart[i].begin(), beam_chart[i].end(), TransitionParser::cmp_particle_weights); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_chart[i].size(); j > beam_size; --j)
          beam_chart[i].pop_back();
      }

      //std::cout << "reduce list size: " << beam_chart[i].size() << std::endl;
      //consider reduce and left arc actions
      //for every item in the list, add valid reduce actions to list i - 1 
      for (unsigned j = 0; (j < beam_chart[i].size()); ++j) {
        Real leftarcreducep = weights->predictAction(static_cast<WordId>(kAction::la), 
                                                       beam_chart[i][j]->actionContext());
        Real reducep = weights->predictAction(static_cast<WordId>(kAction::re), 
                                                       beam_chart[i][j]->actionContext());
        //std::cout << "(la: " << leftarcreducep << ", re: " << reducep << ") ";
        //Real reducetotalp = neg_log_sum_exp(leftarcreducep, reducep);
       
        //actually also need importance weight if either is invalid
        //left arc invalid also after last shift
        if (beam_chart[i][j]->left_arc_valid()) { 
          beam_chart[i-1].push_back(boost::make_shared<ArcEagerParser>(*beam_chart[i][j]));
          beam_chart[i-1].back()->leftArc();
          beam_chart[i-1].back()->add_particle_weight(leftarcreducep);
        } 
        //else //TODO temp experiment - basically no difference
        if (beam_chart[i][j]->reduce_valid()) {          
          beam_chart[i-1].push_back(boost::make_shared<ArcEagerParser>(*beam_chart[i][j]));
          beam_chart[i-1].back()->reduce();
          beam_chart[i-1].back()->add_particle_weight(reducep); 
          
        }
      }
      //std::cout << std::endl;
    }

    if (beam_chart[0].size() > beam_size) {
        std::sort(beam_chart[0].begin(), beam_chart[0].end(), TransitionParser::cmp_particle_weights); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_chart[0].size(); j > beam_size; --j)
          beam_chart[0].pop_back();
    }

    //perform shifts: shift or right arc
    for (unsigned i = 0; (i < k); ++i) { 
      unsigned list_size = beam_chart[i].size();
      for (unsigned j = 0; j < list_size; ++j) {
          
        Real shiftp = weights->predictAction(static_cast<WordId>(kAction::sh), 
                                              beam_chart[i][j]->actionContext());
        Real rightarcshiftp = weights->predictAction(static_cast<WordId>(kAction::ra), 
                                              beam_chart[i][j]->actionContext());
        //Real shifttotalp = neg_log_sum_exp(shiftp, rightarcshiftp);
        //std::cout << "(sh: " << shiftp << ", ra: " << rightarcshiftp << ") ";

        //ra not valid for stop symbol
        if (k < (sent.size() - 1)) {
          Real tagp = weights->predictTag(beam_chart[i][j]->next_tag(), 
                                           beam_chart[i][j]->tagContext(kAction::ra));
          Real wordp = weights->predictWord(beam_chart[i][j]->next_word(), 
                                             beam_chart[i][j]->wordContext());
         
          beam_chart[i].push_back(boost::make_shared<ArcEagerParser>(*beam_chart[i][j]));
          beam_chart[i].back()->rightArc();
          beam_chart[i].back()->add_particle_weight(rightarcshiftp);
          beam_chart[i].back()->add_importance_weight(tagp); 
          beam_chart[i].back()->add_importance_weight(wordp); 
          beam_chart[i].back()->add_particle_weight(tagp); 
          beam_chart[i].back()->add_particle_weight(wordp); 
        }

        //shift is valid
        Real tagp = weights->predictTag(beam_chart[i][j]->next_tag(), 
                                           beam_chart[i][j]->tagContext(kAction::sh));
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
    beam_chart.insert(beam_chart.begin(), AeParserList());
       
    //std::cout << std::endl;
  }
 
  //completion: reduce after last shift
  //std::cout << "completion" << std::endl;
  for (unsigned i = beam_chart.size() - 1; i > 0; --i) {  //sent.size()
    //prune if size exceeds beam_size
    if (beam_chart[i].size() > beam_size) {
      std::sort(beam_chart[i].begin(), beam_chart[i].end(), ArcEagerParser::cmp_reduce_particle_weights); //handle pointers
      //remove items with worst scores, and those that cannot reduce
      for (unsigned j = beam_chart[i].size() - 1; ((j >= beam_size) || ((j > 0) && !beam_chart[i][j]->reduce_valid())); --j)
        beam_chart[i].pop_back();
    }

    //std::cout << i <<  " reduce list size: " << beam_chart[i].size() << std::endl;
    //consider reduce and left arc actions
    //for every item in the list, add valid reduce actions to list i - 1 
    for (unsigned j = 0; (j < beam_chart[i].size()); ++j) {
      Real reducep = weights->predictAction(static_cast<WordId>(kAction::re), 
                                             beam_chart[i][j]->actionContext());
                
      if (beam_chart[i][j]->reduce_valid()) {  
        beam_chart[i-1].push_back(boost::make_shared<ArcEagerParser>(*beam_chart[i][j]));
        beam_chart[i-1].back()->reduce();
        //in some models, don't add the weights TODO
        //beam_chart[i-1].back()->add_particle_weight(reducep); 
        //beam_chart[i-1].back()->add_importance_weight(reducep); 
        //std::cout << j << " re valid ";
      }
    }
    //std::cout << std::endl;
  }

  std::sort(beam_chart[0].begin(), beam_chart[0].end(), TransitionParser::cmp_particle_weights); //handle pointers

  //print parses
  unsigned const n = 0; 
  //std::cout << "Beam size: " << beam_chart[n].size() << std::endl;
  for (unsigned i = 0; (i < beam_chart[n].size()); ++i) 
    beam_chart[n][0]->add_beam_weight(beam_chart[n][i]->particle_weight());

  for (unsigned i = 0; (i < 5) && (i < beam_chart[n].size()); ++i) {
    std::cout << beam_chart[n][i]->particle_weight() << " ";
    beam_chart[n][i]->print_arcs();
    beam_chart[n][i]->print_actions();

    //float dir_acc = (beam_chart[n][i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
    //std::cout << "  Dir Accuracy: " << dir_acc;
    //std::cout << "  Sample weight: " << (beam_chart[n][i]->particle_weight()) << std::endl;
  }  

  if (beam_chart[n].size()==0) {
    //std::cout << "no parse found" << std::endl;
  return ArcEagerParser(static_cast<TaggedSentence>(sent));  
  } else {
    //beam_chart[n][0]->print_arcs();
    //beam_chart[n][0]->print_actions();
    return ArcEagerParser(*beam_chart[n][0]); 
  }
}

ArcEagerParser ArcEagerParseModel::particleParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng, unsigned num_particles,
        bool resample) {
//TODO
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states

  AeParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcEagerParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles))); 

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

      std::vector<double> distr = {shiftp, leftarcreducep, rightarcshiftp, reducep};
      multinomial_distribution_log mult(distr); 
      for (int k = 0; k < num_samples; k++) {
        WordId act = mult(eng);
        ++sample_counts[act];
      }
      
      //for reduce actions: 
      if (sample_counts[1] > 0) {
        beam_stack.push_back(boost::make_shared<ArcEagerParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(leftarcreducep); 
        beam_stack.back()->set_num_particles(sample_counts[1]); 
      }

      if (sample_counts[3] > 0) {
        beam_stack.push_back(boost::make_shared<ArcEagerParser>(*beam_stack[j]));
        beam_stack.back()->reduce();
        beam_stack.back()->add_particle_weight(reducep); 
        beam_stack.back()->set_num_particles(sample_counts[3]); 
      }

      //for shift actions
      if (sample_counts[2] > 0) {
        beam_stack.push_back(boost::make_shared<ArcEagerParser>(*beam_stack[j]));

        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), 
                    beam_stack[j]->tagContext(kAction::ra));
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());

        beam_stack.back()->rightArc();
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


  /* if ((beam_stack.size() > 0) && take_max) {
    //resampleParticles(&beam_stack, eng, num_particles);
    std::sort(beam_stack.begin(), beam_stack.end(), TransitionParser::cmp_particle_weights); 
    return ArcEagerParser(*beam_stack[0]); 
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
        return ArcEagerParser(*beam_stack[i]); 
      }
    }
  }

  //std::cout << "no parse found" << std::endl;
  return ArcEagerParser(static_cast<TaggedSentence>(sent));  
}

//4-way decisions
ArcEagerParser ArcEagerParseModel::particleGoldParseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeightsInterface>& weights, MT19937& eng, 
          unsigned num_particles, bool resample) {
  //TODO
  //Follow approach similar to per-word beam-search, but also keep track of number of particles that is equal to given state
  //perform sampling and resampling to update these counts, and remove 0 count states

  AeParserList beam_stack; 
  beam_stack.push_back(boost::make_shared<ArcEagerParser>(static_cast<TaggedSentence>(sent), static_cast<int>(num_particles))); 

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
        beam_stack.push_back(boost::make_shared<ArcEagerParser>(*beam_stack[j]));
        beam_stack.back()->leftArc();
        beam_stack.back()->add_particle_weight(leftarcreducep); 
        beam_stack.back()->add_importance_weight(leftarcreducep); 
        beam_stack[j]->set_num_particles(0);
      } else if (oracle_next==kAction::ra) {
        Real tagp = weights->predictTag(beam_stack[j]->next_tag(), 
                beam_stack[j]->tagContext(kAction::ra));
        Real wordp = weights->predictWord(beam_stack[j]->next_word(), beam_stack[j]->wordContext());
        
        beam_stack[j]->rightArc();
        beam_stack[j]->add_importance_weight(wordp); 
        beam_stack[j]->add_importance_weight(tagp); 
        beam_stack[j]->add_particle_weight(rightarcshiftp); 
        beam_stack[j]->add_particle_weight(wordp); 
        beam_stack[j]->add_particle_weight(tagp); 
      } else {
        if (oracle_next==kAction::re) {
          //enforce at least 1 particle to reduce
          std::vector<int> sample_counts = {0, 1}; //shift, reduce
          std::vector<double> distr = {shiftp, reducep};
          multinomial_distribution_log mult(distr); 
          for (int k = 1; k < num_samples; k++) {
            WordId act = mult(eng);
            ++sample_counts[act];
          }

          if (sample_counts[1] > 0) {
            beam_stack.push_back(boost::make_shared<ArcEagerParser>(*beam_stack[j]));
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
      std::cout << "  Sample weight: " << (beam_stack[i]->particle_weight()) << std::endl;

      return ArcEagerParser(*beam_stack[i]); 
    }
  }

  std::cout << "no parse found" << std::endl;
  return ArcEagerParser(static_cast<TaggedSentence>(sent));  
}

ArcEagerParser ArcEagerParseModel::staticGoldParseSentence(const ParsedSentence& sent,
                                        const boost::shared_ptr<ParsedWeightsInterface>& weights) {
  ArcEagerParser parser(static_cast<TaggedSentence>(sent));

  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && !(parser.buffer_empty() && (a == kAction::sh))) {
    a = parser.oracleNext(sent);
    if (!(parser.buffer_empty() && (a == kAction::sh))) {
      //update particle weight
      Real actionp = weights->predictAction(static_cast<WordId>(a), parser.actionContext());
      parser.add_particle_weight(actionp);

      if (a == kAction::sh || a == kAction::ra) {
        Real tagp = weights->predictTag(parser.next_tag(), parser.tagContext(a));
        Real wordp = weights->predictWord(parser.next_word(), parser.wordContext());
        parser.add_particle_weight(tagp);
        parser.add_particle_weight(wordp);
      }
      
      parser.executeAction(a); 
    } 
  }

  return parser;
}

ArcEagerParser ArcEagerParseModel::staticGoldParseSentence(const ParsedSentence& sent) {
  ArcEagerParser parser(static_cast<TaggedSentence>(sent));

  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && !(parser.buffer_empty() && (a == kAction::sh))) {
    a = parser.oracleNext(sent);
    if (!(parser.buffer_empty() && (a == kAction::sh))) 
      parser.executeAction(a); 
  }

  return parser;
}

ArcEagerParser ArcEagerParseModel::generateSentence(const boost::shared_ptr<ParsedWeightsInterface>& weights, 
        MT19937& eng) {
  ArcEagerParser parser;
  unsigned sent_limit = 100;
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
      std::vector<double> distr = {shiftp, leftarcreducep, rightarcshiftp, reducep};
      multinomial_distribution_log mult(distr); 
      WordId act = mult(eng);
      //std::cout << "(" << parser.stack_depth() << ") ";
      //std::cout << act << " ";
      parser.add_particle_weight(distr[act]);
      
      if (act==0) {
        a = kAction::sh;
      } else if (act==1) {
        a = kAction::la; 
        parser.leftArc();
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
      std::vector<double> t_distr(weights->numTags() - 1, L_MAX);
      for (WordId w = 1; w < weights->numTags(); ++w) 
        t_distr[w-1] = weights->predictTag(w, t_ctx); 
      multinomial_distribution_log t_mult(t_distr);
      WordId tag = t_mult(eng) + 1;

      Real tagp = weights->predictTag(tag, t_ctx); 
      parser.push_tag(tag);
      parser.add_particle_weight(tagp);

      //sample a word 
      Words w_ctx = parser.wordContext();
      std::vector<double> w_distr(weights->numWords(), 0);

      w_distr[0] = weights->predictWord(-1, w_ctx); //unk probability
      for (WordId w = 1; w < weights->numWords(); ++w) 
        w_distr[w] = weights->predictWord(w, w_ctx); 
      multinomial_distribution_log w_mult(w_distr);
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

  return parser;
}

void ArcEagerParseModel::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcEagerParser parse = staticGoldParseSentence(sent); 
  parse.extractExamples(examples);
}

void ArcEagerParseModel::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  ArcEagerParser parse = staticGoldParseSentence(sent, weights);
  parse.extractExamples(examples);
}

void ArcEagerParseModel::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          MT19937& eng,
          const boost::shared_ptr<ParseDataSet>& examples) {
  unsigned num_particles = 1000;
  bool resample = false;

  ArcEagerParser parse = particleParseSentence(sent, weights, eng, num_particles, resample);
  parse.extractExamples(examples);
}

Real ArcEagerParseModel::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeightsInterface>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) {
  ArcEagerParser parse = beamParseSentence(sent, weights, beam_size);
  acc_counts->countAccuracy(parse, sent);
  ArcEagerParser gold_parse = staticGoldParseSentence(sent, weights);
  
  acc_counts->countLikelihood(parse.weight(), gold_parse.weight());
  return parse.particle_weight();
}

}

