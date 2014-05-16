#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "transition_parser.h"
#include "hpyp_dp_train.h"
#include "hpyplm/hpyplm.h"
#include "corpus/corpus.h"
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

#define kORDER 3  //default 4
#define nPARTICLES 100

using namespace std;
using namespace oxlm;

void resample_particles(vector<ArcStandardParser>& particles, unsigned num_particles, MT19937& eng) {
  vector<double> importance_w(num_particles, 0); //importance weights
  for (unsigned i = 0; i < num_particles; ++i)
    importance_w[i] = particles[i].importance_weight();

  //resample according to importance weight
  multinomial_distribution_log part_mult(importance_w); 
  vector<unsigned> sample_indx;
  //cout << "resampled importance weights: ";
  for (unsigned i = 0; i < num_particles; ++i) {
    unsigned pi = part_mult(eng);
    //cout << (-particle_liw[pi] / log(10)) << " ";
    sample_indx.push_back(pi);
  }
  //cout << endl;
  sort(sample_indx.begin(), sample_indx.end());

  //cout << "sorted resampled indexes: ";
  //for (unsigned i = 0; i < num_particles; ++i) {
  //  cout << sample_indx[i] << " ";
  //}
  //cout << endl;

  vector<bool> occur_indx; //-1 if not in sample, else its position 
  unsigned s_ind = 0; //index in sample_indx
  for (unsigned i = 0; i < num_particles; ++i) { //over original particles
    //cout << s_ind << " ";
    if ((s_ind >= num_particles) || (i < sample_indx[s_ind]))
      occur_indx.push_back(false);
    else if (i == sample_indx[s_ind]) {
      occur_indx.push_back(true); 
      while ((s_ind < num_particles - 1) && (sample_indx[s_ind]==sample_indx[s_ind+1])) ++s_ind;
      ++s_ind;
    } else {
      cerr << "contradiction (" << i << ") " << endl;     
    }
  }

  vector<int> new_indx(num_particles, -1); //just for the record
  //cout << "occur indexes: ";
  for (unsigned i = 0; i < num_particles; ++i) {
    //cout << occur_indx[i] << " ";
    new_indx[i] = i;
  }
  //cout << endl;
  
  unsigned p_pos = 0; //over the particles that may be replaced
  for (unsigned i = 1; i < num_particles; ++i) { //over sorted indexes
    if (sample_indx[i]==sample_indx[i-1]) {
      while ((p_pos < num_particles) && (occur_indx[p_pos])) ++p_pos; //not supposed to go out of index 
      //actual (copy) reassignment
      if (p_pos < num_particles) {
        //cout << p_pos << " ";
        particles[p_pos] = particles[sample_indx[i]]; //key line: copy particle      
        new_indx[p_pos] = sample_indx[i];
        //cout << "(" << i << ", " << sample_indx[i] << ") ";
        ++p_pos;
      } else
        cout << "invalid index ";
      //cout << endl;
    } 
  }

  //cout << "replaced resampled importance weights: ";
  //for (unsigned i = 0; i < num_particles; ++i) {
  //  cout << (-particle_liw[i] / log(10)) << " ";
  //}
  //cout << endl;
  
  //cout << "resampled weights: ";
  //for (unsigned i = 0; i < num_particles; ++i) {
  //  cout << particles[i].importance_weight() << " ";
  //}
  //cout << endl; 
}

//for binary action decisions, resample at each step
ArcStandardParser particle_par_parse_sentence(Words s, WxList goldd, unsigned num_particles, bool resample, Dict& dict, MT19937& eng, set<WordId>& vocabs, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& reduce_lm, PYPLM<kORDER>& arc_lm) {  
  cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl; 

  vector<ArcStandardParser> particles(num_particles, ArcStandardParser(s, kORDER-1)); 

  for (unsigned i = 0; i < s.size() + 1; ++i) {
    for (unsigned j = 0; j < num_particles; ++j) { 
      ArcStandardParser& parser = particles[j];
      parser.reset_importance_weight();

      Action a = Action::re; //placeholder action
      //sample a sequence of possible actions leading up to the next shift
      while (a == Action::re) {
        Words ctx = parser.word_context();
    
        if ((parser.stack_depth()< kORDER-1) && !parser.is_buffer_empty()) 
          a = Action::sh;
        else {
          //shift or reduce
          //for (auto& c: ctx)
          //  cout << c << " ";
          //cout << ctx[0] << " " << ctx[1] << " ";
          double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), ctx);
          double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), ctx);
          //cout << "(sh: " << shiftp << " re: " << reducep << ") ";

          //sample an action
          vector<double> distr = {shiftp, reducep};
          multinomial_distribution<double> mult(distr); 
          WordId act = mult(eng);
          //cout << act << " ";
      
          if (act==0) {
            a = Action::sh;
            parser.add_particle_weight(shiftp);
          } else {
            a = Action::re; 
            parser.add_particle_weight(reducep);
          }
        }

        if (a == Action::re) {
          //for (auto& c: ctx)
          //  cout << c << " ";
          //cout << ctx[0] << " " << ctx[1] << " ";
          double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
          double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);
          //cout << "(la: " << leftarcp << " ra: " << rightarcp << ") ";

          //sample arc direction
          vector<double> distr = {leftarcp, rightarcp};
          multinomial_distribution<double> mult(distr); 
          WordId act = mult(eng);
          
          if (!parser.left_arc_valid()) {
            act = 1;
            parser.add_importance_weight(rightarcp);
          }

          if (act==0) {
            parser.left_arc(); 
            parser.add_particle_weight(leftarcp); 
          } else {
            parser.right_arc();
            parser.add_particle_weight(rightarcp); 
          }
        }
      }

      //perform shift
      Words ctx = parser.word_context();
      WordId w = parser.next_word();
      double wordp = shift_lm.prob(w, ctx); 
      //cout << "[" << wordp << "] ";
      parser.shift();
      parser.add_importance_weight(wordp); //else add_importance_weight
      parser.add_particle_weight(wordp); //else add_importance_weight
    }
    
    //resamples particles
    if ((i > 1) && resample) {
      resample_particles(particles, num_particles, eng);
      
      //cout << "particle weights: ";
      //for (unsigned i = 0; i < num_particles; ++i) 
      //  cout << particles[i].importance_weight() << " ";
      //cout << endl;
    }
    //cout << endl;
  }

  //cout << "start completion" << endl;
  for (unsigned j = 0; j < num_particles; ++j) { 
    ArcStandardParser& parser = particles[j];
    parser.reset_importance_weight();

    //std::cerr << parser.is_buffer_empty() << " " << parser.stack_depth() << std::endl;
    while (!parser.is_terminal_configuration()) {
        //sample arcs to complete the parse
      Words ctx = parser.word_context();
      //for (auto& c: ctx)
      //  cout << c << " ";
      //cout << ctx.at(ctx.size()-2) << " " << ctx.at(ctx.size()-1) << " ";
      double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), ctx);
      double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
      double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);
      //cout << "(la: " << leftarcp << " ra: " << rightarcp << ") ";

      //sample arc direction
      vector<double> distr = {leftarcp, rightarcp};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
  
      parser.add_importance_weight(reducep); 
      if (!parser.left_arc_valid()) {
        act = 1;
        parser.add_importance_weight(rightarcp);
      }

      if (act==0) {
        parser.left_arc(); 
        parser.add_particle_weight(leftarcp*reducep); 
      } else {
        parser.right_arc();
        parser.add_particle_weight(rightarcp*reducep); 
      }
      //cerr << "depth: " << parser.stack_depth() << endl;
    }
   // cout << endl;
  }

  if (resample)
    resample_particles(particles, num_particles, eng);
  sort(particles.begin(), particles.end(), cmp_particle_weights);

  for (unsigned j = 0; j < num_particles; ++j) { 
    ArcStandardParser& parser = particles[j];
    parser.print_arcs();
        
    float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/s.size();
    float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/s.size();

    cout << "  Dir Accuracy: " << dir_acc;
    cout << "  UnDir Accuracy: " << undir_acc;
    cout << "  Sample weight: " << (-parser.particle_weight() / log(10)) << endl;
  }

  return particles[0];
}

//beam parser
ArcStandardParser beam_parse_sentence(Words s, WxList goldd, unsigned beam_size, Dict& dict, MT19937& eng, set<WordId>& vocabs, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& reduce_lm, PYPLM<kORDER>& arc_lm) {  
  //double llh = 0;
  //unsigned cnt = 0; 
  //unsigned oovs = 0;
 
  vector<ArcStandardParser> beam(1, ArcStandardParser(s, kORDER-1));  
  beam[0].print_sentence(dict);
cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl;  

  //vector<ArcStandardParser> particles;

  for (unsigned i = 0; i < s.size() + 1; ++i) {
    //double m_w = 0; // worst weight so far on beam
    unsigned sh_bound = beam.size(); 
    //cerr << "sentence pos: " << i << endl;

    for (unsigned j = 0; j < beam.size(); ++j) { 
      //sample a sequence of possible actions leading up to the next shift
      Words ctx = beam[j].word_context();
      //if (j%10 == 0)
      //  cerr << j << " " << beam.size() << endl;

      if (beam[j].stack_depth() >= kORDER-1) {
        //add paths for reduce actions
        double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);

        //cerr << "p before: " <<  parser.stack_depth() << " " << parser.particle_weight() << " " << parser.buffer_length() << " " << parser.sentence_length() << endl;
       
        //add to the beam 
        beam.push_back(beam[j]);
        if (beam[j].left_arc_valid()) {
          beam.push_back(beam[j]);
          ArcStandardParser& right_p = beam.rbegin()[1]; 
          ArcStandardParser& left_p = beam.back(); 

          left_p.left_arc();
          left_p.add_particle_weight(reducep*leftarcp);
          left_p.set_importance_weight(reducep); 
          right_p.right_arc();
          right_p.add_particle_weight(reducep*rightarcp); 
          right_p.set_importance_weight(reducep); 

        } else {
          ArcStandardParser& right_p = beam.back(); 

          right_p.right_arc();
          right_p.add_particle_weight(reducep*rightarcp); 
          right_p.set_importance_weight(reducep*rightarcp); 
        }
      }

      //perform shift
      ArcStandardParser& shift_p = beam[j];
      double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), ctx);
      WordId w = shift_p.next_word();
      double wordlp = shift_lm.prob(w, ctx); 
      shift_p.shift();
      shift_p.set_importance_weight(wordlp); 
      shift_p.add_particle_weight(shiftp*wordlp); 
      //cerr << "p: " <<  parser.stack_depth() << " " << parser.particle_weight() << " " << parser.buffer_length() << " " << parser.sentence_length() << endl;

      if (j==sh_bound) {
        if (beam.size() > (beam_size + sh_bound)) {
         //cap unprocessed items at beam size
         sort(beam.begin() + (beam_size + sh_bound), beam.end(), cmp_particle_weights);
         for (unsigned j = beam.size()- 1; j >= (beam_size + sh_bound); --j)
           beam.pop_back();
        }

        sh_bound = beam.size();
      }
    }
   
    //TODO compare normalized weights 
    sort(beam.begin(), beam.end(), cmp_particle_weights);
    //remove items with worst scores
    for (unsigned j = beam.size()- 1; j >= beam_size; --j)
      beam.pop_back();
  }

  //cerr << "start completion" << endl;
  vector<ArcStandardParser> final_beam;  

  //for now, let beam just grow larger
  //double min_w = 0; // worst weight so far on beam 
  
  while (beam.size() > 0) {
    unsigned cur_beam_size = beam.size();

    for (unsigned j = 0; j < cur_beam_size; ++j) { 
      //cerr << beam[j].stack_depth() << " " << "(" << beam[j].particle_weight() << ")";
      if (beam[j].is_terminal_configuration()) {
        //has a complete parse
        final_beam.push_back(beam[j]);
        beam[j].set_log_particle_weight(1000); 
      } else {
        //add paths for reduce actions
        Words ctx = beam[j].word_context();
        double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);

        if (beam[j].left_arc_valid()) {
          beam.push_back(beam[j]);
          ArcStandardParser& right_p = beam[j];
          ArcStandardParser& left_p = beam.back(); 

          left_p.left_arc();
          left_p.add_particle_weight(reducep*leftarcp);
          left_p.set_importance_weight(reducep); 
          right_p.right_arc();
          right_p.add_particle_weight(reducep*rightarcp); 
          right_p.set_importance_weight(reducep); 
        } else {
          ArcStandardParser& right_p = beam[j];

          right_p.right_arc();
          right_p.add_particle_weight(reducep*rightarcp); 
          right_p.set_importance_weight(reducep*rightarcp); 
        }
      }  
    }
    //cerr << endl;

    //cerr << "beam size " << cur_beam_size << " " << beam.size() << endl;
    //cerr << "beam size " << beam.size() << " final " <<  final_beam.size() << endl;
    //TODO compare normalized weights 
    sort(beam.begin(), beam.end(), cmp_particle_weights);
    //remove items with worst scores
    for (int j = beam.size()- 1; ((j >= static_cast<int>(beam_size)) || ((j >= 0) && (beam[j].particle_weight() >= 1000))); --j) {
      //cerr << j << " " << beam.size() << endl;
      beam.pop_back();
    }

    if ((final_beam.size() > 2*beam_size) || (beam.size()==0)) {
      sort(final_beam.begin(), final_beam.end(), cmp_particle_weights);
      //remove items with worst scores
      for (int j = final_beam.size()- 1; (j >= static_cast<int>(beam_size)); --j)
        final_beam.pop_back();
    }
  }


  //print parses
  for (unsigned i = 0; (i < 5) && (i < final_beam.size()); ++i) {
    auto& parser = final_beam[i];
    parser.print_arcs();
    cout << parser.actions_str() << endl;

    float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/s.size();
    float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/s.size();

    cout << "  Dir Accuracy: " << dir_acc;
    cout << "  UnDir Accuracy: " << undir_acc;
    cout << "  Sample weight: " << (parser.particle_weight()) << endl; // /log(10)
  }  

  if (final_beam.size()==0) {
    cout << "no parse found" << endl;
    return ArcStandardParser(s, kORDER-1);  
  } else
    return final_beam[0];
}

//for greedy binary action decisions
ArcStandardParser greedy_parse_sentence(Words s, WxList goldd, Dict& dict, MT19937& eng, set<WordId>& vocabs, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& reduce_lm, PYPLM<kORDER>& arc_lm) {  
 
  //read in dev sentence; sample actions: for each action, execute and compute next word probability for shift
  cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl; 

  ArcStandardParser parser(s, kORDER-1); 

  do {
    Action a = Action::re; //placeholder action
    Words ctx = parser.word_context();
    //for (auto& c: ctx)
    //  cout << c << " ";
    //cout << ctx.at(ctx.size()-2) << " " << ctx.at(ctx.size()-1) << " ";
    double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), ctx);
    double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), ctx);
    cout << "(sh " << shiftp << ", re " << reducep << ") ";

    if ((parser.stack_depth()< kORDER-1) && !parser.is_buffer_empty()) 
      a = Action::sh;
    else if (parser.is_buffer_empty()) {
      a = Action::re;
      parser.add_importance_weight(reducep);
    } else {
      //shift or reduce

      if (shiftp >= reducep) {
        a = Action::sh;
        parser.add_particle_weight(shiftp);
      } else {
        a = Action::re; 
        parser.add_particle_weight(reducep);
      }
    }
    
    if (a == Action::sh) {
      WordId w = parser.next_word();
      double wordp = shift_lm.prob(w, ctx); 
      parser.shift();
      parser.add_particle_weight(wordp);

    } else if (a == Action::re) {
      double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
      double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);
      cout << "(la " << leftarcp << ", ra " << rightarcp << ") ";
    
      if ((leftarcp > rightarcp) && parser.left_arc()) {
        parser.add_particle_weight(leftarcp);
      } else {
        if (leftarcp > rightarcp) 
          parser.add_importance_weight(rightarcp);
        parser.right_arc();
        parser.add_particle_weight(rightarcp);
      }
    }
  } while (!parser.is_terminal_configuration());

  cout << endl;
  parser.print_arcs();
  cout << parser.actions_str() << endl;

  float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/s.size();
  float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/s.size();

  cout << "  Dir Accuracy: " << dir_acc;
  cout << "  UnDir Accuracy: " << undir_acc;
  cout << "  Importance weight: " << parser.importance_weight();
  cout << "  Sample weight: " << parser.particle_weight() << endl;

  return parser;
}

//for binary action decisions
void particle_parse_sentence(Words s, WxList goldd, Dict& dict, MT19937& eng, set<WordId>& vocabs, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& reduce_lm, PYPLM<kORDER>& arc_lm) {  
  double llh = 0;
  unsigned cnt = 0; 
  unsigned oovs = 0;
 
  //read in dev sentence; sample actions: for each action, execute and compute next word probability for shift
  cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl; 

  //extend to a number of particles (parser for each)
  //cout << "(" << s.size() << ") " << endl;
  vector<ArcStandardParser> particles(nPARTICLES, ArcStandardParser(s, kORDER-1)); 
  vector<double> particle_liw; //importance weights
  //particle_liw.resize(nPARTICLES);
  vector<double> particle_lp;
  //particle_lp.resize(nPARTICLES);

  //For now, construct the particles serially
  for (auto& parser: particles) {
    double part_llh = 0;
    double shift_llh = 0;
    double action_llh = 0;

    do {
    //while (!parser.is_buffer_empty()) {
      Action a = Action::re; //placeholder action
      Words ctx;
      double wordlp;
      double lp;
         
      ctx = parser.word_context();
      //cout << "word lp: " <<  wordlp << endl;

      if ((parser.stack_depth()< kORDER-1) && !parser.is_buffer_empty()) 
        a = Action::sh;
      else if (parser.is_buffer_empty())
        a = Action::re;
      else {
        //shift or reduce
        double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), ctx);
        double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), ctx);
        //cout << "(sh: " << shiftp << " re: " << reducep << ") ";

        //sample an action
        vector<double> distr = {shiftp, reducep};
        multinomial_distribution<double> mult(distr); 
        WordId act = mult(eng);
        lp = log(distr[act]); // / log(2);
        action_llh -= lp;
        //cout << act << " ";
      
        if (act==0) {
          a = Action::sh;
        } else {
          a = Action::re; 
        }
      }

      if (a == Action::sh) {
        WordId w = parser.next_word();
        wordlp = log(shift_lm.prob(w, ctx)); // log(2);

        parser.shift();
        if (vocabs.count(w) == 0) {
          ++oovs;
        } 
        //shift probability
        shift_llh -= wordlp;
      } else if (a == Action::re) {
        double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);
        //cout << "(la: " << leftarcp << " ra: " << rightarcp << ") ";

        //sample arc direction
        vector<double> distr = {leftarcp, rightarcp};
        multinomial_distribution<double> mult(distr); 
        WordId act = mult(eng);
        lp = log(distr[act]); // / log(2);
        a = static_cast<Action>(act+1);
        //cout << "(act) " << act << " ";

        if (!parser.execute_action(a)) 
          lp = 0;

        /* //if la invalid, force right arc         
        if (!parser.execute_action(a) && (a==Action::la)) {
          parser.right_arc();
          lp = log(distr[1]); // / log(2);
        } */

        action_llh -= lp;
        cnt++;
      }
    } while (!parser.is_terminal_configuration());

    //cnt++; //0 ll for invisible end-of-sentence marker
    //print action sequence
    //cout << parser.actions_str() << endl;
    parser.print_arcs();
        
    part_llh = shift_llh + action_llh;
    particle_liw.push_back(shift_llh);
    particle_lp.push_back(part_llh);
        
    float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/s.size();
    float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/s.size();

    cout << "  Dir Accuracy: " << dir_acc;
    cout << "  UnDir Accuracy: " << undir_acc;
    cout << "  Importance weight: " << (-shift_llh / log(10));
    cout << "  Sample weight: " << (-part_llh / log(10)) << endl;
    //cout << "  Log_10 action prob: " << (-action_llh * log(2) / log(10)) << endl;
    llh += part_llh;

  } //end of particles loop
    
  //**doesn;t using log base 2 in intermediate steps complicate things?

  // TODO: for now, only sample and print, need to add copy constructor for parser 
  // and have nice way to store resampled parses
    
  //cout << "particle weights: ";
  //for (unsigned i = 0; i < nPARTICLES; ++i) 
  //  cout << particle_liw[i] << " ";
  //cout << endl;

  // resample according to importance weight
  multinomial_distribution_log part_mult(particle_liw);
  vector<unsigned> sample_indx;

  //cout << "resampled importance weights: ";
  //cout << "resampled indexes: ";
  for (unsigned i = 0; i < nPARTICLES; ++i) {
    unsigned pi = part_mult(eng);
    //cout << (-particle_liw[pi] / log(10)) << " ";
    sample_indx.push_back(pi);
  }
  //cout << endl;
  sort(sample_indx.begin(), sample_indx.end());

  //cout << "sorted resampled indexes: ";
  //for (unsigned i = 0; i < nPARTICLES; ++i) {
  //  cout << sample_indx[i] << " ";
  //}
  //cout << endl;

  vector<bool> occur_indx; //-1 if not in sample, else its position 
  unsigned s_ind = 0; //index in sample_indx
  for (unsigned i = 0; i < nPARTICLES; ++i) { //over original particles
    //cout << s_ind << " ";
    if ((s_ind>=nPARTICLES) || (i < sample_indx[s_ind]))
      occur_indx.push_back(false);
    else if (i == sample_indx[s_ind]) {
      occur_indx.push_back(true); 
      while ((s_ind < nPARTICLES - 1) && (sample_indx[s_ind]==sample_indx[s_ind+1])) ++s_ind;
      ++s_ind;
    }
    else {
      //cout << "contra (" << i << ") ";     
    }
  }
  //cout << endl;

  vector<int> new_indx(nPARTICLES, -1); //just for the record
  //cout << "occur indexes: ";
  for (unsigned i = 0; i < nPARTICLES; ++i) {
    //cout << occur_indx[i] << " ";
    new_indx[i] = i;
  }
  //cout << endl;
  
  unsigned p_pos = 0; //over the particles that may be replaced
  for (unsigned i = 1; i < nPARTICLES; ++i) { //over sorted indexes
    if (sample_indx[i]==sample_indx[i-1]) {
      while ((p_pos < nPARTICLES) && (occur_indx[p_pos])) {
        //new_indx[p_pos] = p_pos;          
        ++p_pos; //not supposed to go out of index
      }
      //actual (copy) reassignment
      if (p_pos < nPARTICLES) {
        //cout << p_pos << " ";
        particles[p_pos] = particles[sample_indx[i]];       
        particle_liw[p_pos] = particle_liw[sample_indx[i]]; //also copy weight
        new_indx[p_pos] = sample_indx[i];
        //cout << "(" << i << ", " << sample_indx[i] << ") ";
        ++p_pos;
      } else
        cout << "invalid index ";
      //cout << endl;
    } 
  }

  //cout << "replaced resampled importance weights: ";
  //for (unsigned i = 0; i < nPARTICLES; ++i) {
  //  cout << (-particle_liw[i] / log(10)) << " ";
  //}
  //cout << endl;
  
  cout << "resampled indexes: ";
  for (unsigned i = 0; i < nPARTICLES; ++i) {
    cout << new_indx[i] << " ";
  }
  cout << endl;
}

//for three-way action decisions
void particle_parse_sentence(Words s, WxList goldd, Dict& dict, MT19937& eng, set<WordId>& vocabs, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& action_lm) {  
  double llh = 0;
  unsigned cnt = 0; 
  unsigned oovs = 0;
 
  //read in dev sentence; sample actions: for each action, execute and compute next word probability for shift
  //TODO print gold dependencies, get accuracies
  cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl; 

  //extend to a number of particles (parser for each)
  //cout << "(" << s.size() << ") " << endl;
  vector<ArcStandardParser> particles(nPARTICLES, ArcStandardParser(s, kORDER-1)); 
  vector<double> particle_liw; //importance weights
  //particle_liw.resize(nPARTICLES);
  vector<double> particle_lp;
  //particle_lp.resize(nPARTICLES);

  //For now, construct the particles serially
  for (auto& parser: particles) {
    double part_llh = 0;
    double shift_llh = 0;
    double action_llh = 0;

    while (!parser.is_buffer_empty()) {
      Action a = Action::re; //placeholder action
      WordId w = parser.next_word();
      Words ctx;
      double wordlp;
         
      //sample action while action<>shift
      while (a != Action::sh) {
        ctx = parser.word_context();
        wordlp = log(shift_lm.prob(w, ctx)); // log(2);
        //cout << "word lp: " <<  wordlp << endl;

        if ((parser.stack_depth()< kORDER-1) && !parser.is_buffer_empty()) {
          a = Action::sh;
          parser.execute_action(a);
        } else {
          double shiftp = action_lm.prob(static_cast<WordId>(Action::sh), ctx);
          double leftarcp = action_lm.prob(static_cast<WordId>(Action::la), ctx);
          double rightarcp = action_lm.prob(static_cast<WordId>(Action::ra), ctx);
              
          //if (shiftp==0 || leftarcp==0 || rightarcp==0)
          //  cout << "Probs: " << shiftp << " " << leftarcp << " " << rightarcp << endl;

          vector<double> distr = {shiftp, leftarcp, rightarcp};
          multinomial_distribution<double> mult(distr); 
          WordId act = mult(eng);
          a = static_cast<Action>(act);
          if (!parser.execute_action(a)) {
            a = Action::re;
            continue;
          }

          double lp = log(distr[act]); // / log(2);
          action_llh -= lp;
          cnt++;
        }
      } 

      //shift already executed above
      if (vocabs.count(w) == 0) {
        //cout << "OOV prob a: " << wordlp << endl;
        //how should this be handled properly? we do not get 0 prob here...
        //two probabilities
        ++oovs;
      } //else
        //shift probability
        shift_llh -= wordlp;
    }
    //cout << "shift completed" << endl;

    //actions to empty the stack
    while (!parser.is_terminal_configuration()) {
      //action_seq.size() < 2*s.size()) {
      //can we just ignore shift probability?
      Words ctx = parser.word_context();
      // cout << "context: ";
      //for (unsigned i = 0; i < ctx.size(); ++i) 
      //  cout << ctx[i] << " ";
      //cout << endl;
          
      if (parser.stack_depth()<2) {
        cout << "parse failed" << endl;  
      } else {
        double leftarcp = action_lm.prob(static_cast<WordId>(Action::la), ctx);
        double rightarcp = action_lm.prob(static_cast<WordId>(Action::ra), ctx);
        //cout << "Probs: " << leftarcp << " " << rightarcp << endl;
        //if (leftarcp==0 || rightarcp==0)
        //  cout << "Probs: " << " " << leftarcp << " " << rightarcp << endl;

        vector<double> distr = {leftarcp, rightarcp};
        multinomial_distribution<double> mult(distr); 
        WordId act = mult(eng) + 1;
        //cout << act << endl;
        Action a = static_cast<Action>(act);
        parser.execute_action(a);
          
        double lp = log(distr[act-1]); // / log(2); //should be normalized prob?
        action_llh -= lp;
        cnt++; 
      }
    }
        
    cnt++; //0 ll for invisible end-of-sentence marker
    //print action sequence
    //cout << parser.actions_str() << endl;
    parser.print_arcs();
        
    part_llh = shift_llh + action_llh;
    particle_liw.push_back(shift_llh);
    particle_lp.push_back(part_llh);
        
    float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/s.size();
    float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/s.size();

    cout << "  Dir Accuracy: " << dir_acc;
    cout << "  UnDir Accuracy: " << undir_acc;
    cout << "  Importance weight: " << (-shift_llh / log(10));
    cout << "  Sample weight: " << (-part_llh / log(10)) << endl;
    //cout << "  Log_10 action prob: " << (-action_llh * log(2) / log(10)) << endl;
    llh += part_llh;

  } //end of particles loop
    
  //**doesn;t using log base 2 in intermediate steps complicate things?

  // resample according to importance weight
  // TODO: for now, only sample and print, need to add copy constructor for parser 
  // and have nice way to store resampled parses
    
  //cout << "particle weights: ";
  //for (unsigned i = 0; i < nPARTICLES; ++i) 
  //  cout << particle_liw[i] << " ";
  //cout << endl;

  multinomial_distribution_log part_mult(particle_liw);
  cout << "resampled importance weights: ";
  for (unsigned i = 0; i < nPARTICLES; ++i) {
    unsigned pi = part_mult(eng);
    cout << pi << " (" <<  (-particle_liw[pi] / log(10)) << ") ";
  }
  cout << endl;
}

/*train a generative dependency parsing model using given context vectors;
 * test, for now with a single sampling model 
 */
int main(int argc, char** argv) {
  //for now, hard_code filenames
  if (argc != 2) {
    cerr << argv[0] << " <nsamples>\n\nEstimate a " 
         << kORDER << "-gram HPYP LM and report perplexity\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }

  int samples = atoi(argv[1]);
/*
  string train_file = "conll2007-english/english_ptb_train.conll";
  string test_file = "conll2007-english/english_ptb_dev.l.conll.sentences";
  string test_dependencies_file = "conll2007-english/english_ptb_dev.l.conll.dependencies"; */

  string train_file = "dutch-conll/dutch_alpino_train.conll";
  string test_file = "dutch-conll/dutch_alpino_dev.conll.sentences";
  string test_dependencies_file = "dutch-conll/dutch_alpino_dev.conll.dependencies";

  //used for all the models 
  Dict dict("ROOT", "", true);
  MT19937 eng;
  const WordId kSOS = dict.Convert("ROOT"); 
  vector<WordId> ctx(kORDER - 1, kSOS);

  const unsigned num_word_types = 26502; 
  //const unsigned num_word_types = 56574; //hardcoded to save trouble TODO change this

  set<WordId> vocabs;
  std::vector<Words> corpussh;
  std::vector<Words> corpusre;
  

  PYPLM<kORDER> shift_lm(num_word_types, 1, 1, 1, 1);
  //PYPLM<kORDER> reduce_lm(3, 1, 1, 1, 1);
  
  std::vector<Words> corpusarc;
  PYPLM<kORDER> reduce_lm(2, 1, 1, 1, 1);
  PYPLM<kORDER> arc_lm(2, 1, 1, 1, 1);
 
  //train 
  train_raw(train_file, dict, vocabs, corpussh, corpusre, corpusarc); //extract training examples 
  
  cerr << "\nTraining word model...\n";
   train_lm(samples, eng, dict, corpussh, shift_lm);
  cerr << "\nTraining shift/reduce model...\n";
  train_lm(samples, eng, dict, corpusre, reduce_lm);
  cerr << "\nTraining arc model...\n";
  train_lm(3*samples, eng, dict, corpusarc, arc_lm); //seems to converge slower
    
/*  train_raw(train_file, dict, vocabs, corpussh, corpusre); //extract training examples 
  train_lm(samples, eng, dict, corpussh, shift_lm);
  train_lm(samples, eng, dict, corpusre, reduce_lm);  */

  set<WordId> tv;
  vector<Words> test;
  vector<WxList> testd;

  std::cerr << "Reading test corpus...\n";
  ReadFromFile(test_file, &dict, &test, &tv);  
  std::cerr << "Test corpus size: " << test.size() << " sentences\t (" << tv.size() << " word types)\n";
  ReadFromDependencyFile(test_dependencies_file, &testd);

// lm.print(cerr);
  unsigned beam_size = 16;
  //unsigned num_particles = 100;
  int total_length = 0;
  int directed_count = 0;
  int undirected_count = 0;


  cerr << "\nParsing test sentences...\n";
  for (unsigned j = 0; j < test.size(); ++j) {
    //particle_parse_sentence(test[j], testd[j], dict, eng, vocabs, shift_lm, reduce_lm);
    if (test[j].size() <= 100) {
      //ArcStandardParser b_parser = greedy_parse_sentence(test[j], testd[j], dict, eng, vocabs, shift_lm, reduce_lm, arc_lm);
      //ArcStandardParser b_parser = particle_par_parse_sentence(test[j], testd[j], beam_size, false, dict, eng, vocabs, shift_lm, reduce_lm, arc_lm);
      ArcStandardParser b_parser = beam_parse_sentence(test[j], testd[j], beam_size, dict, eng, vocabs, shift_lm, reduce_lm, arc_lm);
      total_length += (testd[j].size() - 1);
      directed_count += b_parser.directed_accuracy_count(testd[j]);
      undirected_count += b_parser.undirected_accuracy_count(testd[j]);
    }
    //simple_beam(test[j]);
  }

  cerr << "Directed Accuracy:" << (directed_count + 0.0)/total_length << endl;
  cerr << "Undirected Accuracy:" << (undirected_count + 0.0)/total_length << endl;

  //cnt -= oovs;

/*  
  cerr << "One sequence sampled score:" << endl;
  cerr << "  Log_10 prob: " << (-llh * log(2) / log(10)) << endl;
  cerr << "        Count: " << cnt << endl;
  cerr << "         OOVs: " << oovs << endl;
  cerr << "Cross-entropy: " << (llh / cnt) << endl;
  cerr << "   Perplexity: " << pow(2, llh / cnt) << endl; */
  return 0;
             
  // test against the gold standard dev for perplexity
/*  
  unsigned nsent = 1308; // for word prediction (add 1 for end of each test sentence)
  double llh = 0;
  unsigned cnt = nsent; // 0; 
  unsigned oovs = 0;

  string test_file = "dutch_alpino_dev.conll.words.contexts";
 
  set<WordId> tvs;
  vector<vector<WordId> > tests;
  ReadFromFile(test_file, &dict, &tests, &tvs);  

  for (auto& s : tests) {
    WordId w = s[0];
    ctx = vector<WordId>(s.begin()+1, s.end());
    double lp = log(shift_lm.prob(w, ctx)) / log(2);
    if (vocabs.count(w) == 0) {
      ++oovs;
      lp = 0;
    }
    llh -= lp;
    cnt++;
  }

  // calculate seperately?
  cnt -= oovs;
  cerr << "Gold standard scoring (words):" << endl;
  cerr << "  Log_10 prob: " << (-llh * log(2) / log(10)) << endl;
  cerr << "        Count: " << cnt << endl;
  cerr << "         OOVs: " << oovs << endl;
  cerr << "Cross-entropy: " << (llh / cnt) << endl;
  cerr << "   Perplexity: " << pow(2, llh / cnt) << endl;

  llh = 0;
  cnt = 0; 
  oovs = 0;   
  //
  test_file = "dutch_alpino_dev.conll.actions.contexts";
 
  set<WordId> tvr;
  vector<vector<WordId> > testr;
  ReadFromFile(test_file, &dict, &testr, &tvr);  

  for (auto& s : testr) {
    WordId w = s[0];
    ctx = vector<WordId>(s.begin()+1, s.end());
    double lp = log(action_lm.prob(w, ctx)) / log(2);
    if (vocabr.count(w) == 0) {
      ++oovs;
      lp = 0;
    }
    llh -= lp;
    cnt++;
  }

  cnt -= oovs;
  cerr << "Gold standard scoring:" << endl;
  cerr << "  Log_10 prob: " << (-llh * log(2) / log(10)) << endl;
  cerr << "        Count: " << cnt << endl;
  cerr << "         OOVs: " << oovs << endl;
  cerr << "Cross-entropy: " << (llh / cnt) << endl;
  cerr << "   Perplexity: " << pow(2, llh / cnt) << endl;
*/ 

}

