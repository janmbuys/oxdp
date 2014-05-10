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

//for binary action decisions, resample at each step
void particle_par_parse_sentence(Words s, WxList goldd, Dict& dict, MT19937& eng, set<WordId>& vocabs, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& reduce_lm, PYPLM<kORDER>& arc_lm) {  
  //double llh = 0;
  //unsigned cnt = 0; 
  //unsigned oovs = 0;
 
  cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl; 

  vector<ArcStandardParser> particles(nPARTICLES, ArcStandardParser(s, kORDER-1)); 
  //vector<double> particle_liw(nPARTICLES, 0); //importance weights
  //vector<double> particle_lp(nPARTICLES, 0); //particle weights

  for (unsigned i = 0; i < s.size() + 1; ++i) {
    for (unsigned j = 0; j < nPARTICLES; ++j) { 
      ArcStandardParser& parser = particles[j];

      Action a = Action::re; //placeholder action
      //sample a sequence of possible actions leading up to the next shift
      while (a == Action::re) {
        Words ctx = parser.word_context();
    
        if ((parser.stack_depth()< kORDER-1) && !parser.is_buffer_empty()) 
          a = Action::sh;
        else {
          //shift or reduce
          double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), ctx);
          double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), ctx);
          //cout << "(sh: " << shiftp << " re: " << reducep << ") ";

          //sample an action
          vector<double> distr = {shiftp, reducep};
          multinomial_distribution<double> mult(distr); 
          WordId act = mult(eng);
          parser.add_particle_weight(-log(distr[act])); 
          //cout << act << " ";
      
          if (act==0) {
            a = Action::sh;
          } else {
            a = Action::re; 
          }
        }

        if (a == Action::re) {
          double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
          double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);
          //cout << "(la: " << leftarcp << " ra: " << rightarcp << ") ";

          //sample arc direction
          vector<double> distr = {leftarcp, rightarcp};
          multinomial_distribution<double> mult(distr); 
          WordId act = mult(eng);
          a = static_cast<Action>(act+1);
          //cout << "(act) " << act << " ";

          if (parser.execute_action(a)) 
            parser.add_particle_weight(-log(distr[act])); 
        }
      }

      //perform shift
      Words ctx = parser.word_context();
      WordId w = parser.next_word();
      double wordlp = -log(shift_lm.prob(w, ctx)); 
      parser.set_importance_weight(wordlp); //else add_importance_weight
      parser.add_particle_weight(wordlp); //else add_importance_weight
      parser.shift();
    }
    //TODO resamples particles
    if (i > 1) {
      //cout << "particle weights: ";
      //for (unsigned i = 0; i < nPARTICLES; ++i) 
      //  cout << particles[i].importance_weight() << " ";
      //cout << endl;

      // resample according to importance weight
      
      vector<double> importance_w(nPARTICLES, 0); //importance weights
      for (unsigned i = 0; i < nPARTICLES; ++i)
        importance_w[i] = particles[i].importance_weight();

      multinomial_distribution_log part_mult(importance_w); 
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
          cout << "contra (" << i << ") ";     
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
            //particle_liw[p_pos] = particle_liw[sample_indx[i]]; //also copy weight
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
  
      //cout << "resampled weights: ";
      //for (unsigned i = 0; i < nPARTICLES; ++i) {
      //  cout << particles[i].importance_weight() << " ";
      //}
      //cout << endl; 
    }
  }

  //cerr << "start completion" << endl;
  for (unsigned j = 0; j < nPARTICLES; ++j) { 
    ArcStandardParser& parser = particles[j];
    Action a = Action::re; //placeholder action
  
    //std::cerr << parser.is_buffer_empty() << " " << parser.stack_depth() << std::endl;
    while (!parser.is_terminal_configuration()) {
        //sample arcs to complete the parse
      Words ctx = parser.word_context();
      double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
      double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);
      //cout << "(la: " << leftarcp << " ra: " << rightarcp << ") ";

      //sample arc direction
      vector<double> distr = {leftarcp, rightarcp};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
      a = static_cast<Action>(act+1);
    
      //cout << "(act) " << act << " ";
      //if la invalid, force right arc         
      if (parser.execute_action(a)) 
        parser.add_particle_weight(-log(distr[act])); 
            
      else if (a==Action::la) {
        parser.right_arc();
        //particle_lp[j] -= log(distr[1]); //don't assign a prob to this? 
      }
      //cerr << "depth: " << parser.stack_depth() << endl;
    }
  }

  //print parses

  sort(particles.begin(), particles.end(), cmp_particle_weights);

  for (unsigned j = 0; j < nPARTICLES; ++j) { 
    ArcStandardParser& parser = particles[j];
    parser.print_arcs();
        
    float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/s.size();
    float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/s.size();

    cout << "  Dir Accuracy: " << dir_acc;
    cout << "  UnDir Accuracy: " << undir_acc;
    cout << "  Sample weight: " << (-parser.particle_weight() / log(10)) << endl;
  } 
}

//TODO beam parser
void beam_parse_sentence(Words s, WxList goldd, unsigned beam_size, Dict& dict, MT19937& eng, set<WordId>& vocabs, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& reduce_lm, PYPLM<kORDER>& arc_lm) {  
  //double llh = 0;
  //unsigned cnt = 0; 
  //unsigned oovs = 0;
 
  cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl; 

  vector<ArcStandardParser> beam(1, ArcStandardParser(s, kORDER-1));  
  //vector<ArcStandardParser> particles;

  for (unsigned i = 0; i < s.size() + 1; ++i) {
    double min_w = 0; // worst weight so far on beam 
    for (unsigned j = 0; j < beam.size(); ++j) { 
      ArcStandardParser& parser = beam[j];

      //sample a sequence of possible actions leading up to the next shift
      Words ctx = parser.word_context();
      double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), ctx);
    
      if (parser.stack_depth() >= kORDER-1) {
        //add paths for reduce actions
        double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);

        ArcStandardParser left_p = parser; //will this copy?
        ArcStandardParser right_p = parser; 

        //TODO add beam size check
        if (left_p.execute_action(Action::la)) {
          left_p.add_particle_weight(-reducep-leftarcp); 
          beam.push_back(left_p);
        }

        right_p.add_particle_weight(-reducep-rightarcp); 
        beam.push_back(right_p);
      }

      //perform shift
      WordId w = parser.next_word();
      double wordlp = log(shift_lm.prob(w, ctx)); 
      parser.set_importance_weight(-wordlp); //else add_importance_weight
      parser.add_particle_weight(-shiftp-wordlp); //else add_importance_weight
      parser.shift();
    }
   
    sort(particles.begin(), particles.end(), cmp_particle_weights);
    
    for (j = beam.size()- 1; j >= beam_size; --j)
      beam.pop_back();
  }

  //cerr << "start completion" << endl;
  

  for (unsigned j = 0; j < nPARTICLES; ++j) { 
    ArcStandardParser& parser = particles[j];
    Action a = Action::re; //placeholder action
  
    //std::cerr << parser.is_buffer_empty() << " " << parser.stack_depth() << std::endl;
    while (!parser.is_terminal_configuration()) {
        //sample arcs to complete the parse
      Words ctx = parser.word_context();
      double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
      double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);
      //cout << "(la: " << leftarcp << " ra: " << rightarcp << ") ";

      //sample arc direction
      vector<double> distr = {leftarcp, rightarcp};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
      a = static_cast<Action>(act+1);
    
      //cout << "(act) " << act << " ";
      //if la invalid, force right arc         
      if (parser.execute_action(a)) 
        parser.add_particle_weight(-log(distr[act])); 
            
      else if (a==Action::la) {
        parser.right_arc();
        //particle_lp[j] -= log(distr[1]); //don't assign a prob to this? 
      }
      //cerr << "depth: " << parser.stack_depth() << endl;
    }
  }

  //print parses

  sort(particles.begin(), particles.end(), cmp_particle_weights);

  for (unsigned j = 0; j < nPARTICLES; ++j) { 
    ArcStandardParser& parser = particles[j];
    parser.print_arcs();
        
    float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/s.size();
    float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/s.size();

    cout << "  Dir Accuracy: " << dir_acc;
    cout << "  UnDir Accuracy: " << undir_acc;
    cout << "  Sample weight: " << (-parser.particle_weight() / log(10)) << endl;
  } 
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
 
  //used for all the models 
  Dict dict("ROOT", "", true);
  MT19937 eng;
  const WordId kSOS = dict.Convert("ROOT"); 
  vector<WordId> ctx(kORDER - 1, kSOS);

  const unsigned num_word_types = 26502; //hardcoded to save trouble

  set<WordId> vocabs;
  std::vector<Words> corpussh;
  std::vector<Words> corpusre;
  
  string train_file = "dutch_alpino_train.conll";

  PYPLM<kORDER> shift_lm(num_word_types, 1, 1, 1, 1);
  //PYPLM<kORDER> reduce_lm(3, 1, 1, 1, 1);
  
  std::vector<Words> corpusarc;
  PYPLM<kORDER> reduce_lm(2, 1, 1, 1, 1);
  PYPLM<kORDER> arc_lm(2, 1, 1, 1, 1);
 
  //train 
  
  train_raw(train_file, dict, vocabs, corpussh, corpusre, corpusarc); //extract training examples 
  train_lm(samples, eng, dict, corpussh, shift_lm);
  train_lm(samples, eng, dict, corpusre, reduce_lm);
  train_lm(samples, eng, dict, corpusarc, arc_lm);  
  
  /*  
  train_raw(train_file, dict, vocabs, corpussh, corpusre); //extract training examples 
  train_lm(samples, eng, dict, corpussh, shift_lm);
  train_lm(samples, eng, dict, corpusre, reduce_lm);  */

  set<WordId> tv;
  vector<Words> test;
  vector<WxList> testd;
  string test_file = "dutch_alpino_dev.conll.sentences";

  std::cerr << "Reading test corpus...\n";
  ReadFromFile(test_file, &dict, &test, &tv);  
  std::cerr << "Test corpus size: " << test.size() << " sentences\t (" << tv.size() << " word types)\n";
  
  std::string test_dependencies_file = "dutch_alpino_dev.conll.dependencies";
  ReadFromDependencyFile(test_dependencies_file, &testd);

// lm.print(cerr);
  
  for (unsigned j = 0; j < test.size(); ++j) {
    //particle_parse_sentence(test[j], testd[j], dict, eng, vocabs, shift_lm, reduce_lm);
    particle_par_parse_sentence(test[j], testd[j], dict, eng, vocabs, shift_lm, reduce_lm, arc_lm);
  
  }

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

