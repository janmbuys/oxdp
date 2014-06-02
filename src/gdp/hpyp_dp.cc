#include <iostream>
#include <unordered_map>
#include <cstdlib>
#include <chrono>

#include "transition_parser.h"
#include "hpyp_dp_train.h"
#include "hpyplm/hpyplm.h"
#include "corpus/corpus.h"
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

#define kORDER 3  //default 4

using namespace std;
using namespace oxlm;

typedef std::unique_ptr<ArcStandardParser> ParserPtr;
typedef std::vector<std::unique_ptr<ArcStandardParser>> ParserList;

void resample_particles(vector<ArcStandardParser>& particles, unsigned num_particles, MT19937& eng) {
  vector<double> importance_w(num_particles, 0); //importance weights
  for (unsigned i = 0; i < num_particles; ++i)
    importance_w[i] = particles[i].importance_weight();

  //resample according to importance weight
  multinomial_distribution_log part_mult(importance_w); 
  vector<unsigned> sample_indx;
  for (unsigned i = 0; i < num_particles; ++i) {
    unsigned pi = part_mult(eng);
    sample_indx.push_back(pi);
  }
  sort(sample_indx.begin(), sample_indx.end());

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
  for (unsigned i = 0; i < num_particles; ++i) {
    new_indx[i] = i;
  }
  
  unsigned p_pos = 0; //over the particles that may be replaced
  for (unsigned i = 1; i < num_particles; ++i) { //over sorted indexes
    if (sample_indx[i]==sample_indx[i-1]) {
      while ((p_pos < num_particles) && (occur_indx[p_pos])) ++p_pos; //not supposed to go out of index 
      //actual (copy) reassignment
      if (p_pos < num_particles) {
        particles[p_pos] = particles[sample_indx[i]]; //key line: copy particle      
        new_indx[p_pos] = sample_indx[i];
        ++p_pos;
      } else
        cout << "invalid index ";
    } 
  }
}

//for binary action decisions, resample at each step
ArcStandardParser particle_par_parse_sentence(Words s, WxList goldd, unsigned num_particles, bool resample, Dict& dict, MT19937& eng, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& reduce_lm, PYPLM<kORDER>& arc_lm) {  
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
   //cout << endl;
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

//gold parser
ArcStandardParser gold_parse_sentence(Words s, WxList goldd, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& reduce_lm, PYPLM<kORDER>& arc_lm) {  
//compute and score oracle parse sequence
  ArcStandardParser gold_p(s, kORDER-1);

  while (!gold_p.is_terminal_configuration()) {
    Words ctx = gold_p.word_context();
    double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), ctx);
    double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), ctx);
    
    Action nexta = gold_p.oracle_next(goldd);
    if (nexta==Action::sh) {
      WordId w = gold_p.next_word();
      double wordp = shift_lm.prob(w, ctx); 
      gold_p.shift();
      gold_p.add_importance_weight(wordp); 
      gold_p.add_particle_weight(shiftp*wordp); 
    } else {
      double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
      double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);

      if (gold_p.is_buffer_empty())
        gold_p.add_importance_weight(reducep);
      if (nexta==Action::la) {
        gold_p.left_arc();
        gold_p.add_particle_weight(reducep*leftarcp); 
      } else if (nexta==Action::ra) {
        gold_p.right_arc();
        gold_p.add_particle_weight(reducep*rightarcp); 
      } else {
        cerr << "Invalid gold parse." << endl;
      }
    }
  }

  cout << "Gold weight: " << gold_p.particle_weight() << endl;
  return gold_p;
}

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>{new T{args...}};
} 

//more sophisticated beam parser
template<unsigned kORD, unsigned sORD, unsigned nORD> 
ArcStandardParser beam_list_parse_sentence(Words s, Words ps, WxList goldd, unsigned beam_size, Dict& dict, MT19937& eng, PYPLM<sORD>& shift_lm, PYPLM<nORD>& reduce_lm, PYPLM<nORD>& arc_lm, PYPLM<kORD>& tag_lm) {  
  //index in beam_stack is depth-of-stack - 1
  vector<ParserList> beam_stack; // (1, ParserList());
  beam_stack.push_back(ParserList());
  //beam_stack.push_back(ParserList(1, make_unique<ArcStandardParser>(s, ps, kORD-1)));
  beam_stack[0].push_back(make_unique<ArcStandardParser>(s, ps, kORD-1));
  beam_stack[0][0]->print_sentence(dict);
  beam_stack[0][0]->print_postags(dict);
  
  cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl;   

  //shift ROOT symbol (probability 1)
  beam_stack[0][0]->shift(); 

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 1; k <= s.size(); ++k) {
    //there are k beam lists. perform reduces down to list 1

    for (unsigned i = k - 1; i > 0; --i) { 
    //for (unsigned i = 0; i < k; ++i) { 
      //prune if size exceeds beam_size
      if (beam_stack[i].size() > beam_size) {
        sort(beam_stack[i].begin(), beam_stack[i].end(), cmp_particle_ptr_weights); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_stack[i].size(); j > beam_size; --j)
          beam_stack[i].pop_back();
      }
  
      //for every item in the list, add valid reduce actions to list i - 1 
      for (unsigned j = 0; (j < beam_stack[i].size()); ++j) {
        Words r_ctx = beam_stack[i][j]->word_pos_context();
        Words a_ctx = beam_stack[i][j]->word_pos_context();
        double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), r_ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), a_ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), a_ctx);
       
        beam_stack[i-1].push_back(make_unique<ArcStandardParser>(*beam_stack[i][j]));
        if (i > 1) { //left arc only invalid when stack size is 2 **
          beam_stack[i-1].push_back(make_unique<ArcStandardParser>(*beam_stack[i][j]));
          //ParserPtr ra_parser = beam_stack[i-1].rbegin()[1]; 
          //ParserPtr la_parser = beam_stack[i-1].back(); 

          beam_stack[i-1].back()->left_arc();
          beam_stack[i-1].back()->add_particle_weight(reducep*leftarcp);
          beam_stack[i-1].rbegin()[1]->right_arc();
          beam_stack[i-1].rbegin()[1]->add_particle_weight(reducep*rightarcp); 

          if (k == s.size()) {  
            beam_stack[i-1].back()->set_importance_weight(reducep); 
            beam_stack[i-1].rbegin()[1]->set_importance_weight(reducep); 
          }
        } else {
          //ParserPtr ra_parser = beam_stack[i-1].back(); 
          beam_stack[i-1].back()->right_arc();
          beam_stack[i-1].back()->add_particle_weight(reducep*rightarcp); 
          
          if (k == s.size()) 
            beam_stack[i-1].back()->set_importance_weight(rightarcp); 
        }
      }
    }

    if ((beam_stack[0].size() > beam_size) || (k == s.size())) {
        sort(beam_stack[0].begin(), beam_stack[0].end(), cmp_particle_ptr_weights); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_stack[0].size(); j > beam_size; --j)
          beam_stack[0].pop_back();
      }

    //perform shifts
    if (k < s.size()) {
      for (unsigned i = 0; (i < k); ++i) { 
        for (unsigned j = 0; j < beam_stack[i].size(); ++j) {
          //ParserPtr sh_parser = beam_stack[i][j];

          Words w_ctx = beam_stack[i][j]->word_posx_context();
          Words r_ctx = beam_stack[i][j]->word_pos_context();
          Words t_ctx = beam_stack[i][j]->pos_context();
          WordId w = beam_stack[i][j]->next_word();
          WordId tag = beam_stack[i][j]->next_tag();
          double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), r_ctx);
          double wordp = shift_lm.prob(w, w_ctx); 
          double tagp = tag_lm.prob(tag, t_ctx);

          beam_stack[i][j]->shift();
          beam_stack[i][j]->set_importance_weight(wordp); 
          beam_stack[i][j]->add_importance_weight(tagp); 
          beam_stack[i][j]->add_particle_weight(shiftp*wordp); 
          beam_stack[i][j]->add_particle_weight(tagp); 
        }
      }
      //beam_stack.push_back(ParserList());
      //insert new beam_stack[0] to increment indexes
      beam_stack.insert(beam_stack.begin(), ParserList());
    }   
  }
  
  //print parses
  unsigned n = 0; //s.size() - 1;
  for (unsigned i = 0; (i < 5) && (i < beam_stack[n].size()); ++i) {
    //ParserPtr parser = beam_stack[0][i];
    beam_stack[n][i]->print_arcs();
    //cout << parser->actions_str() << endl;

    float dir_acc = (beam_stack[n][i]->directed_accuracy_count(goldd) + 0.0)/(s.size()-1);
    //float undir_acc = (beam_stack[n][i]->undirected_accuracy_count(goldd) + 0.0)/(s.size()-1);

    cout << "  Dir Accuracy: " << dir_acc;
    //cout << "  UnDir Accuracy: " << undir_acc;
    cout << "  Sample weight: " << (beam_stack[n][i]->particle_weight()) << endl;
  }  

  if (beam_stack[n].size()==0) {
    cout << "no parse found" << endl;
    return ArcStandardParser(s, kORD-1);  
  } else
    return ArcStandardParser(*beam_stack[n][0]); 
}  


//beam parser
template<unsigned kORD, unsigned sORD, unsigned nORD> 
ArcStandardParser beam_parse_sentence(Words s, Words ps, WxList goldd, unsigned beam_size, Dict& dict, MT19937& eng, PYPLM<sORD>& shift_lm, PYPLM<nORD>& reduce_lm, PYPLM<nORD>& arc_lm, PYPLM<kORD>& tag_lm) {  
  vector<ArcStandardParser> beam(1, ArcStandardParser(s, ps, kORD-1));  
  /* beam[0].print_sentence(dict);
  beam[0].print_postags(dict);
  
  cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl;   */
 
  for (unsigned i = 0; i < s.size(); ++i) {
    unsigned sh_bound = beam.size(); 
    for (unsigned j = 0; j < beam.size(); ++j) { 
      //sample a sequence of possible actions leading up to the next shift
      Words w_ctx = beam[j].word_posx_context();
      Words p_ctx = beam[j].word_pos_context();
      Words r_ctx = beam[j].word_pos_context();
      Words t_ctx = beam[j].pos_context();
      //if (j%10 == 0)
      //  cerr << j << " " << beam.size() << endl;

      if (beam[j].stack_depth() >= kORD-1) {
        //add paths for reduce actions
        double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), r_ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), p_ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), p_ctx);
       
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
      double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), r_ctx);
      WordId w = shift_p.next_word();
      double wordp = shift_lm.prob(w, w_ctx); 
      WordId tag = shift_p.next_tag();
      double tagp = tag_lm.prob(tag, t_ctx);

      shift_p.shift();
      shift_p.set_importance_weight(wordp); 
      shift_p.add_importance_weight(tagp); 
      shift_p.add_particle_weight(shiftp*wordp); 
      shift_p.add_particle_weight(tagp); 
      //cerr << "p: " <<  parser.stack_depth() << " " << parser.particle_weight() << " " << parser.buffer_length() << " " << parser.sentence_length() << endl;

      if (j==sh_bound) {
        if (beam.size() > (beam_size + sh_bound)) {
         //cap unprocessed items for each local number of operations at beam size
         sort(beam.begin() + (beam_size + sh_bound), beam.end(), cmp_particle_weights);
         for (unsigned j = beam.size()- 1; j >= (beam_size + sh_bound); --j)
           beam.pop_back();
        }

        sh_bound = beam.size();
      }
    }
   
    //otherwise compare normalized weights 
    sort(beam.begin(), beam.end(), cmp_particle_weights);
    //remove items with worst scores
    for (unsigned j = beam.size()- 1; j >= beam_size; --j)
      beam.pop_back();
  }

  //cerr << "start completion" << endl;
  vector<ArcStandardParser> final_beam;  

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
        //Words w_ctx = beam[j].word_posx_context();
        Words p_ctx = beam[j].word_pos_context();
        Words r_ctx = beam[j].word_pos_context();
        double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), r_ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), p_ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), p_ctx);

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

    //can also compare normalized weights 
    sort(beam.begin(), beam.end(), cmp_particle_weights);
    //remove items with worst scores
    for (int j = beam.size()- 1; ((j >= static_cast<int>(beam_size)) || ((j >= 0) && (beam[j].particle_weight() >= 1000))); --j) {
      beam.pop_back();
    }

    if ((final_beam.size() > 2*beam_size) || (beam.size()==0)) {
      sort(final_beam.begin(), final_beam.end(), cmp_particle_weights);
      //remove items with worst scores
      for (int j = final_beam.size()- 1; (j >= static_cast<int>(beam_size)); --j)
        final_beam.pop_back();
    }
  }

  /*
  //print parses
  for (unsigned i = 0; (i < 5) && (i < final_beam.size()); ++i) {
    auto& parser = final_beam[i];
      parser.print_arcs();
    //cout << parser.actions_str() << endl;

    float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/(s.size()-1);
    float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/(s.size()-1);

    cout << "  Dir Accuracy: " << dir_acc;
    cout << "  UnDir Accuracy: " << undir_acc;
    cout << "  Sample weight: " << (parser.particle_weight()) << endl; // /log(10) 
  }  */

  if (final_beam.size()==0) {
    cout << "no parse found" << endl;
    return ArcStandardParser(s, kORD-1);  
  } else
    return final_beam[0];
}

//beam parser
template<unsigned kORD, unsigned sORD, unsigned nORD> 
ArcStandardParser beam_parse_align_sentence(Words s, Words ps, WxList goldd, unsigned beam_size, Dict& dict, MT19937& eng, PYPLM<sORD>& shift_lm, PYPLM<nORD>& reduce_lm, PYPLM<nORD>& arc_lm, PYPLM<kORD>& tag_lm) {  
  vector<ArcStandardParser> beam(1, ArcStandardParser(s, ps, kORD-1));  
  beam[0].print_sentence(dict);
  beam[0].print_postags(dict);
  
  cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl;  
 
  //all beam items will terminate after same number of steps
  for (unsigned i = 0; i < 2*(s.size()-1); ++i) {
    unsigned b_size = beam.size(); 
    for (unsigned j = 0; j < b_size; ++j) { 
    //cout << i << " " << beam[j].stack_depth() << " " << beam[j].buffer_length() << endl;
      //sample a sequence of possible actions leading up to the next shift
      Words p_ctx = beam[j].word_pos_context();
      Words r_ctx = beam[j].word_pos_context();
      Words t_ctx = beam[j].pos_context();

      if (beam[j].stack_depth() < 2) {
        //have to shift
        ArcStandardParser& shift_p = beam[j];
        Words w_ctx = beam[j].word_posx_context();
        WordId w = shift_p.next_word();
        double wordp = shift_lm.prob(w, w_ctx); 
        WordId tag = shift_p.next_tag();
        double tagp = tag_lm.prob(tag, t_ctx);

        shift_p.shift();
        shift_p.add_importance_weight(wordp); 
        shift_p.add_importance_weight(tagp); 
        //shift_p.add_particle_weight(wordp); 
        //shift_p.add_particle_weight(tagp); 
      } else {
        //add paths for reduce actions
        double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), r_ctx);
        double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), p_ctx);
        double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), p_ctx);
       
        //add to the beam: replace current position with right arc
        if (beam[j].is_buffer_empty()) {
          if (beam[j].left_arc_valid()) {
            beam.push_back(beam[j]);
            ArcStandardParser& right_p = beam[j]; 
            ArcStandardParser& left_p = beam.back(); 

            left_p.left_arc();
            left_p.add_importance_weight(reducep); 
            //left_p.add_particle_weight(reducep);
            left_p.add_particle_weight(leftarcp);
            right_p.right_arc();
            right_p.add_importance_weight(reducep); 
            //right_p.add_particle_weight(reducep); 
            right_p.add_particle_weight(rightarcp); 
          } else {
            ArcStandardParser& right_p = beam[j]; 
            right_p.right_arc();
            right_p.add_importance_weight(reducep*rightarcp); 
            //right_p.add_particle_weight(reducep*rightarcp); 
          }
        } else {
          beam.push_back(beam[j]);
          if (beam[j].left_arc_valid()) {
            beam.push_back(beam[j]);
            ArcStandardParser& right_p = beam.rbegin()[1]; 
            ArcStandardParser& left_p = beam.back(); 

            left_p.left_arc();
            left_p.set_importance_weight(reducep); 
            //left_p.add_particle_weight(reducep);
            left_p.add_particle_weight(leftarcp);
            right_p.right_arc();
            right_p.set_importance_weight(reducep); 
            //right_p.add_particle_weight(reducep); 
            right_p.add_particle_weight(rightarcp); 
          } else {
            ArcStandardParser& right_p = beam.back(); 
            right_p.right_arc();
            right_p.set_importance_weight(reducep*rightarcp); 
            //right_p.add_particle_weight(reducep*rightarcp); 
          }
          //cout << " + " << beam.size();

          ArcStandardParser& shift_p = beam[j];
          double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), r_ctx);
          WordId w = shift_p.next_word();
          Words w_ctx = beam[j].word_posx_context();
          double wordp = shift_lm.prob(w, w_ctx); 
          WordId tag = shift_p.next_tag();
          double tagp = tag_lm.prob(tag, t_ctx);

          shift_p.shift();
          shift_p.set_importance_weight(wordp); 
          shift_p.add_importance_weight(tagp); 
          //shift_p.add_particle_weight(wordp); 
          //shift_p.add_particle_weight(tagp); 
          shift_p.add_particle_weight(shiftp); 
        }
      }
    }
    //cout << endl;

    //otherwise compare normalized weights 
    sort(beam.begin(), beam.end(), cmp_particle_weights);
    //remove items with worst scores
    for (unsigned j = beam.size()- 1; j >= beam_size; --j)
      beam.pop_back();
  } 
  
  //now sort according to total probability
  sort(beam.begin(), beam.end(), cmp_particle_importance_weights);

  //print parses
  for (unsigned i = 0; (i < 5) && (i < beam.size()); ++i) {
    auto& parser = beam[i];
    parser.print_arcs();
    //cout << parser.arcs_str();
    //cout << parser.actions_str() << endl;

    float dir_acc = (parser.directed_accuracy_count(goldd) + 0.0)/(s.size()-1);
    float undir_acc = (parser.undirected_accuracy_count(goldd) + 0.0)/(s.size()-1);

    cout << "  Dir Accuracy: " << dir_acc;
    //cout << "  UnDir Accuracy: " << undir_acc;
    cout << "  Sample weight: " << (parser.particle_weight()) << endl; // /log(10) 
  }  

  if (beam.size()==0) {
    cout << "no parse found" << endl;
    return ArcStandardParser(s, kORD-1);  
  } else
    return beam[0];
}

//for greedy binary action decisions
ArcStandardParser greedy_parse_sentence(Words s, WxList goldd, Dict& dict, MT19937& eng, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& reduce_lm, PYPLM<kORDER>& arc_lm) {  
  //read in dev sentence; sample actions: for each action, execute and compute next word probability for shift
  cout << "gold arcs: ";
  for (auto d: goldd)
    cout << d << " ";
  cout << endl; 

  ArcStandardParser parser(s, kORDER-1); 

  do {
    Action a = Action::re; //placeholder action
    Words ctx = parser.word_context();
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

  string train_file = "english-wsj/english_wsj_train.conll";
  string test_file = "english-wsj/english_wsj_dev.conll";
  string out_file = "wsj_dev.system.conll";
  //string test_dependencies_file = "conll2007-english/english_ptb_dev.l.conll.dependencies";
/*
  string train_file = "dutch-conll/dutch_alpino_train.conll";
  string test_file = "dutch-conll/dutch_alpino_dev.conll.sentences";
  string test_dependencies_file = "dutch-conll/dutch_alpino_dev.conll.dependencies"; */

  //used for all the models 
  Dict dict("ROOT", "");
  MT19937 eng;

  std::vector<Words> corpussh;
  std::vector<Words> corpusre;
  std::vector<Words> corpusarc;
  std::vector<Words> corpustag;
  
  cerr << "\nStarting training\n";
  auto tr_start = chrono::steady_clock::now();
  //train 
  train_raw(train_file, dict, 2, corpussh, corpusre, corpusarc, corpustag); //extract training examples 

  PYPLM<6> shift_lm(dict.size()+1, 1, 1, 1, 1);
  PYPLM<5> reduce_lm(2, 1, 1, 1, 1); 
  PYPLM<5> arc_lm(2, 1, 1, 1, 1);
  PYPLM<3> tag_lm(dict.pos_size(), 1, 1, 1, 1);

  //samples = 50;  //assume for now
  cerr << "\nTraining word model...\n";
  train_lm(samples, eng, dict, corpussh, shift_lm);  //4*samples
  cerr << "\nTraining shift/reduce model...\n";
  train_lm(samples, eng, dict, corpusre, reduce_lm);
  cerr << "\nTraining arc model...\n";
  train_lm(samples, eng, dict, corpusarc, arc_lm); //6*samples seems to converge slower
  cerr << "\nTraining pos tag model...\n";
  train_lm(samples, eng, dict, corpustag, tag_lm); 

  auto tr_dur = chrono::steady_clock::now() - tr_start;
  cerr << "Training done...time " << chrono::duration_cast<chrono::seconds>(tr_dur).count() << "s\n";

  vector<Words> test;
  vector<Words> testpos;
  vector<WxList> testd;

  std::cerr << "Reading test corpus...\n";
  ReadFromConllFile(test_file, &dict, &test, &testpos, &testd, true);
  std::cerr << "Test corpus size: " << test.size() << " sentences\n";

  vector<unsigned> beam_sizes{1, 2, 4, 8, 16, 32, 64, 128};
  //vector<unsigned> beam_sizes{4};
  //unsigned beam_size = 8;

  for (unsigned beam_size: beam_sizes) {

    AccuracyCounts acc_counts;
    AccuracyCounts acc_counts1;
    cerr << "\nParsing test sentences... (beam size " << beam_size <<  ")\n";
    auto pr_start = chrono::steady_clock::now();
    ofstream outs;
    outs.open(out_file);

    for (unsigned j = 0; j < test.size(); ++j) {
      //if ((test[j].size() <= 10) && is_projective_dependency(testd[j])) {

        //ArcStandardParser g_parser = gold_parse_sentence(test[j], testd[j], shift_lm, reduce_lm, arc_lm);

        //ArcStandardParser b_parser = greedy_parse_sentence(test[j], testd[j], dict, eng, shift_lm, reduce_lm, arc_lm);
        //ArcStandardParser b_parser = particle_par_parse_sentence(test[j], testd[j], beam_size, true, dict, eng, shift_lm, reduce_lm, arc_lm);
        ArcStandardParser b_parser = beam_list_parse_sentence(test[j], testpos[j], testd[j], beam_size, dict, eng, shift_lm, reduce_lm, arc_lm, tag_lm);
        b_parser.count_accuracy(acc_counts, testd[j]);
        //ArcStandardParser bl_parser = beam_parse_align_sentence(test[j], testpos[j], testd[j], beam_size, dict, eng, shift_lm, reduce_lm, arc_lm, tag_lm);
        //bl_parser.count_accuracy(acc_counts1, testd[j]);

        //write output to conll-format file
        for (unsigned i = 1; i < test[j].size(); ++i) 
          outs << i << "\t" << dict.Convert(test[j][i]) << "\t_\t_\t" << dict.ConvertPOS(testpos[j][i]) << "\t_\t" << b_parser.arcs()[i] << "\tROOT\t_\t_\n";

        outs << "\n";
        //cerr << ".";
    //}
    }

    outs.close();
    auto pr_dur = chrono::steady_clock::now() - pr_start;
    cerr << "\nParsing done...time " << chrono::duration_cast<chrono::seconds>(pr_dur).count() << "s\n";
 
    cerr << "Word-aligned beam search" << endl; 
    cerr << "Directed Accuracy: " << acc_counts.directed_accuracy() << endl;
    cerr << "Undirected error rate: " << (1 - acc_counts.undirected_accuracy()) << endl;
    cerr << "Final reduce error rate: " << acc_counts.final_reduce_error_rate() << endl;
    cerr << "Completely correct: " << acc_counts.complete_accuracy() << endl;
    cerr << "Root correct: " << acc_counts.root_accuracy() << endl;
    cerr << "ArcDirection Precision: " << acc_counts.arc_dir_precision() << endl;
    cerr << "Shift recall: " << acc_counts.shift_recall() << endl;
    cerr << "Reduce recall: " << acc_counts.reduce_recall() << endl;   
/*
    cerr << "\nAction-aligned beam search" << endl; 
    cerr << "Directed Accuracy: " << acc_counts1.directed_accuracy() << endl;
    cerr << "Undirected error rate: " << (1 - acc_counts1.undirected_accuracy()) << endl;
    cerr << "Final reduce error rate: " << acc_counts1.final_reduce_error_rate() << endl;
    cerr << "Completely correct: " << acc_counts1.complete_accuracy() << endl;
    cerr << "Root correct: " << acc_counts1.root_accuracy() << endl;
    cerr << "ArcDirection Precision: " << acc_counts1.arc_dir_precision() << endl;
    cerr << "Shift recall: " << acc_counts1.shift_recall() << endl;
    cerr << "Reduce recall: " << acc_counts1.reduce_recall() << endl; */
  }

  return 0;
}

