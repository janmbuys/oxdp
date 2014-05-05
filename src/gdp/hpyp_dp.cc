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

  const unsigned num_actions = 3; 
  const unsigned num_word_types = 26502; //hardcoded to save trouble

  set<WordId> vocabs;
  //set<WordId> vocabr;
  std::vector<Words> corpussh;
  std::vector<Words> corpusre;
  
  string train_file = "dutch_alpino_train.conll";

  PYPLM<kORDER> shift_lm(num_word_types, 1, 1, 1, 1);
  PYPLM<kORDER> action_lm(num_actions, 1, 1, 1, 1);
 
  //train
  train_raw(train_file, dict, vocabs, corpussh, corpusre); //extract training examples 
  train_shift(samples, eng, dict, corpussh, shift_lm);
  train_action(samples, eng, dict, corpusre, action_lm);
       
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
  double llh = 0;
  unsigned cnt = 0; 
  unsigned oovs = 0;

  for (unsigned j = 0; j < test.size(); ++j) {
    //auto& s : test) {
    //read in dev sentence; sample actions: for each action, execute and compute next word probability for shift
    Words s = test[j];
    WxList goldd = testd[j];
    //TODO print gold dependencies, get accuracies
    cout << "gold arcs: ";
    for (auto d: goldd)
      cout << d << " ";
    cout << endl; 

    //extend to a number of particles (parser for each)
    //cout << "(" << s.size() << ") " << endl;
    vector<ArcStandardParser> particles(nPARTICLES, ArcStandardParser(s, kORDER-1)); // is this ok?
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

  cnt -= oovs;

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

