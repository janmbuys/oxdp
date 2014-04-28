#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "transition_parser.h"
#include "hpyplm/hpyplm.h"
#include "corpus/corpus.h"
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

#define kORDER 3  //default 4
#define nPARTICLES 10

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

  //train a word generation (shift) model
  vector<Words> corpuss;
  set<WordId> vocabs;

  string wc_train_file = "dutch_alpino_train.conll.words.contexts";
  cerr << "Reading corpus...\n";
  ReadFromFile(wc_train_file, &dict, &corpuss, &vocabs);
  cerr << "E-corpus size: " << corpuss.size() << " sentences\t (" << vocabs.size() << " word types)\n";

  PYPLM<kORDER> shift_lm(vocabs.size(), 1, 1, 1, 1);
  
  for (int sample=0; sample < samples; ++sample) {
    for (const auto& s : corpuss) {
      WordId w = s[0];
      ctx = vector<WordId>(s.begin()+1, s.end());
      if (sample > 0) shift_lm.decrement(w, ctx, eng);
      shift_lm.increment(w, ctx, eng);
      //ctx.push_back(w);
    }
    if (sample % 10 == 9) {
      cerr << " [LLH=" << shift_lm.log_likelihood() << "]" << endl;
      if (sample % 30u == 29) shift_lm.resample_hyperparameters(eng);
    } else { cerr << '.' << flush; }
  }

  //train a action generation (shift/reduce) model
  
  vector<Words> corpusr;
  set<WordId> vocabr;

  string ac_train_file = "dutch_alpino_train.conll.actions.contexts";
  cerr << "Reading corpus...\n";
  //will still add actions to the dictionary
  ReadFromActionFile(ac_train_file, &dict, &corpusr, &vocabr);
  cerr << "E-corpus size: " << corpusr.size() << " sentences\t (" << vocabr.size() << " word types)\n";
 
  const unsigned num_actions = 3; 
  PYPLM<kORDER> action_lm(num_actions, 1, 1, 1, 1);
  
  for (int sample=0; sample < samples; ++sample) {
    for (const auto& s : corpusr) {
      WordId w = s[0];  //index over actions
      ctx = vector<WordId>(s.begin()+1, s.end());
      if (sample > 0) action_lm.decrement(w, ctx, eng);
      //else cout << w << " ";
      action_lm.increment(w, ctx, eng);
      //ctx.push_back(w);
    }
      //cout << endl;
    if (sample % 10 == 9) {
      cerr << " [LLH=" << action_lm.log_likelihood() << "]" << endl;
      if (sample % 30u == 29) action_lm.resample_hyperparameters(eng);
    } else { cerr << '.' << flush; }
  }
 
  // test against the gold standard dev
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

  //sample sentences from the trained model
    vector<ArcStandardParser> particles(nPARTICLES, ArcStandardParser(kORDER-1)); 

    for (auto& parser: particles) {
        //one sentence
        double llh = 0;
      
        parser.shift(0);
        
        //it seems that we can be in a terminal configuration along the way, not just once in 
        //a valid derivation?? No, as long as root stays at bottom of the stack
        do {
          //do we need some distribution to determine when to stop generating new words?
          //else just carry on until stack is empty, but this may not be ideal (jet)
          
          Action a; //placeholder action
          Words ctx;
          double lp;
          double wordlp;
          //cout << "word: " << w << endl;
         
          ctx = parser.word_context();
          cout << "context: ";
          for (auto w: ctx)
            cout << w << " ";
          // cout << "context: ";
          //for (unsigned i = 0; i < ctx.size(); ++i) 
          //  cout << ctx[i] << " ";
          //cout << endl; 

            //cout << "word lp: " <<  wordlp << endl;

          if (parser.stack_depth()< 2) {
            a = Action::sh;
          } else {
            double shiftp = action_lm.prob(static_cast<WordId>(Action::sh), ctx);
            double leftarcp = action_lm.prob(static_cast<WordId>(Action::la), ctx);
            double rightarcp = action_lm.prob(static_cast<WordId>(Action::ra), ctx);

            //sample an action
            vector<double> distr = {shiftp, leftarcp, rightarcp};
            multinomial_distribution<double> mult(distr); 
            WordId act = mult(eng);
            lp = log(distr[act]) / log(2);
            //cout << act << " ";
            a = static_cast<Action>(act);
            cout << "(action) " << act << endl;
         } 
         
          if (a == Action::sh) {
            //maybe add check to upper bound sentence length
            if (parser.sentence_length() > 20)
              continue;
              
            ctx = parser.word_context();
            cout << "context: ";
            for (auto w: ctx)
              cout << w << " ";
            
            //sample a word
            WordId w = shift_lm.generate(ctx, vocabs.size(), eng);
            
            cout << "(word) " << w << endl;
            parser.shift(w);
            wordlp = log(shift_lm.prob(w, ctx)) / log(2);
            llh -= wordlp; //at least no oov problem
          } else {
            parser.execute_action(a);
                        
            llh -= lp;
            //cnt++;
          }
        } while (!parser.is_terminal_configuration());
        
        //cout << endl;

        parser.print_sentence(dict);
        parser.print_arcs();
    }
        
 /*
  //read in dev sentence; sample actions: for each action, execute and compute next word probability for shift
    
  string test_file = "dutch_alpino_dev.conll.sentences";
 
  set<WordId> tv;
  vector<Words> test;
  ReadFromFile(test_file, &dict, &test, &tv);  

//  lm.print(cerr);
  double llh = 0;
  unsigned cnt = 0; 
  unsigned oovs = 0;

  for (auto& s : test) {
    //extend to a number of particles (parser for each)
    cout << "(" << s.size() << ") " << endl;
    vector<ArcStandardParser> particles(nPARTICLES, ArcStandardParser(s, kORDER-1)); // is this ok?
    //For now, construct the particles serially
    for (auto& parser: particles) {

        //ArcStandardParser parser(s, kORDER-1);
        
        // cout << "sentence: ";
        //for (unsigned i = 0; i < s.size(); ++i) 
       //   cout << s[i] << " ";
       // cout << endl;  

        //for (unsigned i = 0; i < s.size(); ++i) {
        while (!parser.is_buffer_empty()) {
          Action a = Action::re; //placeholder action
          WordId w = parser.next_word();
          Words ctx;
          double wordlp;
          //cout << "word: " << w << endl;
         
          //sample action while action<>shift
          //do {
          while (a != Action::sh) {
            ctx = parser.word_context();
            // cout << "context: ";
            //for (unsigned i = 0; i < ctx.size(); ++i) 
            //  cout << ctx[i] << " ";
            //cout << endl; 

            wordlp = log(shift_lm.prob(w, ctx)) / log(2);
            //cout << "word lp: " <<  wordlp << endl;

            if ((parser.stack_depth()< kORDER-1) && !parser.is_buffer_empty()) {
              a = Action::sh;
              parser.execute_action(a);
            } else {
              double shiftp = action_lm.prob(static_cast<WordId>(Action::sh), ctx);
              double leftarcp = action_lm.prob(static_cast<WordId>(Action::la), ctx);
              double rightarcp = action_lm.prob(static_cast<WordId>(Action::ra), ctx);
              
              if (shiftp==0 || leftarcp==0 || rightarcp==0)
                cout << "Probs: " << shiftp << " " << leftarcp << " " << rightarcp << endl;

              vector<double> distr = {shiftp, leftarcp, rightarcp};
              multinomial_distribution<double> mult(distr); 
              WordId act = mult(eng);
              //cout << act << endl;
              a = static_cast<Action>(act);
              if (!parser.execute_action(a)) {
                a = Action::re;
                continue;
              }

              double lp = log(distr[act]) / log(2);
              llh -= lp;
              cnt++;
            }
          } 

          //shift already executed above
          if (vocabs.count(w) == 0) {
              //cout << "OOV: " << w << endl;
            //how should this be handled properly? we do get 0 prob here...
            ++oovs;
            //lp = 0;
          } else
            //shift probability
            llh -= wordlp;
        }
        //cout << "shift completed" << endl;

        //actions to empty the stack
        while (!parser.is_terminal_configuration()) {
                // action_seq.size() < 2*s.size()) {
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
            
              vector<double> distr = {leftarcp, rightarcp};
              multinomial_distribution<double> mult(distr); 
              WordId act = mult(eng) + 1;
              //cout << act << endl;
              Action a = static_cast<Action>(act);
              parser.execute_action(a);
          
              double lp = log(distr[act]) / log(2); //should be normalized prob?
              llh -= lp;
              cnt++; 
            }
        }
        
        cnt++; //0 ll for invisible end-of-sentence marker
        //print action sequence
        cout << parser.actions_str() << endl;
        parser.print_arcs();
    }
  }

  //cnt -= oovs;
  cerr << "One sequence sampled score:" << endl;
  cerr << "  Log_10 prob: " << (-llh * log(2) / log(10)) << endl;
  cerr << "        Count: " << cnt << endl;
  cerr << "         OOVs: " << oovs << endl;
  cerr << "Cross-entropy: " << (llh / cnt) << endl;
  cerr << "   Perplexity: " << pow(2, llh / cnt) << endl;
  return 0;
            */  
}

