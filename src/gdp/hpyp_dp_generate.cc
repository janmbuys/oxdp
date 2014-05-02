#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "transition_parser.h"
#include "hpyp_dp_train.h"
#include "hpyplm/hpyplm.h"
#include "pyp/random.h"

#define kORDER 3  //default 4
#define nPARTICLES 10

using namespace std;
using namespace oxlm;

void generate_sentence(ArcStandardParser& parser, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& action_lm, int vocab_size, MT19937& eng) {
    double llh = 0;
    bool terminate_generation = false;
    bool terminate_shift = false;
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
      //cerr << "context: ";
      //for (auto w: ctx)
      //  cerr << w << " ";
      //cerr << endl; 

        //cout << "word lp: " <<  wordlp << endl;

      if (parser.stack_depth()< 2) {
        a = Action::sh;
      } else if (terminate_shift) {
        double leftarcp = action_lm.prob(static_cast<WordId>(Action::la), ctx);
        double rightarcp = action_lm.prob(static_cast<WordId>(Action::ra), ctx);

        //sample an action
        vector<double> distr = {leftarcp, rightarcp};
        multinomial_distribution<double> mult(distr); 
        WordId act = mult(eng) + 1;
        lp = log(distr[act-1]) / log(2);
        a = static_cast<Action>(act);
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
        //cout << "(action) " << act << endl;
     } 
     
     if (a == Action::sh) {
        //maybe add check to upper bound sentence length
        if (parser.sentence_length() > 20) {
           cerr << "sentence length limit to 20" << endl;
           terminate_shift = true;
           continue;
        }
    
        //cerr << vocab_size;      
        ctx = parser.word_context();
        //cerr << " context: ";
        //for (auto w: ctx)
        //  cerr << w << " ";
        //cerr << endl;

        //sample a word
        WordId w = shift_lm.generate(ctx, vocab_size, eng);
        
        //cout << "(word) " << w << endl;
        parser.shift(w);
        wordlp = log(shift_lm.prob(w, ctx)) / log(2);
        llh -= wordlp; //at least no oov problem
      } else {
        parser.execute_action(a);
                    
        llh -= lp;
        //cnt++;
      }
    } while (!parser.is_terminal_configuration() && !terminate_generation);
    
    //cout << endl;
}

/*train a generative dependency parsing model using given context vectors;
 * generate (unbiased) sample sentences and dependencies
 */
int main(int argc, char** argv) {
  //for now, hard_code filenames
  if (argc != 2) {
    cerr << argv[0] << " <nsamples>\n\nEstimate a " 
         << kORDER << "-gram HPYP LM and report perplexity\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }

  //training 

  int samples = atoi(argv[1]);
  MT19937 eng;
  Dict dict("ROOT", "", true); //used for all the models 
  const unsigned num_actions = 3; 
  const unsigned num_word_types = 26502; //hardcoded to save trouble

  set<WordId> vocabs;
  //set<WordId> vocabr;
  std::vector<Words> corpussh;
  std::vector<Words> corpusre;
  
  string train_file = "dutch_alpino_train.conll";
  //string wc_train_file = "dutch_alpino_train.conll.words.contexts";
  //string ac_train_file = "dutch_alpino_train.conll.actions.contexts";
 
  PYPLM<kORDER> shift_lm(num_word_types, 1, 1, 1, 1);
  PYPLM<kORDER> action_lm(num_actions, 1, 1, 1, 1);
 
  //train
  train_raw(train_file, dict, vocabs, corpussh, corpusre); //extract training examples 
  train_shift(samples, eng, dict, corpussh, shift_lm);
  train_action(samples, eng, dict, corpusre, action_lm);

  //train_shift(samples, wc_train_file, eng, dict, vocabs, shift_lm);
  //train_action(samples, ac_train_file, eng, dict, vocabr, action_lm);

  //sample sentences from the trained model
  vector<ArcStandardParser> particles(nPARTICLES, ArcStandardParser(kORDER-1)); 

  for (auto& parser: particles) {
    generate_sentence(parser, shift_lm, action_lm, vocabs.size(), eng);  

    parser.print_sentence(dict);
    parser.print_arcs();
  }
}

