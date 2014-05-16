#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "transition_parser.h"
#include "hpyp_dp_train.h"
#include "hpyplm/hpyplm.h"
#include "pyp/random.h"

#define kORDER 3  //default 4
#define nPARTICLES 100

using namespace std;
using namespace oxlm;

//For three-way decision
void generate_sentence(ArcStandardParser& parser, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& action_lm, int vocab_size, MT19937& eng) {
  int rw_size = 7;
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
      //cout << "(la: " << leftarcp << " ra: " << rightarcp << ") ";

      //sample an action
      vector<double> distr = {leftarcp, rightarcp};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng) + 1;
      lp = log(distr[act-1]); // / log(2);
      a = static_cast<Action>(act);
    } else {
      double shiftp = action_lm.prob(static_cast<WordId>(Action::sh), ctx);
      double leftarcp = action_lm.prob(static_cast<WordId>(Action::la), ctx);
      double rightarcp = action_lm.prob(static_cast<WordId>(Action::ra), ctx);
      //cout << "(sh: " << shiftp << " la: " << leftarcp << " ra: " << rightarcp << ") ";

      //sample an action
      vector<double> distr = {shiftp, leftarcp, rightarcp};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
      lp = log(distr[act]); // / log(2);
      //cout << act << " ";
      a = static_cast<Action>(act);
      //cout << "(action) " << act << endl;
    } 
     
    if (a == Action::sh) {
      //maybe add check to upper bound sentence length
      /* if (parser.sentence_length() > 20) {
        cerr << "sentence length limit to 20" << endl;
        terminate_shift = true;
        continue;
      } */
    
      //cerr << vocab_size;      
      ctx = parser.word_context();
      //cerr << " context: ";
      //for (auto w: ctx)
      //  cerr << w << " ";
      //cerr << endl;

      //sample a word
      WordId w = shift_lm.generate(ctx, vocab_size, rw_size, eng);
      wordlp = log(shift_lm.prob(w, ctx)); // / log(2);
        
      //cout << "(word) " << w << endl;
      parser.shift(w);
      llh -= wordlp; //at least no oov problem
    } else {
      parser.execute_action(a);
                    
      llh -= lp;
      //cnt++;
    }
  } while (!parser.is_terminal_configuration() && !terminate_generation);
    
    //cout << endl;
}

//For binary decisions
void generate_sentence(ArcStandardParser& parser, PYPLM<kORDER>& shift_lm, PYPLM<kORDER>& reduce_lm, PYPLM<kORDER>& arc_lm, int vocab_size, MT19937& eng) {
  int rw_size = 7;
  bool terminate_generation = false;
  bool terminate_shift = false;
  parser.shift(0);
    
  do {
    Action a; //placeholder action
    Words ctx; 
    
    ctx = parser.word_context(); 
    if (parser.stack_depth()< 2) {
      a = Action::sh;
    } else if (parser.sentence_length() > 100) {
        // check to upper bound sentence length
        if (!terminate_shift)
          cerr << " LENGTH LIMITED ";
        terminate_shift = true;
        a = Action::re;
    }  
    else {
      double shiftp = reduce_lm.prob(static_cast<WordId>(Action::sh), ctx);
      double reducep = reduce_lm.prob(static_cast<WordId>(Action::re), ctx);
      cout << "(sh: " << shiftp << " re: " << reducep << ") ";

      //sample an action
      vector<double> distr = {shiftp, reducep};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
      parser.add_particle_weight(distr[act]);
      
      if (act==0) {
        a = Action::sh;
      } else {
        a = Action::re; 
      }
    } 

    if (a == Action::sh) {
      ctx = parser.word_context();
      //sample a word
      WordId w = shift_lm.generate(ctx, vocab_size, rw_size, eng);
      double wordp = shift_lm.prob(w, ctx); 

      parser.shift(w);
      parser.add_particle_weight(wordp);
    } else if (a == Action::re) {
      double leftarcp = arc_lm.prob(static_cast<WordId>(Action::la), ctx);
      double rightarcp = arc_lm.prob(static_cast<WordId>(Action::ra), ctx);
      cout << "(la: " << leftarcp << " ra: " << rightarcp << ") ";

      //sample arc direction
      vector<double> distr = {leftarcp, rightarcp};
      multinomial_distribution<double> mult(distr); 
      WordId act = mult(eng);
      a = static_cast<Action>(act+1);
      parser.add_particle_weight(distr[act]);

      //may need to enforce the la constraint here
      parser.execute_action(a);
    }
  } while (!parser.is_terminal_configuration() && !terminate_generation);
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
  //const unsigned num_actions = 3; 
  const unsigned num_word_types = 56574; //hardcoded to save trouble
  string train_file = "conll2007-english/english_ptb_train.conll";
  
  //const unsigned num_word_types =  26502; //hardcoded to save trouble
  //string train_file = "dutch_alpino_train.conll";

  set<WordId> vocabs;
  std::vector<Words> corpussh;
  std::vector<Words> corpusre;
  std::vector<Words> corpusarc;
  
  PYPLM<kORDER> shift_lm(num_word_types, 1, 1, 1, 1); //next word
  //for two-way decisions
  
  PYPLM<kORDER> reduce_lm(2, 1, 1, 1, 1); //shift/reduce
  PYPLM<kORDER> arc_lm(2, 1, 1, 1, 1); //left/right arc

  train_raw(train_file, dict, vocabs, corpussh, corpusre, corpusarc); //extract training examples 
  train_lm(samples, eng, dict, corpussh, shift_lm);
  train_lm(samples, eng, dict, corpusre, reduce_lm);  
  train_lm(samples, eng, dict, corpusarc, arc_lm);   


  //for three-way decision
  //train
  //PYPLM<kORDER> action_lm(3, 1, 1, 1, 1);
  //train_raw(train_file, dict, vocabs, corpussh, corpusre); //extract training examples 
  //train_lm(samples, eng, dict, corpussh, shift_lm);
  //train_lm(samples, eng, dict, corpusre, action_lm);  

  //sample sentences from the trained model
  vector<ArcStandardParser> particles(nPARTICLES, ArcStandardParser(kORDER-1)); 

  for (auto& parser: particles) {
    //generate_sentence(parser, shift_lm, action_lm, vocabs.size(), eng);  
    generate_sentence(parser, shift_lm, reduce_lm, arc_lm, vocabs.size(), eng);  

    cout << parser.sentence_length() << ": ";
    parser.print_sentence(dict);
    cout << parser.actions_str() << endl;
    parser.print_arcs();
  }
}

