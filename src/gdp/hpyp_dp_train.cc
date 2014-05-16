#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "hpyp_dp_train.h"
#include "hpyplm/hpyplm.h"
#include "pyp/random.h"

#define kORDER 3  //default 4

using namespace std;
using namespace oxlm;

/*train a generative dependency parsing model using given context vectors;
 */
int main(int argc, char** argv) {
  //for now, hard_code filenames
  if (argc != 2) {
    cerr << argv[0] << " <nsamples>\n\nEstimate a " 
         << kORDER << "-gram HPYP LM and report perplexity\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }

  MT19937 eng;
  Dict dict("ROOT", "", true); //used for all the models 
  int samples = atoi(argv[1]);
  //const unsigned num_word_types = 26502; //hardcoded to save trouble (dutch)
  const unsigned num_word_types = 56574; //hardcoded to save trouble (english)

  string train_file = "conll2007-english/english_ptb_train.conll";
  //string train_file = "dutch_alpino_train.conll";
  set<WordId> vocabs;
  std::vector<Words> corpussh;
  std::vector<Words> corpusre;
  std::vector<Words> corpusarc;
  
  PYPLM<kORDER> shift_lm(num_word_types, 1, 1, 1, 1); //next word
  PYPLM<kORDER> reduce_lm(2, 1, 1, 1, 1); //shift/reduce
  PYPLM<kORDER> arc_lm(2, 1, 1, 1, 1); //left/right arc

  train_raw(train_file, dict, vocabs, corpussh, corpusre, corpusarc); //extract training examples 
  //print out constructed corpora
  for (auto& ngram: corpusarc) {
    cout << dict.Convert(ngram[1]) << " " << dict.Convert(ngram[2]) << " " << ngram[0] << endl;    
  }

  cerr << "\nTraining word model...\n";
   train_lm(samples, eng, dict, corpussh, shift_lm);
  cerr << "\nTraining shift/reduce model...\n";
  train_lm(samples, eng, dict, corpusre, reduce_lm);
  cerr << "\nTraining arc model...\n";
  train_lm(samples, eng, dict, corpusarc, arc_lm);    
 
  //training for 3-way decision
/*
  const unsigned num_actions = 3; 
  std::vector<Words> corpusre;
  
  PYPLM<kORDER> shift_lm(num_word_types, 1, 1, 1, 1);  
  PYPLM<kORDER> action_lm(num_actions, 1, 1, 1, 1);
  
  //training 
  
  train_raw(train_file, dict, vocabs, corpussh, corpusre); //extract training examples 
  //train_shift(samples, eng, dict, corpussh, shift_lm);
  //train_action(samples, eng, dict, corpusre, action_lm);    
  
  train_lm(samples, eng, dict, corpussh, shift_lm);
  train_lm(samples, eng, dict, corpusre, action_lm);    */

//training from context files

  //string wc_train_file = "dutch_alpino_train.conll.words.contexts";
  ///string ac_train_file = "dutch_alpino_train.conll.actions.contexts";
  //train_shift(samples, wc_train_file, eng, dict, vocabs, shift_lm);
  //train_action(samples, ac_train_file, eng, dict, vocabr, action_lm);
}

