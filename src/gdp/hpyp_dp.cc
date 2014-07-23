#include <iostream>
#include <cstdlib>
#include <chrono>

#include "pyp/crp.h"
#include "hpyplm/hpyplm.h"
#include "corpus/corpus.h"
#include "transition_parser.h"
#include "pyp/random.h"
#include "hpyp_dp_parse.h"
#include "hpyp_dp_train.h"
#include "hpyp_dp_test.h"

using namespace oxlm;

/*train, generate from and test a generative incremental dependency parsing model 
 */
int main(int argc, char** argv) {
  //for now, hard_code filenames
  //TODO include args for train, generate and test, and filenames
  if (argc != 2) {
      std::cerr << argv[0] << " <nsamples>\n";
    return 1;
  }

  /*Training */

  int num_samples = atoi(argv[1]);
  //std::string train_file = "english-wsj-partlex/english_wsj_train.conll";
  //std::string train_file = "english-wsj/english_wsj_train.conll";
  //std::string train_file = "english-wsj-nowords/english_wsj_train.conll";
  std::string train_file = "english-wsj-nowords-nopunc10/english_wsj_train.conll";
  //std::string train_file = "dutch-alpino/dutch_alpino_train.conll";

  Dict dict("ROOT", "");
  MT19937 eng;
  
  std::vector<Words> corpus_sents;
  std::vector<Words> corpus_tags;
  std::vector<WxList> corpus_deps;
  
  std::cerr << "Reading training corpus...\n";
  dict.readFromConllFile(train_file, &corpus_sents, &corpus_tags, &corpus_deps, false);
  std::cerr << "Corpus size: " << corpus_sents.size() << " sentences\t (" << dict.size() << " word types, " << dict.tag_size() << " tags)\n";

  //define pyp models with their orders
  const unsigned kShOrder = 8; //5, 6 
  const unsigned kArcOrder = 4;
  const unsigned kReOrder = 6; //8
  const unsigned kTagOrder = 5; //9, 8
  
  //remember to update vocab sizes!  
  PYPLM<kShOrder> shift_lm(dict.size()+1, 1, 1, 1, 1);
  PYPLM<kReOrder> reduce_lm(3, 1, 1, 1, 1); 
  PYPLM<kArcOrder> arc_lm(2, 1, 1, 1, 1);
  PYPLM<kTagOrder> tag_lm(dict.tag_size(), 1, 1, 1, 1);
  
  std::cerr << "\nStarting training\n";
  auto tr_start = std::chrono::steady_clock::now();
                    
  bool supervised = false;
  bool with_words = false; 
  bool static_oracle = true;
  bool arceager = false;
  bool init = true;  

  if (supervised)
    trainSupervisedParser(corpus_sents, corpus_tags, corpus_deps, num_samples, with_words, arceager, static_oracle, dict, eng, &shift_lm, &reduce_lm, &tag_lm);
  else 
    trainUnsupervisedParser(corpus_tags, corpus_deps, num_samples, init, dict, eng, &shift_lm, &reduce_lm, &tag_lm);
    
  auto tr_dur = std::chrono::steady_clock::now() - tr_start;
  std::cerr << "Training done...time " << std::chrono::duration_cast<std::chrono::seconds>(tr_dur).count() << "s\n";  
    
  /*Testing */

  //std::string test_file = "dutch-alpino/dutch_alpino_dev.conll";
  //std::string test_file = "english-wsj-partlex/english_wsj_dev.conll";
  //std::string test_file = "english-wsj/english_wsj_dev.conll";
  //std::string test_file = "english-wsj-nowords/english_wsj_dev.conll";
  
  std::string test_file = "english-wsj-nowords-nopunc/english_wsj_dev.conll";
  evaluate(test_file, arceager, with_words, dict, eng, shift_lm, reduce_lm, tag_lm);
  
  test_file = "english-wsj-nowords-nopunc10/english_wsj_dev.conll";
  evaluate(test_file, arceager, with_words, dict, eng, shift_lm, reduce_lm, tag_lm); 
 
  /* Generating 
  //sample sentences from the trained model
  const int kNumGenerations = 100;
  std::vector<ArcStandardParser> particles(kNumGenerations, ArcStandardParser()); 

  for (auto& parser: particles) {
    parser = generateSentence(dict, eng, shift_lm, reduce_lm, arc_lm, tag_lm);  

    std::cout << parser.sentence_length() << " ";
    parser.print_sentence(dict);
    parser.print_tags(dict);
    //length_dist.push_back(parser.sentence_length());
    //cout << parser.actions_str() << endl;
    parser.print_arcs();
    std::cout << std::endl;   
  }

  //sort(length_dist.begin(), length_dist.end());
  //for (auto l: length_dist)
  //  cout << l << " ";   
  */ 

  return 0;
}

