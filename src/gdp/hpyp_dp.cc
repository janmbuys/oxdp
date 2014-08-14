#include <iostream>
#include <cstdlib>
#include <chrono>

#include "pyp/crp.h"
#include "hpyplm/hpyplm.h"
#include "corpus/corpus.h"
#include "transition_parser.h"
#include "eisner_parser.h"
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
  std::string train_file = "english-wsj-conll07/english_wsj_train.conll";
  //std::string train_file = "english-wsj-conll07-nowords/english_wsj_train.conll";
  //std::string train_file = "english-wsj-conll07-nowords-nopunc/english_wsj_train.conll";
  //std::string train_file = "english-wsj-nowords-nopunc/english_wsj_train.conll";
  //std::string train_file = "dutch-alpino/dutch_alpino_train.conll";

  MT19937 eng;
  
  bool supervised = true;
  bool semisupervised = false;
  bool with_words = false; 
  bool static_oracle = true;
  bool arceager = false;
  //bool eisner = true;
  bool init = false;  

  //Dict dict("ROOT", "");
  Dict dict(true, arceager);

  std::vector<Words> corpus_sents;
  std::vector<Words> corpus_tags;
  std::vector<WxList> corpus_deps;
  
  std::cerr << "Reading training corpus...\n";
  dict.readFromConllFile(train_file, &corpus_sents, &corpus_tags, &corpus_deps, false);
  std::cerr << "Corpus size: " << corpus_sents.size() << " sentences\t (" << dict.size() << " word types, " << dict.tag_size() << " tags)\n";

  //define pyp models with their orders
  const unsigned kShOrder = 6; //5 
  //const unsigned kArcOrder = 4;
  const unsigned kReOrder = 9; //7, 8, 9, 6, 11
  const unsigned kTagOrder = 9; //5, 9, 5, 6, 9
  
  unsigned num_transitions = 3;
  if (arceager)
    num_transitions = 4;

  PYPLM<kShOrder> shift_lm(dict.size()+1, 1, 1, 1, 1);
  //PYPLM<kArcOrder> arc_lm(2, 1, 1, 1, 1);
  PYPLM<kTagOrder> tag_lm(dict.tag_size(), 1, 1, 1, 1);
  PYPLM<kReOrder> reduce_lm(num_transitions, 1, 1, 1, 1); 
  
  std::cerr << "\nStarting training\n";
  auto tr_start = std::chrono::steady_clock::now();
      
  //TODO train and test for eisner model

   if (semisupervised)
    trainSemisupervisedParser(corpus_sents, corpus_tags, corpus_deps, num_samples, with_words, arceager, static_oracle, dict, eng, &shift_lm, &reduce_lm, &tag_lm);
  else if (supervised)
    trainSupervisedParser(corpus_sents, corpus_tags, corpus_deps, num_samples, with_words, arceager, static_oracle, dict, eng, &shift_lm, &reduce_lm, &tag_lm);
  else 
    trainUnsupervisedParser(corpus_tags, corpus_deps, num_samples, init, arceager, dict, eng, &shift_lm, &reduce_lm, &tag_lm);
    
  auto tr_dur = std::chrono::steady_clock::now() - tr_start;
  std::cerr << "Training done...time " << std::chrono::duration_cast<std::chrono::seconds>(tr_dur).count() << "s\n";  
 
  /*Testing 

  std::string test_file;
  //std::string test_file = "dutch-alpino/dutch_alpino_dev.conll";
  //std::string test_file = "english-wsj-partlex/english_wsj_dev.conll";
  //std::string test_file = "english-wsj/english_wsj_dev.conll";
  //std::string test_file = "english-wsj-nowords/english_wsj_dev.conll";
  
  //test_file = "english-wsj-nowords-nopunc10/english_wsj_dev.conll";
  test_file = "english-wsj-conll07-nowords-nopunc10/english_wsj_dev.conll";
  evaluate(test_file, arceager, with_words, dict, eng, shift_lm, reduce_lm, tag_lm); 
 
  test_file = "english-wsj-conll07-nowords-nopunc20/english_wsj_dev.conll";
  evaluate(test_file, arceager, with_words, dict, eng, shift_lm, reduce_lm, tag_lm); 
 
  //std::string test_file = "english-wsj-conll07-nowords/english_wsj_dev.conll";
  test_file = "english-wsj-conll07-nowords-nopunc/english_wsj_dev.conll";
  evaluate(test_file, arceager, with_words, dict, eng, shift_lm, reduce_lm, tag_lm); */
  
  std::string test_file = "english-wsj-conll07/english_wsj_dev.conll";
  evaluate(test_file, arceager, with_words, dict, eng, shift_lm, reduce_lm, tag_lm); 
  
  /* Generating 
  //sample sentences from the trained model
  const int kNumGenerations = 100;
  unsigned sentence_limit = 100;
  std::vector<unsigned> length_dist;

  if (arceager) {
    std::vector<ArcEagerParser> particles(kNumGenerations, ArcEagerParser()); 

    for (auto& parser: particles) {
      parser = generateEagerSentence(dict, eng, sentence_limit, shift_lm, reduce_lm, tag_lm);  

      if (parser.sentence_length() < sentence_limit) {
        std::cout << parser.sentence_length() << " ";
        parser.print_sentence(dict);
        parser.print_tags(dict);
        length_dist.push_back(parser.sentence_length());
        //cout << parser.actions_str() << endl;
        parser.print_arcs();
        std::cout << "\n\n";   
      }
    }
  } else {
    std::vector<ArcStandardParser> particles(kNumGenerations, ArcStandardParser()); 

    for (auto& parser: particles) {
      parser = generateSentence(dict, eng, sentence_limit, shift_lm, reduce_lm, tag_lm);  
      
      if (!parser.is_complete_parse()) {
        std::cout << "INCOMPLETE\n";
        continue;
      }

      if (parser.sentence_length() < sentence_limit) {
        std::cout << parser.sentence_length() << " ";
        parser.print_sentence(dict);
        parser.print_tags(dict);
        //cout << parser.actions_str() << endl;
        parser.print_arcs();
        std::cout << "\n\n";   
      }
      
      length_dist.push_back(parser.sentence_length());
    }
  }

  std::cout << "Length distribution:\n";
  std::sort(length_dist.begin(), length_dist.end());
  for (auto l: length_dist)
    std::cout << l << " ";  // */
   
  return 0;
}

