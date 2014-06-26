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

using namespace oxlm;

/*train, generate from and etsta generative incremental dependency parsing model 
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
  std::string train_file = "english-wsj-nowords/english_wsj_train.conll";
  //std::string train_file = "english-wsj-nowords-nopunc10/english_wsj_train.conll";
  //std::string train_file = "english-wsj/english_wsj_train.conll";
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
  const unsigned kShOrder = 4;
  const unsigned kReOrder = 6;
  const unsigned kArcOrder = 6;
  const unsigned kTagOrder = 4;
      
  PYPLM<kShOrder> shift_lm(dict.size()+1, 1, 1, 1, 1);
  PYPLM<kReOrder> reduce_lm(4, 1, 1, 1, 1); 
  PYPLM<kArcOrder> arc_lm(2, 1, 1, 1, 1);
  PYPLM<kTagOrder> tag_lm(dict.tag_size(), 1, 1, 1, 1);
    
  std::cerr << "\nStarting training\n";
  auto tr_start = std::chrono::steady_clock::now();
      
  bool supervised = true;
  bool with_words = false; 
  bool static_oracle = true;
  bool init = true;  
        
  if (supervised)
    trainSupervisedParser(corpus_sents, corpus_tags, corpus_deps, num_samples, with_words, static_oracle, dict, eng, &shift_lm, &reduce_lm, &tag_lm);
    //trainSupervisedParser(corpus_sents, corpus_tags, corpus_deps, num_samples, with_words, static_oracle, dict, eng, &shift_lm, &reduce_lm, &arc_lm, &tag_lm);
  else 
    trainUnsupervisedParser(corpus_tags, corpus_deps, num_samples, init, dict, eng, &shift_lm, &reduce_lm, &arc_lm, &tag_lm);
    
  auto tr_dur = std::chrono::steady_clock::now() - tr_start;
  std::cerr << "Training done...time " << std::chrono::duration_cast<std::chrono::seconds>(tr_dur).count() << "s\n";  
    
  /*Testing */

  //std::string test_file = "dutch-alpino/dutch_alpino_dev.conll";
  //std::string out_file = "alpino_dev.system.conll";
  
  std::string test_file = "english-wsj-nowords/english_wsj_dev.conll";
  //std::string test_file = "english-wsj/english_wsj_dev.conll";
  //std::string test_file = "english-wsj-nowords-nopunc10/english_wsj_dev.conll";
  std::string out_file = "wsj_dev.system.conll.out";
  
  std::vector<Words> test_sents;
  std::vector<Words> test_tags;
  std::vector<WxList> test_deps;

  std::cerr << "Reading test corpus...\n";
  dict.readFromConllFile(test_file, &test_sents, &test_tags, &test_deps, true);
  std::cerr << "Corpus size: " << test_sents.size() << " sentences\n";
   
  //std::vector<unsigned> beam_sizes{1, 10, 100, 500};
  std::vector<unsigned> beam_sizes{1, 2, 4, 8, 16, 32};
  //unsigned beam_size = 8;
       
  for (unsigned beam_size: beam_sizes) {
      
    AccuracyCounts acc_counts;
    std::cerr << "\nParsing test sentences... (beam size " << beam_size <<  ")\n";
    auto pr_start = std::chrono::steady_clock::now();
    std::ofstream outs;
    outs.open(out_file);

    for (unsigned j = 0; j < test_sents.size(); ++j) {
      ArcList gold_arcs(test_deps[j].size());
      gold_arcs.set_arcs(test_deps[j]);
      //if (gold_arcs.is_projective_dependency())

        ArcEagerParser parser = beamParseSentenceEager(test_sents[j], test_tags[j], gold_arcs, beam_size, with_words, dict, eng, shift_lm, reduce_lm, tag_lm);
        //ArcStandardParser parser = beamParseSentence(test_sents[j], test_tags[j], gold_arcs, beam_size, with_words, dict, eng, shift_lm, reduce_lm, arc_lm, tag_lm);
        //ArcStandardParser parser = beamParseSentence(test_sents[j], test_tags[j], gold_arcs, beam_size, with_words, dict, eng, shift_lm, reduce_lm, tag_lm);
        //ArcStandardParser parser = particleParseSentence(test_sents[j], test_tags[j], gold_arcs, beam_size, resample, with_words, dict, eng, shift_lm, reduce_lm, arc_lm, tag_lm);
        acc_counts.countAccuracy(parser, gold_arcs);

        //write output to conll-format file
        for (unsigned i = 1; i < test_sents[j].size(); ++i) 
          outs << i << "\t" << dict.lookup(test_sents[j][i]) << "\t_\t_\t" << dict.lookupTag(test_tags[j][i]) << "\t_\t" << parser.arcs().at(i) << "\tROOT\t_\t_\n";

        outs << "\n";
    }

    outs.close();
    auto pr_dur = std::chrono::steady_clock::now() - pr_start;
    int milli_sec_dur = std::chrono::duration_cast<std::chrono::milliseconds>(pr_dur).count();
    double sents_per_sec = (static_cast<int>(test_sents.size()) * 1000.0 / milli_sec_dur);
    std::cerr << "\nParsing done...time " << static_cast<int>(milli_sec_dur / 1000.0)
        << "s (" << static_cast<int>(sents_per_sec) << " sentences per second)\n";
 
    std::cerr << "Word-aligned beam search\n"; 
    std::cerr << "Directed Accuracy: " << acc_counts.directed_accuracy() << std::endl;
    std::cerr << "Undirected error rate: " << (1 - acc_counts.undirected_accuracy()) << std::endl;
    std::cerr << "Final reduce error rate: " << acc_counts.final_reduce_error_rate() << std::endl;
    std::cerr << "Completely correct: " << acc_counts.complete_accuracy() << std::endl;
    std::cerr << "Root correct: " << acc_counts.root_accuracy() << std::endl;
    std::cerr << "ArcDirection Precision: " << acc_counts.arc_dir_precision() << std::endl;
    std::cerr << "Shift recall: " << acc_counts.shift_recall() << std::endl;
    std::cerr << "Reduce recall: " << acc_counts.reduce_recall() << std::endl;   
  }  

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

