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
  //num_samples = 50;  
  //std::string train_file = "english-wsj-nowords/english_wsj_train.conll";
  std::string train_file = "english-wsj/english_wsj_train.conll";
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
  const unsigned kReOrder = 5;
  const unsigned kArcOrder = 5;
  const unsigned kTagOrder = 4;
    
  PYPLM<kShOrder> shift_lm(dict.size()+1, 1, 1, 1, 1);
  PYPLM<kReOrder> reduce_lm(2, 1, 1, 1, 1); 
  PYPLM<kArcOrder> arc_lm(2, 1, 1, 1, 1);
  PYPLM<kTagOrder> tag_lm(dict.tag_size(), 1, 1, 1, 1);

  std::cerr << "\nStarting training\n";
  auto tr_start = std::chrono::steady_clock::now();

  //TODO initialize tag_lm with a sequential trigram model
  //std::vector<Words> init_examples_tag;
  
  //trainPYPModel(num_samples, eng, init_examples_tag, &tag_lm); 
    
  //trainUnsupervised(corpus_sents, corpus_tags, corpus_deps, num_samples, dict, eng, &shift_lm, &reduce_lm, &arc_lm, &tag_lm);

  //Supervised training
  std::vector<Words> examples_sh;
  std::vector<Words> examples_re;
  std::vector<Words> examples_arc;
  std::vector<Words> examples_tag;
  
  constructTrainExamples(corpus_sents, corpus_tags, corpus_deps, &examples_sh, &examples_re, &examples_arc, &examples_tag);

  std::cerr << "\nTraining word model...\n";
  trainPYPModel(num_samples, eng, examples_sh, &shift_lm);  //4*samples
  std::cerr << "\nTraining shift/reduce model...\n";
  trainPYPModel(num_samples, eng, examples_re, &reduce_lm);
  std::cerr << "\nTraining arc model...\n";
  trainPYPModel(num_samples, eng, examples_arc, &arc_lm); //6*samples seems to converge slower
  std::cerr << "\nTraining pos tag model...\n";
  trainPYPModel(num_samples, eng, examples_tag, &tag_lm); 


  auto tr_dur = std::chrono::steady_clock::now() - tr_start;
  std::cerr << "Training done...time " << std::chrono::duration_cast<std::chrono::seconds>(tr_dur).count() << "s\n";

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

  /*Testing */ 

  //std::string test_file = "dutch-alpino/dutch_alpino_dev.conll";
  //std::string out_file = "alpino_dev.system.conll";
  
  std::string test_file = "english-wsj/english_wsj_dev.conll";
  //std::string test_file = "english-wsj-no-words/english_wsj_dev.conll";
  std::string out_file = "wsj_dev.system.conll";
  
  std::vector<Words> test_sents;
  std::vector<Words> test_tags;
  std::vector<WxList> test_deps;

  std::cerr << "Reading test corpus...\n";
  dict.readFromConllFile(test_file, &test_sents, &test_tags, &test_deps, true);
  std::cerr << "Corpus size: " << test_sents.size() << " sentences\n";
   
  //std::vector<unsigned> beam_sizes{1, 10, 50, 100, 200, 500};
  std::vector<unsigned> beam_sizes{1, 2, 4, 8, 16, 32, 64, 128};
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
      //if ((test_sents[j].size() <= 10) && gold_arcs.is_projective_dependency()) {

        //ArcStandardParser parser = particleParseSentence(test_sents[j], test_tags[j], gold_arcs, beam_size, true, dict, eng, shift_lm, reduce_lm, arc_lm, tag_lm);
        ArcStandardParser parser = beamParseSentence(test_sents[j], test_tags[j], gold_arcs, beam_size, dict, eng, shift_lm, reduce_lm, arc_lm, tag_lm);
        acc_counts.countAccuracy(parser, gold_arcs);

        //write output to conll-format file
        for (unsigned i = 1; i < test_sents[j].size(); ++i) 
          outs << i << "\t" << dict.lookup(test_sents[j][i]) << "\t_\t_\t" << dict.lookupTag(test_tags[j][i]) << "\t_\t" << parser.arcs().at(i) << "\tROOT\t_\t_\n";

        outs << "\n";
        //std::cerr << ". ";
    //}
    }

    outs.close();
    auto pr_dur = std::chrono::steady_clock::now() - pr_start;
    std::cerr << "\nParsing done...time " << std::chrono::duration_cast<std::chrono::seconds>(pr_dur).count() << "s\n";
 
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

  return 0;
}

