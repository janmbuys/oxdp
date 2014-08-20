#ifndef _GDP_HPYP_DP_TEST_H_
#define _GDP_HPYP_DP_TEST_H_

#include <iostream>
#include <cstdlib>
#include <chrono>

#include "transition_parser.h"
#include "eisner_parser.h"
#include "hpyp_dp_parse.h"
#include "hpyplm/hpyplm.h"
#include "pyp/random.h"
#include "pyp/crp.h"

namespace oxlm {

template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
double evaluate(std::string test_file, bool arceager, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>& shift_lm, PYPLM<kReduceOrder>& reduce_lm, PYPLM<kTagOrder>& tag_lm) {
  std::vector<Words> test_sents;
  std::vector<Words> test_tags;
  std::vector<WxList> test_deps;

  std::cerr << "Reading test corpus...\n";
  dict.read_from_conll_file(test_file, &test_sents, &test_tags, &test_deps, true);
  std::cerr << "Corpus size: " << test_sents.size() << " sentences\n";
      
  //std::string out_file = "alpino_dev.system.conll";
  //std::string out_file = "wsj_dev.system.conll.out";
  //std::vector<unsigned> beam_sizes{1};
  //std::vector<unsigned> beam_sizes{2};
  std::vector<unsigned> beam_sizes{1, 2, 4, 8, 16, 32, 64};
  double acc = 0.0;

  for (unsigned beam_size: beam_sizes) {
    AccuracyCounts acc_counts;
    std::cerr << "\nParsing test sentences... (beam size " << beam_size <<  ")\n";
    //std::cout << "\nParsing test sentences... (beam size " << beam_size <<  ")\n";
    auto pr_start = std::chrono::steady_clock::now();
    //std::ofstream outs;
    //outs.open(out_file);
            
    for (unsigned j = 0; j < test_sents.size(); ++j) {
      ArcList gold_arcs(test_deps[j].size());
      gold_arcs.set_arcs(test_deps[j]);

      //auto parser;
      if (arceager) {
        ArcEagerParser parser = beamParseSentenceEager(test_sents[j], test_tags[j], gold_arcs, beam_size, with_words, dict, eng, shift_lm, reduce_lm, tag_lm);
        //ArcEagerParser parser = particleEagerParseSentence(test_sents[j], test_tags[j], gold_arcs, beam_size, resample, true, with_words, dict, eng, shift_lm, reduce_lm, tag_lm);
        ArcEagerParser gold_parse = staticEagerGoldParseSentence(test_sents[j], test_tags[j], gold_arcs, with_words, shift_lm, reduce_lm, tag_lm);
        acc_counts.countAccuracy(parser, gold_parse);
        
        //write output to conll-format file
        //for (unsigned i = 1; i < test_sents[j].size(); ++i) 
        //  outs << i << "\t" << dict.lookup(test_sents[j][i]) << "\t_\t_\t" << dict.lookupTag(test_tags[j][i]) << "\t_\t" << parser.arcs().at(i) << "\tROOT\t_\t_\n";

        //outs << "\n";
      } else {
        ArcStandardParser parser = beamParseSentence(test_sents[j], test_tags[j], gold_arcs, beam_size, with_words, dict, eng, shift_lm, reduce_lm, tag_lm);
        //if (!gold_arcs.is_projective_dependency())
        //  std::cerr << "Non-projective\n";
        //ArcStandardParser parser = particleParseSentence(test_sents[j], test_tags[j], gold_arcs, beam_size, resample, true, with_words, dict, eng, shift_lm, reduce_lm, tag_lm);
        ArcStandardParser gold_parse = staticGoldParseSentence(test_sents[j], test_tags[j], gold_arcs, with_words, shift_lm, reduce_lm, tag_lm);
        acc_counts.countAccuracy(parser, gold_parse);

        //write output to conll-format file
        //for (unsigned i = 1; i < test_sents[j].size(); ++i) 
        //  outs << i << "\t" << dict.lookup(test_sents[j][i]) << "\t_\t_\t" << dict.lookupTag(test_tags[j][i]) << "\t_\t" << parser.arcs().at(i) << "\tROOT\t_\t_\n";

        //outs << "\n";
      }
    }

    //outs.close();
    auto pr_dur = std::chrono::steady_clock::now() - pr_start;
    int milli_sec_dur = std::chrono::duration_cast<std::chrono::milliseconds>(pr_dur).count();
    double sents_per_sec = (static_cast<int>(test_sents.size()) * 1000.0 / milli_sec_dur);
    std::cerr << "\nParsing done...time " << static_cast<int>(milli_sec_dur / 1000.0)
        << "s (" << static_cast<int>(sents_per_sec) << " sentences per second)\n";
 
    acc = acc_counts.directed_accuracy();
    std::cerr << "Word-aligned beam search\n"; 
    std::cerr << "Directed Accuracy: " << acc_counts.directed_accuracy() << std::endl;
    std::cerr << "Undirected Accuracy: " << acc_counts.undirected_accuracy() << std::endl;
    std::cerr << "Final reduce error rate: " << acc_counts.final_reduce_error_rate() << std::endl;
    std::cerr << "Completely correct: " << acc_counts.complete_accuracy() << std::endl;
    std::cerr << "Root correct: " << acc_counts.root_accuracy() << std::endl;
    std::cerr << "ArcDirection Precision: " << acc_counts.arc_dir_precision() << std::endl;
    std::cerr << "Shift recall: " << acc_counts.shift_recall() << std::endl;
    std::cerr << "Reduce recall: " << acc_counts.reduce_recall() << std::endl;   
    std::cerr << "Total length: " << acc_counts.total_length() << std::endl;   
    std::cerr << "Gold Log likelihood: " << acc_counts.gold_likelihood() << std::endl;   
    std::cerr << "Gold Cross entropy: " << acc_counts.gold_cross_entropy() << std::endl;   
    std::cerr << "Gold Perplexity: " << acc_counts.gold_perplexity() << std::endl;   
    std::cerr << "Gold more likely: " << acc_counts.gold_more_likely() << std::endl;   
    std::cerr << "Log likelihood: " << acc_counts.likelihood() << std::endl;   
    std::cerr << "Cross entropy: " << acc_counts.cross_entropy() << std::endl;   
    std::cerr << "Perplexity: " << acc_counts.perplexity() << std::endl;   
    std::cerr << "Beam Log likelihood: " << acc_counts.beam_likelihood() << std::endl;   
    std::cerr << "Beam Cross entropy: " << acc_counts.beam_cross_entropy() << std::endl;   
    std::cerr << "Beam Perplexity: " << acc_counts.beam_perplexity() << std::endl;   
    std::cerr << "Importance Log likelihood: " << acc_counts.importance_likelihood() << std::endl;   
    std::cerr << "Importance Cross entropy: " << acc_counts.importance_cross_entropy() << std::endl;   
    std::cerr << "Importance Perplexity: " << acc_counts.importance_perplexity() << std::endl;   
  }  

  return acc;
}

template<unsigned kShiftOrder, unsigned kTagOrder>
double evaluateEisner(std::string test_file, bool with_words, Dict& dict, PYPLM<kShiftOrder>& shift_lm, PYPLM<kTagOrder>& tag_lm) {
  std::vector<Words> test_sents;
  std::vector<Words> test_tags;
  std::vector<WxList> test_deps;

  std::cerr << "Reading test corpus...\n";
  dict.read_from_conll_file(test_file, &test_sents, &test_tags, &test_deps, true);
  std::cerr << "Corpus size: " << test_sents.size() << " sentences\n";
      
  double acc = 0.0;

  AccuracyCounts acc_counts;
  std::cerr << "\nParsing test sentences...\n";
  //std::cout << "\nParsing test sentences...\n";
  auto pr_start = std::chrono::steady_clock::now();
  //std::ofstream outs;
  //outs.open(out_file);
            
  for (unsigned j = 0; j < test_sents.size(); ++j) {
    ArcList gold_arcs(test_deps[j].size());
    gold_arcs.set_arcs(test_deps[j]);

    EisnerParser gold_parse(test_sents[j], test_tags[j], test_deps[j]);
    std::cout << "gold ";
    eisnerScoreSentence(&gold_parse, with_words, dict, shift_lm, tag_lm);
    std::cout <<  gold_parse.weight() << " ";
    gold_parse.print_arcs();
    EisnerParser parser = eisnerParseSentence(test_sents[j], test_tags[j], gold_arcs, with_words, dict, shift_lm, tag_lm);
    acc_counts.countAccuracy(parser, gold_parse);
    parser.print_sentence(dict);
    if (gold_parse.weight() < parser.weight())
      std::cout << "GOLD MORE LIKELY" << std::endl;

    //write output to conll-format file
    //for (unsigned i = 1; i < test_sents[j].size(); ++i) 
    //  outs << i << "\t" << dict.lookup(test_sents[j][i]) << "\t_\t_\t" << dict.lookupTag(test_tags[j][i]) << "\t_\t" << parser.arcs().at(i) << "\tROOT\t_\t_\n";
    //outs << "\n";
  }

  //outs.close();
  auto pr_dur = std::chrono::steady_clock::now() - pr_start;
  int milli_sec_dur = std::chrono::duration_cast<std::chrono::milliseconds>(pr_dur).count();
  double sents_per_sec = (static_cast<int>(test_sents.size()) * 1000.0 / milli_sec_dur);
  std::cerr << "\nParsing done...time " << static_cast<int>(milli_sec_dur / 1000.0)
      << "s (" << static_cast<int>(sents_per_sec) << " sentences per second)\n";
 
  acc = acc_counts.directed_accuracy();
  std::cerr << "Eisner algorithm decoding\n"; 
  std::cerr << "Directed Accuracy: " << acc_counts.directed_accuracy() << std::endl;
  std::cerr << "Undirected Accuracy: " << acc_counts.undirected_accuracy() << std::endl;
  std::cerr << "Completely correct: " << acc_counts.complete_accuracy() << std::endl;
  std::cerr << "Root correct: " << acc_counts.root_accuracy() << std::endl;
  std::cerr << "ArcDirection Precision: " << acc_counts.arc_dir_precision() << std::endl;
  std::cerr << "Total length: " << acc_counts.total_length() << std::endl;   
  std::cerr << "Gold Log likelihood: " << acc_counts.gold_likelihood() << std::endl;   
  std::cerr << "Gold Cross entropy: " << acc_counts.gold_cross_entropy() << std::endl;   
  std::cerr << "Gold Perplexity: " << acc_counts.gold_perplexity() << std::endl;   
  std::cerr << "Gold more likely: " << acc_counts.gold_more_likely() << std::endl;   
  std::cerr << "Log likelihood: " << acc_counts.likelihood() << std::endl;   
  std::cerr << "Cross entropy: " << acc_counts.cross_entropy() << std::endl;   
  std::cerr << "Perplexity: " << acc_counts.perplexity() << std::endl;   

  return acc;
}




}

#endif
