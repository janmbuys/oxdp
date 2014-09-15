#include "accuracy_counts.h"

namespace oxlm {

AccuracyCounts::AccuracyCounts(): 
  likelihood_{0},
  beam_likelihood_{0},
  importance_likelihood_{0},
  gold_likelihood_{0},
  reduce_count_{0},
  reduce_gold_{0},
  shift_count_{0},
  shift_gold_{0},
  final_reduce_error_count_{0},
  total_length_{0},
  directed_count_{0},
  undirected_count_{0}, 
  root_count_{0},
  gold_more_likely_count_{0},
  num_actions_{0},
  complete_sentences_{0},
  num_sentences_{0}
  {
  }

void AccuracyCounts::parseCountAccuracy(const Parser& prop_parse, const ParsedSentence& gold_parse) {
  //sentence level
  inc_num_sentences();
  if (prop_parse.has_equal_arcs(gold_parse))
    inc_complete_sentences();
  add_total_length(gold_parse.size() - 1);
  
  //maybe change this again later ...
  add_likelihood(prop_parse.weight());
  /* add_gold_likelihood(gold_parse.weight());
  if (gold_parse.weight() < prop_parse.weight())
    inc_gold_more_likely_count(); */

  //arc level
  for (WordIndex j = 1; j < gold_parse.size(); ++j) {
    if (prop_parse.arc_at(j)==gold_parse.arc_at(j)) {
      inc_directed_count();
      inc_undirected_count(); 
    } else if (prop_parse.has_parent_at(j) && 
              (gold_parse.arc_at(prop_parse.arc_at(j))==j)) {
      inc_undirected_count(); 
    }
      
    if (prop_parse.has_arc(j, 0) && gold_parse.has_arc(j, 0)) 
      inc_root_count();
  }
}

void AccuracyCounts::transitionCountAccuracy(const TransitionParser& prop_parse, 
                                   const ParsedSentence& gold_parse) {
  //parent method
  parseCountAccuracy(prop_parse, gold_parse); 
  
  //general for transition parser 
  add_importance_likelihood(prop_parse.importance_weight());
  add_beam_likelihood(prop_parse.beam_weight());
  add_num_actions(prop_parse.num_actions());
}

void AccuracyCounts::countAccuracy(const EisnerParser& prop_parse, 
                                   const ParsedSentence& gold_parse) {
  //just call parent method
  parseCountAccuracy(prop_parse, gold_parse); 
}

//this isn't ideal, but good enough for now
void AccuracyCounts::countAccuracy(const ArcStandardParser& prop_parse, 
                                   const ParsedSentence& gold_parse) {
  //parent method
  transitionCountAccuracy(prop_parse, gold_parse); 
  
  //resimulate the computation of the proposed action sequence to compute accuracy  
  ArcStandardParser simul(prop_parse); //need sentence and tags

  for (auto& a: prop_parse.actions()) {
    kAction next = simul.oracleNext(gold_parse);

    //count when shifted/reduced when it should have shifted/reduced
    if (next==kAction::sh) {
      inc_shift_gold();
      if (a==kAction::sh)
        inc_shift_count();
    } else if (next==kAction::la || next==kAction::ra) {
      inc_reduce_gold();
      if (a==kAction::la || a==kAction::ra) //counts either direction
        inc_reduce_count();
    } 
  
    if (simul.buffer_empty() && next==kAction::re)
      inc_final_reduce_error_count();
    
    simul.executeAction(a);
  }
}

void AccuracyCounts::countAccuracy(const ArcEagerParser& prop_parse, const ParsedSentence& gold_parse) {
  //parent method
  transitionCountAccuracy(prop_parse, gold_parse); 
  
  //resimulate the computation of the proposed action sequence to compute accuracy  
  ArcEagerParser simul(prop_parse);
  
  for (auto& a: prop_parse.actions()) {
    kAction next = simul.oracleNext(gold_parse);
    
    //include more sophisticated statistics later
    //count when shifted/reduced when it should have shifted/reduced
    if (next==kAction::sh || next==kAction::ra) {
      inc_shift_gold();
      if (a==kAction::sh || a==kAction::ra)
        inc_shift_count();
    } else if (next==kAction::la || next==kAction::re) {
      inc_reduce_gold();
      if (a==kAction::la || a==kAction::re) 
        inc_reduce_count();
    } 
    
    simul.executeAction(a);
  }
}  

void AccuracyCounts::countLikelihood(double parse_l, double gold_l) {
  add_gold_likelihood(gold_l);
  if (gold_l < parse_l)
    inc_gold_more_likely_count();
}

void AccuracyCounts::printAccuracy() const {
  std::cerr << "Directed Accuracy: " << directed_accuracy() << std::endl;
  std::cerr << "Undirected Accuracy: " << undirected_accuracy() << std::endl;
  std::cerr << "Final reduce error rate: " << final_reduce_error_rate() << std::endl;
  std::cerr << "Completely correct: " << complete_accuracy() << std::endl;
  std::cerr << "Root correct: " << root_accuracy() << std::endl;
  std::cerr << "ArcDirection Precision: " << arc_dir_precision() << std::endl;
  std::cerr << "Shift recall: " << shift_recall() << std::endl;
  std::cerr << "Reduce recall: " << reduce_recall() << std::endl;   
  std::cerr << "Total length: " << total_length() << std::endl;   
  std::cerr << "Gold Log likelihood: " << gold_likelihood() << std::endl;   
  std::cerr << "Gold Cross entropy: " << gold_cross_entropy() << std::endl;   
  std::cerr << "Gold Perplexity: " << gold_perplexity() << std::endl;   
  std::cerr << "Gold more likely: " << gold_more_likely() << std::endl;   
  std::cerr << "Log likelihood: " << likelihood() << std::endl;   
  std::cerr << "Cross entropy: " << cross_entropy() << std::endl;   
  std::cerr << "Perplexity: " << perplexity() << std::endl;   
  std::cerr << "Beam Log likelihood: " << beam_likelihood() << std::endl;   
  std::cerr << "Beam Cross entropy: " << beam_cross_entropy() << std::endl;   
  std::cerr << "Beam Perplexity: " << beam_perplexity() << std::endl;   
  std::cerr << "Importance Log likelihood: " << importance_likelihood() << std::endl;   
  std::cerr << "Importance Cross entropy: " << importance_cross_entropy() << std::endl;
  std::cerr << "Importance Perplexity: " << importance_perplexity() << std::endl;   
}

}

