#include "accuracy_counts.h"

#include "arc_standard_parser.h"
#include "arc_eager_parser.h"
#include "eisner_parser.h"

//TODO update

//mother method
void AccuracyCounts::countAccuracy(const Parse& prop_parse, const Parse& gold_parse) {
  //sentence level
  inc_num_sentences();
  if (ParsedSentence::equal_arcs(prop_parse, gold_parse))
    inc_complete_sentences();
  add_total_length(gold_parse.size() - 1);
  
  add_likelihood(prop_parse.weight());
  add_gold_likelihood(gold_parse.weight());
  if (gold_parse.weight() < prop_parse.weight())
    inc_gold_more_likely_count();

  //arc level
  for (WordIndex j = 1; j < gold_parse.size(); ++j) {
    if (prop_parse.arc_at(j)==gold_parse_arc_at(j)) {
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

void AccuracyCounts::countAccuracy(const EisnerParser& prop_parse, 
                                   const EisnerParser& gold_parse) {
  //just call parent method
  countAccuracy(prop_parse, gold_parse); 
}

//TODO find the right signature for this
void AccuracyCounts::countAccuracy(const ArcStandardParser& prop_parse, 
                                   const ArcStandardParser& gold_parse) {
  //parent method
  countAccuracy(prop_parse, gold_parse); 
  
  //general for transition parser 
  add_importance_likelihood(prop_parse.importance_weight());
  add_beam_likelihood(prop_parse.beam_particle_weight());
  add_num_actions(prop_parse.num_actions());

  //resimulate the computation of the proposed action sequence to compute accuracy  
  //TODO need an appropriate constructor
  ArcStandardParser simul(prop_parse.sentence(), prop_parse.tags());

  for (auto& a: prop_parse.actions()) {
    kAction next = simul.oracleDynamicNext(gold_arcs);
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
    
    simul.execute_action(a);
  }
}

void AccuracyCounts::countAccuracy(const ArcEagerParser& prop_parse, const ArcEagerParser& gold_parse) {
  //parent method
  countAccuracy(prop_parse, gold_parse); 
  
  //general for transition parser 
  add_importance_likelihood(prop_parse.importance_weight());
  add_beam_likelihood(prop_parse.beam_particle_weight());
  add_num_actions(prop_parse.num_actions());

  //resimulate the computation of the proposed action sequence to compute accuracy  
  //TODO need an appropriate constructor
  ArcEagerParser simul(prop_parse.sentence(), prop_parse.tags());
  
  for (auto& a: prop_parse.actions()) {
    kAction next = simul.oracleNext(gold_arcs);
    
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
    
    simul.execute_action(a);
  }
            
}  

