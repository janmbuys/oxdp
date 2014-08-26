#include "accuracy_counts.h"
#include "arc_standard_parser.h"
#include "arc_eager_parser.h"
#include "eisner_parser.h"

//TODO update

void AccuracyCounts::countAccuracy(const EisnerParser& prop_parse, const EisnerParser& gold_parse) {
  ArcList gold_arcs = gold_parse.arcs();      

  inc_num_sentences();
  if (gold_arcs==prop_parse.arcs())
    inc_complete_sentences();
  for (WordIndex i = 1; i < gold_arcs.size(); ++i)
    if (prop_parse.arcs().has_arc(i, 0) && gold_arcs.has_arc(i, 0)) 
      inc_root_count();

  add_likelihood(prop_parse.weight());
  add_gold_likelihood(gold_parse.weight());
  if (gold_parse.weight() < prop_parse.weight())
    inc_gold_more_likely_count();

  add_total_length(gold_arcs.size() - 1);
  add_directed_count(prop_parse.directed_accuracy_count(gold_arcs));
  add_undirected_count(prop_parse.undirected_accuracy_count(gold_arcs));
}

//TODO update
  unsigned directed_accuracy_count(ArcList g_arcs) const {
    unsigned count = 0;
    for (WordIndex j = 1; j < arcs_.size(); ++j) {
      if (arcs_.at(j)==g_arcs.at(j))
        ++count;
    }
    return count;
  }

  unsigned undirected_accuracy_count(ArcList g_arcs) const {
    unsigned count = 0;
    for (WordIndex j = 1; j < arcs_.size(); ++j) {
      if ((arcs_.at(j)==g_arcs.at(j)) || (arcs_.has_parent(j) && g_arcs.at(arcs_.at(j))==static_cast<int>(j)))
        ++count;
    }
    return count;
  }



void AccuracyCounts::countAccuracy(const ArcStandardParser& prop_parse, const ArcStandardParser& gold_parse) {
  //resimulate the computation of the proposed action sequence to compute accuracy  
  ArcStandardParser simul(prop_parse.sentence(), prop_parse.tags());
  ArcList gold_arcs = gold_parse.arcs();      

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
    } /*  else if (next==kAction::la2 || next==kAction::ra2) {
          //for reduce: if there exists an arc it should be added...
          //Unless we know that it is added later in the given parse
        //action taken is considered the gold action
        if (a==kAction::sh) {
          inc_shift_count();
          inc_shift_gold();
        } else {
          inc_reduce_count();
          inc_reduce_gold();
        }
      } */

    if (simul.is_buffer_empty() && next==kAction::re)
      inc_final_reduce_error_count();
    
    simul.execute_action(a);
  }

  inc_num_sentences();
  if (gold_arcs==prop_parse.arcs())
    inc_complete_sentences();
  for (WordIndex i = 1; i < gold_arcs.size(); ++i)
    if (prop_parse.arcs().has_arc(i, 0) && gold_arcs.has_arc(i, 0)) 
      inc_root_count();

  add_likelihood(prop_parse.particle_weight());
  add_importance_likelihood(prop_parse.importance_weight());
  add_beam_likelihood(prop_parse.beam_particle_weight());
  add_gold_likelihood(gold_parse.particle_weight());
  add_num_actions(prop_parse.num_actions());
  if (gold_parse.particle_weight() < prop_parse.particle_weight())
    inc_gold_more_likely_count();

  add_total_length(gold_arcs.size() - 1);
  add_directed_count(prop_parse.directed_accuracy_count(gold_arcs));
  add_undirected_count(prop_parse.undirected_accuracy_count(gold_arcs));
}

void AccuracyCounts::countAccuracy(const ArcEagerParser& prop_parse, const ArcEagerParser& gold_parse) {
  //resimulate the computation of the proposed action sequence to compute accuracy
  ArcEagerParser simul(prop_parse.sentence(), prop_parse.tags());
  ArcList gold_arcs = gold_parse.arcs();      
    
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
            
  inc_num_sentences();
  if (gold_arcs==prop_parse.arcs())
    inc_complete_sentences();
  for (WordIndex i = 1; i < gold_arcs.size(); ++i)
    if (prop_parse.arcs().has_arc(i, 0) && gold_arcs.has_arc(i, 0)) 
      inc_root_count();

  add_likelihood(prop_parse.particle_weight());
  add_importance_likelihood(prop_parse.importance_weight());
  add_beam_likelihood(prop_parse.beam_particle_weight());
  add_gold_likelihood(gold_parse.particle_weight());
  add_num_actions(prop_parse.num_actions());
  if (gold_parse.particle_weight() < prop_parse.particle_weight())
    inc_gold_more_likely_count();

  add_total_length(gold_arcs.size() - 1);
  add_directed_count(prop_parse.directed_accuracy_count(gold_arcs));
  add_undirected_count(prop_parse.undirected_accuracy_count(gold_arcs));
}


