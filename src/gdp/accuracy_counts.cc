#include "accuracy_counts.h"

namespace oxlm {

AccuracyCounts::AccuracyCounts(boost::shared_ptr<Dict> dict)
    : dict_(dict),
      likelihood_{0},
      beam_likelihood_{0},
      importance_likelihood_{0},
      gold_likelihood_{0},
      marginal_likelihood_{0},
      reduce_count_{0},
      reduce_gold_{0},
      shift_count_{0},
      shift_gold_{0},
      final_reduce_error_count_{0},
      total_length_{0},
      total_length_punc_{0},
      total_length_punc_headed_{0},
      total_length_nopunc_{0},
      total_length_nopunc_headed_{0},
      tag_count_{0},
      directed_count_{0},
      directed_count_lab_{0},
      directed_count_nopunc_{0},
      directed_count_lab_nopunc_{0},
      undirected_count_{0},
      undirected_count_nopunc_{0},
      root_count_{0},
      gold_more_likely_count_{0},
      num_actions_{0},
      complete_sentences_{0},
      complete_sentences_lab_{0},
      complete_sentences_nopunc_{0},
      complete_sentences_lab_nopunc_{0},
      num_sentences_{0} {}

void AccuracyCounts::parseCountAccuracy(const Parser& prop_parse,
                                        const ParsedSentence& gold_parse) {
  // Sentence level counts.
  inc_num_sentences();
  if (prop_parse.equal_arcs(gold_parse)) {
    inc_complete_sentences();
    if (prop_parse.equal_labels(gold_parse)) inc_complete_sentences_lab();
  }

  add_total_length(gold_parse.size());
  add_total_length_punc(gold_parse.size() - 1);
  bool punc_complete = true;
  bool lab_punc_complete = true;

  add_likelihood(prop_parse.weight());

  // Arc level counts.
  for (WordIndex j = 1; j < gold_parse.size(); ++j) {
    if (prop_parse.arc_at(j) == gold_parse.arc_at(j)) {
      inc_directed_count();
      inc_undirected_count();
      if (prop_parse.label_at(j) == gold_parse.label_at(j)) {
        inc_directed_count_lab();
      }

      if (!dict_->punctTag(prop_parse.tag_at(j))) {
        inc_directed_count_nopunc();
        inc_undirected_count_nopunc();
        if (prop_parse.label_at(j) == gold_parse.label_at(j)) {
          inc_directed_count_lab_nopunc();
        }
      }

    } else if (prop_parse.has_parent_at(j) &&
               (gold_parse.arc_at(prop_parse.arc_at(j)) == j)) {
      inc_undirected_count();
      if (!dict_->punctTag(prop_parse.tag_at(j))) {
        inc_undirected_count_nopunc();
      }
    }

    if (prop_parse.tag_at(j) == gold_parse.tag_at(j)) {
      inc_tag_count();
    }
    if (prop_parse.has_arc(j, 0) && gold_parse.has_arc(j, 0)) {
      inc_root_count();
    }
    if (prop_parse.has_parent_at(j))
      inc_total_length_punc_headed();
    else
      inc_unheaded_count();

    if (!dict_->punctTag(prop_parse.tag_at(j))) {
      inc_total_length_nopunc();
      if (prop_parse.has_parent_at(j)) {
        inc_total_length_nopunc_headed();
      } else {
        inc_unheaded_count_nopunc();
      }
      if (prop_parse.arc_at(j) != gold_parse.arc_at(j)) {
        punc_complete = false;
        lab_punc_complete = false;
      } else if (prop_parse.label_at(j) != gold_parse.label_at(j)) {
        lab_punc_complete = false;
      }
    }
  }

  if (punc_complete) inc_complete_sentences_nopunc();
  if (lab_punc_complete) inc_complete_sentences_lab_nopunc();
}

void AccuracyCounts::transitionCountAccuracy(const TransitionParser& prop_parse,
                                             const ParsedSentence& gold_parse) {
  parseCountAccuracy(prop_parse, gold_parse);

  // Counts for transition parsers.
  add_importance_likelihood(prop_parse.importance_weight());
  add_marginal_likelihood(prop_parse.marginal_weight());
  add_beam_likelihood(prop_parse.beam_weight());
  add_num_actions(prop_parse.num_actions());
}

void AccuracyCounts::countAccuracy(const ArcStandardLabelledParser& prop_parse,
                                   const ParsedSentence& gold_parse) {
  transitionCountAccuracy(prop_parse, gold_parse);

  // Resimulates the computation of the proposed action sequence to compute
  // accuracy.
  ArcStandardLabelledParser simul(static_cast<TaggedSentence>(prop_parse),
                                  prop_parse.config());

  for (unsigned i = 0; i < prop_parse.actions().size(); ++i) {
    kAction a = prop_parse.actions().at(i);
    WordId alab = prop_parse.action_label_at(i);

    kAction next = simul.oracleNext(gold_parse);
    WordId nextLabel = simul.oracleNextLabel(gold_parse);

    // Counts shift/reduce recall.
    if (next == kAction::sh) {
      inc_shift_gold();
      if (a == kAction::sh) inc_shift_count();
    } else if (next == kAction::la || next == kAction::ra) {
      inc_reduce_gold();
      if (a == kAction::la || a == kAction::ra) inc_reduce_count();
    }

    if (simul.buffer_empty() && next == kAction::re) {
      inc_final_reduce_error_count();
    }

    simul.executeAction(a, alab);
  }
}

void AccuracyCounts::countGoldLikelihood(Real parse_l, Real gold_l) {
  add_gold_likelihood(gold_l);
  if (gold_l < parse_l) inc_gold_more_likely_count();
}

void AccuracyCounts::printAccuracy() const {
  std::cerr << "Labelled Accuracy No Punct: " << directed_accuracy_lab_nopunc()
            << std::endl;
  std::cerr << "Unlabelled Accuracy No Punct: " << directed_accuracy_nopunc()
            << std::endl;
  std::cerr << "Root correct: " << root_accuracy() << std::endl;
  std::cerr << "Labelled Precision No Punct: "
            << directed_precision_lab_nopunc() << std::endl;
  std::cerr << "Unlabelled Precision No Punct: " << directed_precision_nopunc()
            << std::endl;
  std::cerr << "Unheaded recall No Punct: " << unheaded_recall_nopunc()
            << std::endl;
  std::cerr << "Labelled Completely correct No Punct: "
            << complete_accuracy_lab_nopunc() << std::endl;
  std::cerr << "Completely correct No Punct: " << complete_accuracy_nopunc()
            << std::endl;

  std::cerr << std::endl;
  std::cerr << "Perplexity: " << perplexity() << std::endl;
  std::cerr << "Perplexity No EOS: " << perplexity_noeos() << std::endl;
  std::cerr << "Cross entropy: " << cross_entropy() << std::endl;
  std::cerr << "Log likelihood: " << likelihood() << std::endl;
  std::cerr << "Beam Perplexity: " << beam_perplexity() << std::endl;
  std::cerr << "Beam Perplexity No EOS: " << beam_perplexity_noeos()
            << std::endl;
  std::cerr << "Beam Cross entropy: " << beam_cross_entropy() << std::endl;
  std::cerr << "Beam Log likelihood: " << beam_likelihood() << std::endl;
  std::cerr << "Importance Perplexity: " << importance_perplexity() << std::endl;
  std::cerr << "Importance Perplexity No EOS: " << importance_perplexity_noeos()
            << std::endl;
  std::cerr << "Importance Cross entropy: " << importance_cross_entropy()
            << std::endl;
  std::cerr << "Importance Log likelihood: " << importance_likelihood()
            << std::endl;
  std::cerr << "Marginal Perplexity: " << marginal_perplexity() << std::endl;
  std::cerr << "Marginal Perplexity No EOS: " << marginal_perplexity_noeos()
            << std::endl;
  std::cerr << "Marginal Cross entropy: " << marginal_cross_entropy()
            << std::endl;
  std::cerr << "Marginal Log likelihood: " << marginal_likelihood()
            << std::endl;
 
  std::cerr << "Gold Perplexity: " << gold_perplexity() << std::endl;
  std::cerr << "Gold Perplexity No EOS: " << gold_perplexity_noeos() << std::endl;
  std::cerr << "Gold Cross entropy: " << gold_cross_entropy() << std::endl;
  std::cerr << "Gold Log likelihood: " << gold_likelihood() << std::endl;
  std::cerr << "Gold more likely: " << gold_more_likely() << std::endl;

  std::cerr << std::endl;
  std::cerr << "Tag Accuracy With Punct: " << tag_accuracy() << std::endl;
  std::cerr << "Labelled Accuracy With Punct: " << directed_accuracy_lab()
            << std::endl;
  std::cerr << "Unlabelled Accuracy With Punct: " << directed_accuracy()
            << std::endl;
  std::cerr << "Labelled Precision With Punct: " << directed_precision_lab()
            << std::endl;
  std::cerr << "Unlabelled Precision With Punct: " << directed_precision()
            << std::endl;
  std::cerr << "Unheaded recall With Punct: " << unheaded_recall() << std::endl;
  std::cerr << "Labelled Completely correct With Punct: "
            << complete_accuracy_lab() << std::endl;
  std::cerr << "Completely correct With Punct: " << complete_accuracy()
            << std::endl;
  std::cerr << "Final reduce error rate: " << final_reduce_error_rate()
            << std::endl;
  std::cerr << "ArcDirection Precision With Punct: " << arc_dir_precision()
            << std::endl;
  std::cerr << "ArcDirection Precision No Punct: " << arc_dir_precision_nopunc()
            << std::endl;
  std::cerr << "Shift recall: " << shift_recall() << std::endl;
  std::cerr << "Reduce recall: " << reduce_recall() << std::endl;
  std::cerr << "Total length: " << total_length() << std::endl;
  std::cerr << "Total length no punct: " << total_length_nopunc() << std::endl;
  std::cerr << std::endl;
}

}  // namespace oxlm
