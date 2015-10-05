#ifndef _GDP_ACC_COUNTS_H_
#define _GDP_ACC_COUNTS_H_

#include "arc_standard_labelled_parser.h"

namespace oxlm {

class AccuracyCounts {

public:
  AccuracyCounts(boost::shared_ptr<Dict> dict);
   
  void inc_reduce_count() {
    ++reduce_count_;
  }

  void inc_reduce_gold() {
    ++reduce_gold_;
  }

  void inc_shift_count() {
    ++shift_count_;
  }

  void inc_shift_gold() {
    ++shift_gold_;
  }

  void inc_final_reduce_error_count() {
    ++final_reduce_error_count_;
  }

  void inc_complete_sentences() {
    ++complete_sentences_;
  }

  void inc_complete_sentences_lab() {
    ++complete_sentences_lab_;
  }

  void inc_complete_sentences_nopunc() {
    ++complete_sentences_nopunc_;
  }

  void inc_complete_sentences_lab_nopunc() {
    ++complete_sentences_lab_nopunc_;
  }

  void inc_gold_more_likely_count() {
    ++gold_more_likely_count_;
  }

  void inc_root_count() {
    ++root_count_;
  }

  void inc_directed_count() {
    ++directed_count_;
  }
  
  void inc_tag_count() {
    ++tag_count_;
  }

  void inc_directed_count_lab() {
    ++directed_count_lab_;
  }

  void inc_directed_count_nopunc() {
    ++directed_count_nopunc_;
  }

  void inc_directed_count_lab_nopunc() {
    ++directed_count_lab_nopunc_;
  }

  void inc_undirected_count() {
    ++undirected_count_;
  }

  void inc_undirected_count_nopunc() {
    ++undirected_count_nopunc_;
  }

  void inc_unheaded_count() {
    ++unheaded_count_;
  }

  void inc_unheaded_count_nopunc() {
    ++unheaded_count_nopunc_;
  }

  void inc_num_sentences() {
    ++num_sentences_;
  }

  void inc_total_length_nopunc() {
    ++total_length_nopunc_; 
  }

  void inc_total_length_nopunc_headed() {
    ++total_length_nopunc_headed_; 
  }

  void inc_total_length_punc_headed() {
    ++total_length_punc_headed_; 
  }

  void add_likelihood(Real l) {
    likelihood_ += l;
  }

  void add_importance_likelihood(Real l) {
    importance_likelihood_ += l;
  }

  void add_beam_likelihood(Real l) {
    beam_likelihood_ += l;
  }

  void add_gold_likelihood(Real l) {
    gold_likelihood_ += l;
  }

  void add_total_length(int l) {
    total_length_ += l; 
  }

  void add_total_length_punc(int l) {
    total_length_punc_ += l; 
  }

  void add_num_actions(int l) {
    num_actions_ += l; 
  }

  void add_directed_count(int l) {
    directed_count_ += l; 
  }

  void add_directed_count_nopunc(int l) {
    directed_count_nopunc_ += l; 
  }

  void add_undirected_count(int l) {
    undirected_count_ += l; 
  }
 
  void add_undirected_count_nopunc(int l) {
    undirected_count_nopunc_ += l; 
  }
  
  void parseCountAccuracy(const Parser& prop_parse, const ParsedSentence& gold_parse); 

  void transitionCountAccuracy(const TransitionParser& prop_parse, const ParsedSentence& gold_parse); 

  void countAccuracy(const ArcStandardLabelledParser& prop_parse, const ParsedSentence& gold_parse); 

  void countGoldLikelihood(Real parse_l, Real gold_l);

  void printAccuracy() const;

  Real directed_accuracy() const {
    return (directed_count_ + 0.0)/total_length_punc_;
  }

  Real tag_accuracy() const {
    return (tag_count_ + 0.0)/total_length_punc_;
  }

  Real directed_precision() const {
    return (directed_count_ + 0.0)/total_length_punc_headed_;
  }

  Real directed_accuracy_lab() const {
    return (directed_count_lab_ + 0.0)/total_length_punc_;
  }

  Real directed_precision_lab() const {
    return (directed_count_lab_ + 0.0)/total_length_punc_headed_;
  }

  Real directed_accuracy_nopunc() const {
    return (directed_count_nopunc_ + 0.0)/total_length_nopunc_;
  }

  Real directed_precision_nopunc() const {
    return (directed_count_nopunc_ + 0.0)/total_length_nopunc_headed_;
  }

  Real directed_accuracy_lab_nopunc() const {
    return (directed_count_lab_nopunc_ + 0.0)/total_length_nopunc_;
  }

  Real directed_precision_lab_nopunc() const {
    return (directed_count_lab_nopunc_ + 0.0)/total_length_nopunc_headed_;
  }

  Real unheaded_recall() const {
    return (unheaded_count_ + 0.0)/total_length_punc_;
  }

  Real unheaded_recall_nopunc() const {
    return (unheaded_count_nopunc_ + 0.0)/total_length_nopunc_;
  }
  Real undirected_accuracy() const {
    return (undirected_count_ + 0.0)/total_length_punc_;
  }

  Real undirected_accuracy_nopunc() const {
    return (undirected_count_nopunc_ + 0.0)/total_length_nopunc_;
  }

  Real complete_accuracy() const {
    return (complete_sentences_ + 0.0)/num_sentences_;
  }

  Real complete_accuracy_lab() const {
    return (complete_sentences_lab_ + 0.0)/num_sentences_;
  }

  Real complete_accuracy_nopunc() const {
    return (complete_sentences_nopunc_ + 0.0)/num_sentences_;
  }

  Real complete_accuracy_lab_nopunc() const {
    return (complete_sentences_lab_nopunc_ + 0.0)/num_sentences_;
  }

  Real root_accuracy() const {
    return (root_count_ + 0.0)/num_sentences_;
  }

  Real gold_more_likely() const {
    return (gold_more_likely_count_ + 0.0)/num_sentences_;
  }

  Real arc_dir_precision() const {
    return (directed_count_ + 0.0)/undirected_count_;
  }

  Real arc_dir_precision_nopunc() const {
    return (directed_count_nopunc_ + 0.0)/undirected_count_nopunc_;
  }

  Real reduce_recall() const {
    return (reduce_count_ + 0.0)/reduce_gold_;
  }

  Real shift_recall() const {
    return (shift_count_ + 0.0)/shift_gold_;
  }

  Real likelihood() const {
    return likelihood_;
  }

  Real importance_likelihood() const {
    return importance_likelihood_;
  }

  Real beam_likelihood() const {
    return beam_likelihood_;
  }

  Real gold_likelihood() const {
    return gold_likelihood_;
  }

  int total_length() const {
    return total_length_;
  }

  int total_length_punc() const {
    return total_length_punc_;
  }

  int total_length_punc_headed() const {
    return total_length_punc_headed_;
  }

  int total_length_nopunc() const {
    return total_length_nopunc_;
  }

  int total_length_nopunc_headed() const {
    return total_length_nopunc_headed_;
  }

  Real final_reduce_error_rate() const {
    return (final_reduce_error_count_ + 0.0)/total_length_punc_;
  }

  Real cross_entropy() const {
    return likelihood_/total_length_;
  }

  Real beam_cross_entropy() const {
    return beam_likelihood_/total_length_;
  }
  
  Real cross_entropy_noeos() const {
    return likelihood_/total_length_punc_;
  }

  Real beam_cross_entropy_noeos() const {
    return beam_likelihood_/total_length_punc_;
  }

  Real importance_cross_entropy() const {
    return importance_likelihood_/total_length_;
  }
  
  Real gold_cross_entropy() const {
    return gold_likelihood_/total_length_;
  }

  Real perplexity_noeos() const {
    return std::exp(cross_entropy_noeos());
  }

  Real beam_perplexity_noeos() const {
    return std::exp(beam_cross_entropy_noeos());
  }

  Real perplexity() const {
    return std::exp(cross_entropy());
  }

  Real beam_perplexity() const {
    return std::exp(beam_cross_entropy());
  }

  Real importance_perplexity() const {
    return std::exp(importance_cross_entropy());
  }

  Real gold_perplexity() const {
    return std::exp(gold_cross_entropy());
  }

  static double sentence_uas(const Parser& prop_parse, const ParsedSentence& gold_parse) {
    double count = 0;
    for (WordIndex j = 1; j < gold_parse.size(); ++j) {
      if (prop_parse.arc_at(j)==gold_parse.arc_at(j)) {
        ++count;
      }
    }

    return count/gold_parse.size();
  }

private:
    boost::shared_ptr<Dict> dict_;
    Real likelihood_;  
    Real beam_likelihood_;  
    Real importance_likelihood_;  
    Real gold_likelihood_;  
    int reduce_count_; 
    int reduce_gold_;
    int shift_count_;
    int shift_gold_;
    int final_reduce_error_count_;
    int total_length_;
    int total_length_punc_;
    int total_length_punc_headed_;
    int total_length_nopunc_;
    int total_length_nopunc_headed_;
    int tag_count_;
    int directed_count_;
    int directed_count_lab_;
    int directed_count_nopunc_;
    int directed_count_lab_nopunc_;
    int unheaded_count_;
    int unheaded_count_nopunc_;
    int undirected_count_;
    int undirected_count_nopunc_;
    int root_count_;
    int gold_more_likely_count_;
    int num_actions_;
    int complete_sentences_;
    int complete_sentences_lab_;
    int complete_sentences_nopunc_;
    int complete_sentences_lab_nopunc_;
    int num_sentences_;
};

}

#endif
