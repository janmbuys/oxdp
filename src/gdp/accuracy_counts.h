#ifndef _GDP_ACC_COUNTS_H_
#define _GDP_ACC_COUNTS_H_

namespace oxlm {

class AccuracyCounts {

public:
  AccuracyCounts(): 
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

  void inc_gold_more_likely_count() {
    ++gold_more_likely_count_;
  }

  void inc_root_count() {
    ++root_count_;
  }

  void inc_num_sentences() {
    ++num_sentences_;
  }

  void add_likelihood(double l) {
    likelihood_ += l;
  }

  void add_importance_likelihood(double l) {
    importance_likelihood_ += l;
  }

  void add_beam_likelihood(double l) {
    beam_likelihood_ += l;
  }

  void add_gold_likelihood(double l) {
    gold_likelihood_ += l;
  }

  void add_total_length(int l) {
    total_length_ += l; 
  }

  void add_num_actions(int l) {
    num_actions_ += l; 
  }

  void add_directed_count(int l) {
    directed_count_ += l; 
  }

  void add_undirected_count(int l) {
    undirected_count_ += l; 
  }
   
  void countAccuracy(const ArcStandardParser& prop_parse, const ArcStandardParser& gold_parse); 

  void countAccuracy(const ArcEagerParser& prop_parse, const ArcEagerParser& gold_parse); 

  void countAccuracy(const EisnerParser& prop_parse, const EisnerParser& gold_parse); 

  double directed_accuracy() const {
    return (directed_count_ + 0.0)/total_length_;
  }

  double undirected_accuracy() const {
    return (undirected_count_ + 0.0)/total_length_;
  }

  double complete_accuracy() const {
    return (complete_sentences_ + 0.0)/num_sentences_;
  }

  double root_accuracy() const {
    return (root_count_ + 0.0)/num_sentences_;
  }

  double gold_more_likely() const {
    return (gold_more_likely_count_ + 0.0)/num_sentences_;
  }

  double arc_dir_precision() const {
    return (directed_count_ + 0.0)/undirected_count_;
  }

  double reduce_recall() const {
    return (reduce_count_ + 0.0)/reduce_gold_;
  }

  double shift_recall() const {
    return (shift_count_ + 0.0)/shift_gold_;
  }

  double likelihood() const {
    return likelihood_;
  }

  double importance_likelihood() const {
    return importance_likelihood_;
  }

  double beam_likelihood() const {
    return beam_likelihood_;
  }

  double gold_likelihood() const {
    return gold_likelihood_;
  }

  int total_length() const {
    return total_length_;
  }

  double final_reduce_error_rate() const {
    return (final_reduce_error_count_ + 0.0)/total_length_;
  }

  double cross_entropy() const {
    return likelihood_/(std::log(2)*total_length_);
    //return likelihood_/(std::log(2)*num_actions_);
  }

  double beam_cross_entropy() const {
    return beam_likelihood_/(std::log(2)*total_length_);
    //return beam_likelihood_/(std::log(2)*num_actions_);
  }
  
  double importance_cross_entropy() const {
    return importance_likelihood_/(std::log(2)*total_length_);
  }
  
  double gold_cross_entropy() const {
    return gold_likelihood_/(std::log(2)*total_length_);
    //return gold_likelihood_/(std::log(2)*num_actions_);
  }

  double perplexity() const {
    return std::pow(2, cross_entropy());
  }

  double beam_perplexity() const {
    return std::pow(2, beam_cross_entropy());
  }

  double importance_perplexity() const {
    return std::pow(2, importance_cross_entropy());
  }

  double gold_perplexity() const {
    return std::pow(2, gold_cross_entropy());
  }

private:
    double likelihood_;  
    double beam_likelihood_;  
    double importance_likelihood_;  
    double gold_likelihood_;  
    int reduce_count_; 
    int reduce_gold_;
    int shift_count_;
    int shift_gold_;
    int final_reduce_error_count_;
    int total_length_;
    int directed_count_;
    int undirected_count_;
    int root_count_;
    int gold_more_likely_count_;
    int num_actions_;
    int complete_sentences_;
    int num_sentences_;
};

}

#endif
