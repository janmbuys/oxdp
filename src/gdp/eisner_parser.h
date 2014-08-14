#ifndef _GDP_ESNR_PARSER_H_
#define _GDP_ESNR_PARSER_H_

#include<string>
#include<array>
#include<cstdlib>

#include<corpus/corpus.h>

namespace oxlm {

typedef std::array<double, 4> EChartItem;
typedef std::array<WordIndex, 4> ESplitChartItem;
typedef std::vector<std::vector<EChartItem>> EChart;
typedef std::vector<std::vector<ESplitChartItem>> ESplitChart;

class EisnerParser {
  public:

  //specifically for generation
  EisnerParser():
    chart_(),
    split_chart_(),
    arcs_(),
    sentence_(),
    tags_(),
    lw_{0},
    is_generating_{true}
  {
    //will need different approach for generation (not sequential) - don't worry now
    arcs_.push_back();
  }      
   
  EisnerParser(Words tags):
    chart_(tags.size(), std::vector<EChartItem>(tags.size(), EChartItem())),
    split_chart_(tags.size(), std::vector<ESplitChartItem>(tags.size(), ESplitChartItem())),
    arcs_(tags.size()),
    sentence_(tags.size(), 0),
    tags_(tags),
    lw_{0},
    is_generating_{false}
  {

  }
   
  EisnerParser(Words sent, Words tags):
    chart_(sent.size(), std::vector<EChartItem>(sent.size(), EChartItem())),
    split_chart_(sent.size(), std::vector<ESplitChartItem>(sent.size(), ESplitChartItem())),
    arcs_(sent.size()),
    sentence_(sent),
    tags_(tags),
    lw_{0},
    is_generating_{false}
  {

  }

  void push_arc() {
    arcs_.push_back();
  }

  void push_word(WordId w) {
   sentence_.push_back(w);
  }

  //modifier i, head j
  void set_arc(WordIndex i, WordIndex j) {
    arcs_.set_arc(i, j);
  }

  void reset_weight() {
    lw_ = 0;
  }

  void set_weight(double w) {
    lw_ = -std::log(w);
  }

  void set_log_weight(double w) {
    lw_ = w;
  }

  void add_weight(double w) {
    lw_ -= std::log(w);
  }

  void set_left_incomplete_weight(WordIndex i, WordIndex j, double w) {
    chart_[i][j][0] = w;
  }

  void set_right_incomplete_weight(WordIndex i, WordIndex j, double w) {
    chart_[i][j][1] = w;
  }

  void set_left_complete_weight(WordIndex i, WordIndex j, double w) {
    chart_[i][j][2] = w;
  }

  void set_right_complete_weight(WordIndex i, WordIndex j, double w) {
    chart_[i][j][3] = w;
  }

  void set_left_incomplete_split(WordIndex i, WordIndex j, WordIndex k) {
    split_chart_[i][j][0] = k;
  }

  void set_right_incomplete_split(WordIndex i, WordIndex j, WordIndex k) {
    split_chart_[i][j][1] = k;
  }

  void set_left_complete_split(WordIndex i, WordIndex j, WordIndex k) {
    split_chart_[i][j][2] = k;
  }

  void set_right_complete_split(WordIndex i, WordIndex j, WordIndex k) {
    split_chart_[i][j][3] = k;
  }

  void recover_parse_tree(WordIndex s, WordIndex t, bool complete, bool left_arc) {
    if (!complete) {
      WordIndex r;

      if (left_arc) {
        //left arc
        set_arc(s, t);
        r = left_incomplete_split(s, t);
      } else {
        //right arc
        set_arc(t, s);
        r = right_incomplete_split(s, t);
      }

      recover_parse_tree(s, r, true, false);
      recover_parse_tree(r+1, t, true, true);
    } else {
      if (left_arc) {
        WordIndex r = left_complete_split(s, t);
        recover_parse_tree(s, r, true, true);
        recover_parse_tree(r, t, false, true);
      } else {
        WordIndex r = right_complete_split(s, t);
        recover_parse_tree(s, r, false, false);
        recover_parse_tree(r, t, true, false);
      }
    }
  }

  void print_arcs() const {
    for (auto a: arcs_.arcs())
      std::cout << a << " ";
    //std::cout << std::endl;
  }

  void print_sentence(Dict& dict) const {
    for (auto a: sentence_)
      std::cout << dict.lookup(a) << " ";
    std::cout << std::endl;
  }

  void print_tags(Dict& dict) const {
    for (auto a: tags_)
      std::cout << dict.lookupTag(a) << " ";
    std::cout << std::endl;
  }

  unsigned sentence_length() const {
    return sentence_.size();
  }

  Words sentence() const {
    return sentence_;
  }

  Words tags() const {
    return tags_;
  }
  
  WordId tag_at(WordIndex i) const {
    return tags_.at(i);
  }

  ArcList arcs() const {
    return arcs_;
  }

  double weight() const {
    return lw_;
  }

  bool is_generating() const {
    return is_generating_;
  }

  //number of children at sentence position i
  int child_count_at(int i) const {
    return arcs_.child_count_at(i);
  }

  bool root_has_child() const {
    return (arcs_.child_count_at(0) > 0);
  }

  bool has_parent(int i) const {
    return arcs_.has_parent(i);
  }

  bool is_complete_parse() const {
    for (WordIndex i = 1; i < arcs_.size() - 1; ++i) {
      if (!arcs_.has_parent(i) && (tags_.at(i)!=1))
        return false;
    }

    return true;
  }

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

  //get chart item weights

  double left_incomplete_weight(WordIndex i, WordIndex j) const {
    return chart_[i][j][0];
  }

  double right_incomplete_weight(WordIndex i, WordIndex j) const {
    return chart_[i][j][1];
  }

  double left_complete_weight(WordIndex i, WordIndex j) const {
    return chart_[i][j][2];
  }

  double right_complete_weight(WordIndex i, WordIndex j) const {
    return chart_[i][j][3];
  }

  WordIndex left_incomplete_split(WordIndex i, WordIndex j) const {
    return split_chart_[i][j][0];
  }

  WordIndex right_incomplete_split(WordIndex i, WordIndex j) const {
    return split_chart_[i][j][1];
  }

  WordIndex left_complete_split(WordIndex i, WordIndex j) const {
    return split_chart_[i][j][2];
  }

  WordIndex right_complete_split(WordIndex i, WordIndex j) const {
    return split_chart_[i][j][3];
  }

  //context functions
  //modifier i, head j
  //will add non-arc-factored context later
  //TODO
  
  private: 
  EChart chart_;
  ESplitChart split_chart_;
  ArcList arcs_;
  Words sentence_;
  Words tags_;
  double lw_;
  bool is_generating_;
};

}
#endif
