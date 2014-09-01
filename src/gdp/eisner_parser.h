#ifndef _GDP_ESNR_PARSER_H_
#define _GDP_ESNR_PARSER_H_

#include<string>
#include<array>
#include<cstdlib>

#include "corpus/dict.h"
#include "parser.h"

namespace oxlm {

typedef std::array<double, 4> EChartItem;
typedef std::array<WordIndex, 4> ESplitChartItem;
typedef std::vector<std::vector<EChartItem>> EChart;
typedef std::vector<std::vector<ESplitChartItem>> ESplitChart;

class EisnerParser: public Parser {
  public:

  EisnerParser();
   
  EisnerParser(Words tags);

  EisnerParser(Words sent, Words tags);

  EisnerParser(Words sent, Words tags, Indices arcs);

  EisnerParser(const ParsedSentence& parse);

  void recoverParseTree(WordIndex s, WordIndex t, bool complete, bool left_arc);

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

  void print_chart() const {
    for (unsigned i = 0; i < size(); ++i) {
      for (unsigned j = 0; j < size(); ++j) {
        std::cout << "(" << chart_[i][j][0] << ", " << chart_[i][j][1] << ", " <<
chart_[i][j][2] << ", " <<  chart_[i][j][3] << ") "; 
      }
      std::cout << std::endl;
    }
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
 
  Words wordContext(WordIndex i, WordIndex j, WordIndex k) const;

  Words wordContext(WordIndex i, WordIndex j) const;

  Words tagContext(WordIndex i, WordIndex j, WordIndex k) const;

  Words tagContext(WordIndex i, WordIndex j) const;
  
  private: 
  EChart chart_;
  ESplitChart split_chart_;
};

}
#endif
