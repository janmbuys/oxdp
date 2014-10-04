#ifndef _GDP_ESNR_PARSER_H_
#define _GDP_ESNR_PARSER_H_

#include<string>
#include<array>
#include<cstdlib>

#include "corpus/dict.h"
#include "corpus/parse_data_set.h"
#include "gdp/parser.h"

namespace oxlm {

typedef std::array<Real, 4> EChartItem;
typedef std::array<WordIndex, 4> ESplitChartItem;
typedef std::vector<std::vector<EChartItem>> EChart;
typedef std::vector<std::vector<ESplitChartItem>> ESplitChart;

class EisnerParser: public Parser {
  public:

  EisnerParser();
   
  EisnerParser(Words tags);

  EisnerParser(Words sent, Words tags);

  EisnerParser(Words sent, Words tags, Indices arcs);

  EisnerParser(const TaggedSentence& parse);

  void recoverParseTree(WordIndex s, WordIndex t, bool complete, bool left_arc);

  void extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const;

  void set_left_incomplete_weight(WordIndex i, WordIndex j, Real w) {
    chart_[i][j][0] = w;
  }

  void set_right_incomplete_weight(WordIndex i, WordIndex j, Real w) {
    chart_[i][j][1] = w;
  }

  void set_left_complete_weight(WordIndex i, WordIndex j, Real w) {
    chart_[i][j][2] = w;
  }

  void set_right_complete_weight(WordIndex i, WordIndex j, Real w) {
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

  Real left_incomplete_weight(WordIndex i, WordIndex j) const {
    return chart_[i][j][0];
  }

  Real right_incomplete_weight(WordIndex i, WordIndex j) const {
    return chart_[i][j][1];
  }

  Real left_complete_weight(WordIndex i, WordIndex j) const {
    return chart_[i][j][2];
  }

  Real right_complete_weight(WordIndex i, WordIndex j) const {
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

  WordId eos() const {
    return eos_;
  }

  //context functions
  //modifier i, head j, previously generated sibling k
  
  Words wordContext(WordIndex i, WordIndex j, WordIndex k) const;

  Words tagContext(WordIndex i, WordIndex j, WordIndex k) const;
 
  Words word_tag_context(WordIndex i, WordIndex j, WordIndex k) const {
    Words ctx(5, 0);
    if (i >= 0 && i < static_cast<int>(size()))
      ctx[4] = tag_at(i);
    else
      ctx[4] = eos(); //stop word
    ctx[3] = tag_at(j);
    if (k > 0)
      ctx[2] = tag_at(k);
    ctx[1] = word_at(j);
    if (k > 0)
      ctx[0] = word_at(k);

    return ctx; 
  }
  
  Words word_tag_context(WordIndex i, WordIndex j) const {
    Words ctx(3, 0);
    if (i >= 0)
      ctx[2] = tag_at(i);
    if (j >= 0) {
      ctx[1] = tag_at(j);
      ctx[0] = word_at(j);
    }

    return ctx;
  }

  Words word_only_context(WordIndex i, WordIndex j, WordIndex k) const {
    Words ctx(5, 0);
    if (i >= 0 && i < static_cast<int>(size()))
      ctx[4] = word_at(i);
    else
      ctx[4] = eos(); //stop word
    ctx[3] = word_at(j);
    if (k > 0)
      ctx[2] = word_at(k);
    ctx[1] = word_at(j);
    if (k > 0)
      ctx[0] = word_at(k);

    return ctx; 
  }
  
  Words word_only_context(WordIndex i, WordIndex j) const {
    Words ctx(3, 0);
    if (i >= 0)
      ctx[2] = tag_at(i);
    if (j >= 0) {
      ctx[1] = tag_at(j);
      ctx[0] = word_at(j);
    }

    return ctx;
  }

  Words tag_only_context(WordIndex i, WordIndex j, WordIndex k) const {
   //similar to Eisner generative
    Words ctx(6, 0);
    ctx[4] = tag_at(j);
    if (j > i) //if left arc
      ctx[5] = 1;
    //previous child k
    if (((k > i) && (k < j)) || ((k > j) && (k < i)))
      ctx[3] = tag_at(k);
    ctx[2] = std::min(10, std::abs(i - j));

    if (j > (i+1)) {
      ctx[1] = tag_at(j-1); 
      if (j > (i+2))
        ctx[0] = tag_at(i+1); 
    } else if (i > (j+1)) {
      ctx[1] = tag_at(j+1); 
      if (i > (j+2))
        ctx[0] = tag_at(i-1); 
    } 

    return ctx; 
  }

  Words tag_only_context(WordIndex i, WordIndex j) const {
    //for now, try to replicate Adhi's conditioning context
    Words ctx(7, 0);
    if (j >= 0)
      ctx[6] = tag_at(j);
    if (j > i) //if left arc
      ctx[5] = 1;
    ctx[4] = std::min(10, std::abs(i - j));
    if (j > (i+1)) {
      ctx[2] = tag_at(j-1); 
      ctx[1] = tag_at(i+1); 
    } else if (i > (j+1)) {
      ctx[0] = tag_at(i-1); 
      ctx[3] = tag_at(j+1); 
    }

    if (i < j) {
      if (j+1 < static_cast<int>(size()))
        ctx[3] = tag_at(j+1);
      if (i-1 >= 0)
        ctx[0] = tag_at(i-1);
    } else {
      if (j-1 >= 0)
        ctx[2] = tag_at(j-1);
      if (i+1 < static_cast<int>(size()))
        ctx[1] = tag_at(i+1);
    } 

    return ctx;
  } 
  
  private: 
  EChart chart_;
  ESplitChart split_chart_;
  WordId eos_;
};

}
#endif
