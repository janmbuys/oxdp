#ifndef _GDP_ESNR_PARSER_H_
#define _GDP_ESNR_PARSER_H_

#include<string>
#include<array>
#include<cstdlib>

#include "corpus/corpus.h"
#include "arc_list.h"

namespace oxlm {

typedef std::array<double, 4> EChartItem;
typedef std::array<WordIndex, 4> ESplitChartItem;
typedef std::vector<std::vector<EChartItem>> EChart;
typedef std::vector<std::vector<ESplitChartItem>> ESplitChart;

class EisnerParser: public Parse {
  public:

  //specifically for generation
  EisnerParser();
   
  EisnerParser(Words tags);

  EisnerParser(Words sent, Words tags);

  EisnerParser(Words sent, Words tags, Indices arcs);

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
    if (s >= t)
      return;
    //std::cout << "(" <<  s << " " << t << " " << complete << " " << left_arc <<  ") ";
    if (!complete) {
      WordIndex r;

      if (left_arc) {
        //left arc
        set_arc(s, t);
        r = left_incomplete_split(s, t);

        /*double left_weight = -std::log(tag_lm.prob(s, parser.tag_context(s, t)));
        if (with_words_) 
          left_weight -= std::log(shift_lm.prob(s, parser.shift_context(s, t)));
        add_weight(left_weight); */
      } else {
        //right arc
        set_arc(t, s);
        r = right_incomplete_split(s, t);

        /* double right_weight = -std::log(tag_lm.prob(t, parser.tag_context(t, s)));
        if (with_words_)
          right_weight -= std::log(shift_lm.prob(t, parser.shift_context(t, s)));
        add_weight(right_weight); */
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
  //will add non-arc-factored context later
 
  Words shift_context(WordIndex i, WordIndex j, WordIndex k) const {
    Words ctx(5, 0);
    if (i >= 0 && i < static_cast<int>(tags_.size()))
      ctx[4] = tag_at(i);
    else
      ctx[4] = 1; //stop word
    ctx[3] = tag_at(j);
    if (k > 0)
      ctx[2] = tag_at(k);
    ctx[1] = word_at(j);
    if (k > 0)
      ctx[0] = word_at(k);

    return ctx;
  }
  
  Words shift_context(WordIndex i, WordIndex j) const {
    Words ctx(3, 0);
    ctx[2] = tag_at(i);
    ctx[1] = tag_at(j);
    ctx[0] = word_at(j);

    return ctx;
  }

  Words tag_context(WordIndex i, WordIndex j, WordIndex k) const {
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

  Words tag_context(WordIndex i, WordIndex j) const {
    //for now, try to replicate Adhi's conditioning context
    Words ctx(7, 0);
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
};

}
#endif
