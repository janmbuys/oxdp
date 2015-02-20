#include "eisner_parser.h"

namespace oxlm {

EisnerParser::EisnerParser():
  Parser(),
  chart_(),
  split_chart_(),
  eos_{1}
  {
  }      

  EisnerParser::EisnerParser(const TaggedSentence& parse):
  Parser(parse),
  chart_(parse.size(), std::vector<EChartItem>(parse.size(), EChartItem())),
  split_chart_(parse.size(), std::vector<ESplitChartItem>(parse.size(), 
                         ESplitChartItem{-1, -1, -1, -1})),
  eos_{1} 
  {
  }

void EisnerParser::recoverParseTree(WordIndex s, WordIndex t, bool complete, 
       bool left_arc) {
  if (s >= t)
    return;
  //std::cout << "(" <<  s << " " << t << " " << complete << " " << left_arc <<  ") ";
  if (!complete) {
    WordIndex r;

    if (left_arc) {
      set_arc(s, t);
      r = left_incomplete_split(s, t);
    } else {
      //right arc
      set_arc(t, s);
      r = right_incomplete_split(s, t);
    }

    recoverParseTree(s, r, true, false);
    recoverParseTree(r+1, t, true, true);
  } else {
    if (left_arc) {
      WordIndex r = left_complete_split(s, t);
      recoverParseTree(s, r, true, true);
      recoverParseTree(r, t, false, true);
    } else {
      WordIndex r = right_complete_split(s, t);
      recoverParseTree(s, r, false, false);
      recoverParseTree(r, t, true, false);
    }
  }
}

Words EisnerParser::wordContext(WordIndex i, WordIndex j, WordIndex k) const {
  return word_only_context(i, j, k); //order 6
  //return word_tag_context(i, j, k); //order 6
  //return word_tag_context(i, j); //order 4
}

Words EisnerParser::tagContext(WordIndex i, WordIndex j, WordIndex k) const {
  return tag_only_context(i, j, k); //order 7
  //return tag_only_context(i, j); //order 8
}

void EisnerParser::extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const {
  //we should actually extract training examples in the same order as generation,
  //but this shouldn't matter too much in practice
  for (WordIndex i = 1; i < static_cast<int>(size()); ++i) {
    //training example head j to i
    WordIndex j = arc_at(i);
    if (j == -1)
      continue;
    WordIndex prev_child = 0;
    if (i < j)
      prev_child = prev_left_child_at(i, j);
    else if (j < i)
      prev_child = prev_right_child_at(i, j);

    examples->add_tag_example(DataPoint(tag_at(i), tagContext(i, j, prev_child)));
    //if (!(word_examples == nullptr)) 
    examples->add_word_example(DataPoint(word_at(i), wordContext(i, j, prev_child)));
  } 

  for (WordIndex j = 0; j < static_cast<int>(size()); ++j) {
    //training examples: no further left children
    WordIndex prev_child = leftmost_child_at(j);
 
    examples->add_tag_example(DataPoint(eos(), tagContext(-1, j, prev_child)));
    //if (!(word_examples == nullptr)) 
    examples->add_word_example(DataPoint(eos(), wordContext(-1, j, prev_child)));
    
    //training examples: no further right children
    prev_child = rightmost_child_at(j);

    examples->add_tag_example(DataPoint(eos(), tagContext(size(), j, prev_child)));
    //if (!(word_examples == nullptr)) 
    examples->add_word_example(DataPoint(eos(), wordContext(size(), j, prev_child)));
  } 
}

}

