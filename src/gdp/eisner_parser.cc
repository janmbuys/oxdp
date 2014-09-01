#include "eisner_parser.h"

namespace oxlm {

EisnerParser::EisnerParser():
  Parser(),
  chart_(),
  split_chart_()
  {
  }      
   
EisnerParser::EisnerParser(Words tags):
  Parser(tags),
  chart_(tags.size(), std::vector<EChartItem>(tags.size(), EChartItem())),
  split_chart_(tags.size(), std::vector<ESplitChartItem>(tags.size(), ESplitChartItem{-1, -1, -1, -1}))
  {
  }
   
EisnerParser::EisnerParser(Words sent, Words tags):
  Parser(tags),
  chart_(sent.size(), std::vector<EChartItem>(sent.size(), EChartItem())),
  split_chart_(sent.size(), std::vector<ESplitChartItem>(sent.size(), ESplitChartItem{-1, -1, -1, -1}))
  {
  }

EisnerParser::EisnerParser(Words sent, Words tags, Indices arcs):
  Parser(sent, tags, arcs),
  chart_(sent.size(), std::vector<EChartItem>(sent.size(), EChartItem())),
  split_chart_(sent.size(), std::vector<ESplitChartItem>(sent.size(), ESplitChartItem{-1, -1, -1, -1}))
  {
  }

EisnerParser::EisnerParser(const ParsedSentence& parse):
  Parser(parse),
  chart_(parse.size(), std::vector<EChartItem>(parse.size(), EChartItem())),
  split_chart_(parse.size(), std::vector<ESplitChartItem>(parse.size(), ESplitChartItem{-1, -1, -1, -1})) {}

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
  Words ctx(5, 0);
  if (i >= 0 && i < static_cast<int>(size()))
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
  
Words EisnerParser::wordContext(WordIndex i, WordIndex j) const {
  Words ctx(3, 0);
  ctx[2] = tag_at(i);
  ctx[1] = tag_at(j);
  ctx[0] = word_at(j);

  return ctx;
}

Words EisnerParser::tagContext(WordIndex i, WordIndex j, WordIndex k) const {
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

Words EisnerParser::tagContext(WordIndex i, WordIndex j) const {
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

}

