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


}

