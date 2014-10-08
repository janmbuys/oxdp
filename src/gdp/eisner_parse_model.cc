#include "gdp/eisner_parse_model.h"

namespace oxlm {

template<class ParsedWeights>
EisnerParser EisnerParseModel<ParsedWeights>::parseSentence(const ParsedSentence& sent, 
        const boost::shared_ptr<ParsedWeights>& weights) {
  EisnerParser parser(sent); 

  for (WordIndex k = 1; k < sent.size(); ++k) {
    //std::cout << k << " ";
    for (WordIndex s = 0; ((s + k) < sent.size()); ++s) {
      WordIndex t = s + k;

      //assign weights and splits to the four things in chart item (s, t)

      Real left_incomp_w = L_MAX;
      WordIndex split = -1;
      for (WordIndex r = s; r < t; ++r) {
        WordIndex prev_child = parser.left_complete_split(r+1, t); // + 1;
        Real left_weight = weights->predictTag(parser.tag_at(s), 
                parser.tagContext(s, t, prev_child)) + weights->predictWord(
                   parser.word_at(s), parser.wordContext(s, t, prev_child));
        //s has no more right children
        prev_child = parser.right_complete_split(s, r);
        Real stop_weight = weights->predictTag(parser.eos(), parser.tagContext(sent.size(), 
                    s, prev_child)) + weights->predictWord(parser.eos(), 
                      parser.wordContext(sent.size(), s, prev_child));
        
        Real w = parser.right_complete_weight(s, r)
                   + parser.left_complete_weight(r+1, t)
                   + left_weight
                   + stop_weight;
        if (w < left_incomp_w) {
          left_incomp_w = w;
          split = r;
        }
      }
      parser.set_left_incomplete_weight(s, t, left_incomp_w);
      parser.set_left_incomplete_split(s, t, split);
      
      Real right_incomp_w = L_MAX;
      split = -1;
      for (WordIndex r = s; r < t; ++r) {
        WordIndex prev_child = parser.right_complete_split(s, r);
        Real right_weight = weights->predictTag(parser.tag_at(t), 
                 parser.tagContext(t, s, prev_child)) + weights->predictWord(
                   parser.word_at(t), parser.wordContext(t, s, prev_child));
        
        //t has no more left children
        prev_child = parser.left_complete_split(r+1, t);
        Real stop_weight = weights->predictTag(parser.eos(), 
                parser.tagContext(-1, t, prev_child)) + weights->predictWord(parser.eos(), 
                  parser.wordContext(-1, t, prev_child));

        Real w = parser.right_complete_weight(s, r)
                   + parser.left_complete_weight(r+1, t)
                   + right_weight
                   + stop_weight;
        if (w < right_incomp_w) {
          right_incomp_w = w;
          split = r;
        }
      }
      
      parser.set_right_incomplete_weight(s, t, right_incomp_w);
      parser.set_right_incomplete_split(s, t, split);

      Real left_comp_w = L_MAX;
      split = -1;
      for (WordIndex r = s; r < t; ++r) {
        //r has no more left children
        WordIndex prev_child = parser.left_complete_split(s, r);
        Real stop_weight = weights->predictTag(parser.eos(), parser.tagContext(-1, r, 
                    prev_child)) + weights->predictWord(parser.eos(), parser.wordContext(-1, r, 
                        prev_child));      
        Real w = parser.left_complete_weight(s, r)
                   + parser.left_incomplete_weight(r, t)
                   + stop_weight;
        if (w < left_comp_w) {
          left_comp_w = w;
          split = r;
        }
      }
      parser.set_left_complete_weight(s, t, left_comp_w);
      parser.set_left_complete_split(s, t, split);

      Real right_comp_w = L_MAX;
      split = -1;
      for (WordIndex r = s + 1; r <= t; ++r) {
        //r has no more right children
        WordIndex prev_child = parser.right_complete_split(r, t);
        Real stop_weight = weights->predictTag(parser.eos(), parser.tagContext(sent.size(), r, 
                    prev_child)) + weights->predictWord(parser.eos(), 
                      parser.wordContext(sent.size(), r, prev_child));

        //for sentence completion, root has no more right children
        if (s==0 && t==(sent.size()-1)) {
          stop_weight += weights->predictTag(parser.eos(), parser.tagContext(sent.size(), s, r))
              + weights->predictWord(parser.eos(), parser.wordContext(sent.size(), s, r));
        }

        Real w = parser.right_incomplete_weight(s, r)
             + parser.right_complete_weight(r, t)
             + stop_weight;
        if (w < right_comp_w) {
          right_comp_w = w;
          split = r;
        }
      }
      parser.set_right_complete_weight(s, t, right_comp_w);
      parser.set_right_complete_split(s, t, split);
    }
  } 
 
  Real best_weight = parser.right_complete_weight(0, sent.size()-1);
  std::cout << best_weight << " ";

  //recover and assign best parse
  //must have right arc from node 0 
  //indexes are inclusive
  parser.recoverParseTree(0, sent.size() - 1, true, false);
  
  scoreSentence(&parser, weights);
  //parser.print_chart();

  std::cout << parser.weight() << " ";
  parser.print_arcs();
  return parser;
}

template<class ParsedWeights>
void EisnerParseModel<ParsedWeights>::scoreSentence(EisnerParser* parser, const boost::shared_ptr<ParsedWeights>& weights) {
  parser->reset_weight();

  for (WordIndex i = 1; (i < static_cast<int>(parser->size())); ++i) {
    //dependent i, head j
    WordIndex j = parser->arc_at(i);
    if (j == -1)
      continue;
    WordIndex prev_child = 0;
    if (i < j)
      prev_child = parser->prev_left_child_at(i, j);
    else if (j < i)
      prev_child = parser->prev_right_child_at(i, j);

    Real weight = weights->predictTag(parser->tag_at(i), parser->tagContext(i, j, 
                prev_child)) + weights->predictWord(parser->word_at(i), 
                  parser->wordContext(i, j, prev_child));
    parser->add_weight(weight);
    //std::cout << "(" << dict.lookupTag(i) << ", " << dict.lookupTag(j) << ", " << weight << ") ";
  }  

  for (WordIndex j = 0; (j < static_cast<int>(parser->size())); ++j) {
    WordIndex prev_child = parser->leftmost_child_at(j);
    Real weight;
    //root only generates right children
    if (j > 0) {
      weight = weights->predictTag(parser->eos(), parser->tagContext(-1, j, prev_child))
          + weights->predictWord(parser->eos(), parser->wordContext(-1, j, prev_child));
      parser->add_weight(weight);
    }

    prev_child = parser->rightmost_child_at(j);
    weight = weights->predictTag(parser->eos(), parser->tagContext(parser->size(), j, prev_child))
        + weights->predictWord(parser->eos(), parser->wordContext(parser->size(), j, prev_child));
    parser->add_weight(weight);
  }
  //std::cout << std::endl;
  //std::cout << parser->weight() << std::endl;
} 

template<class ParsedWeights>
void EisnerParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  EisnerParser parse(static_cast<TaggedSentence>(sent));
  //add arcs
  for (unsigned i = 0; i < sent.size(); ++i)
    if (sent.arc_at(i) >= 0)
      parse.set_arc(i, sent.arc_at(i));

  parse.extractExamples(examples);
}

template<class ParsedWeights>
void EisnerParseModel<ParsedWeights>::extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
 extractSentence(sent, examples);
 //scoreSentence(&parse, weights);
}

template<class ParsedWeights>
void EisnerParseModel<ParsedWeights>::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<ParseDataSet>& examples) {
  //TODO sample a sentence
  EisnerParser parse = parseSentence(sent, weights);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
void EisnerParseModel<ParsedWeights>::extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples) {
  EisnerParser parse = parseSentence(sent, weights);
  parse.extractExamples(examples);
}

template<class ParsedWeights>
Real EisnerParseModel<ParsedWeights>::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) {
  EisnerParser parse = parseSentence(sent, weights);
  acc_counts->countAccuracy(parse, sent);
  return parse.weight(); 
}   

template<class ParsedWeights>
Real EisnerParseModel<ParsedWeights>::evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size) {
  //TODO sample a sentence
  EisnerParser parse = parseSentence(sent, weights);
  acc_counts->countAccuracy(parse, sent);
  return parse.weight(); 
}

template class EisnerParseModel<ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>;
template class EisnerParseModel<ParsedPypWeights<tagLMOrderE, 1>>;
template class EisnerParseModel<ParsedFactoredWeights>;

}
