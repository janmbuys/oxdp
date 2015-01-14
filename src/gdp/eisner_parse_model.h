#ifndef _GDP_ESNR_PARSE_MODEL_H_
#define _GDP_ESNR_PARSE_MODEL_H_

#include "utils/random.h"
#include "corpus/utils.h"

#include "gdp/eisner_parser.h"
#include "gdp/accuracy_counts.h"

#include "pyp/parsed_pyp_weights.h"
#include "pyp/parsed_lex_pyp_weights.h"
#include "pyp/parsed_chlex_pyp_weights.h"
#include "lbl/parsed_factored_weights.h"
#include "lbl/parsed_weights.h"

namespace oxlm {

template<class ParsedWeights>
class EisnerParseModel {
  public:

  EisnerParseModel(boost::shared_ptr<ModelConfig> config);

  EisnerParser parseSentence(const ParsedSentence& sent, const boost::shared_ptr<ParsedWeights>& weights);

  void scoreSentence(EisnerParser* parser, const boost::shared_ptr<ParsedWeights>& weights);
  
  void extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParseDataSet>& examples);

  void extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples);

  void extractSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<ParseDataSet>& examples);

  void extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<ParseDataSet>& examples);

  void extractSentenceUnsupervised(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<ParseDataSet>& examples);

  Parser evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size); 

  Parser evaluateSentence(const ParsedSentence& sent, 
          const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng, const boost::shared_ptr<AccuracyCounts>& acc_counts,
          size_t beam_size); 

  EisnerParser generateSentence(const boost::shared_ptr<ParsedWeights>& weights, 
          MT19937& eng);

  private:
  boost::shared_ptr<ModelConfig> config_;
};

}

#endif
