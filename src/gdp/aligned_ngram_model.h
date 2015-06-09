#ifndef _CORPUS_AL_NGRAM_MODEL_H_
#define _CORPUS_AL_NGRAM_MODEL_H_

#include "corpus/utils.h"
#include "corpus/parallel_sentence.h"
#include "corpus/data_set.h"
#include "corpus/corpus.h"
#include "gdp/parser.h"

#include "pyp/pyp_weights.h"
#include "lbl/weights.h"
#include "lbl/factored_weights.h"

namespace oxlm {

template<class Weights>
class AlignedNGramModel {
  public:
  AlignedNGramModel(unsigned out_ctx_size, unsigned in_window_size, WordId sos, WordId eos);

  Context extractContext(const ParallelSentence& sent, int position);

  //void extract(const boost::shared_ptr<Corpus> corpus, int position,     
  //        const boost::shared_ptr<DataSet>& examples);

  //Real evaluate(const boost::shared_ptr<Corpus> corpus, int position, 
  //        const boost::shared_ptr<Weights>& weights); 

  void extractSentence(const ParallelSentence& sent, 
          const boost::shared_ptr<DataSet>& examples);

  //return likelihood
  Real evaluateSentence(const ParallelSentence& sent, 
          const boost::shared_ptr<Weights>& weights);

  ParallelSentence generateSentence(const Words& in_sent,
      const boost::shared_ptr<Weights>& weights, int beam_size, int max_beam_increment);

  private:
  unsigned out_ctx_size_;
  unsigned in_window_size_;
  WordId sos_;
  WordId eos_;
          
};

}

#endif
