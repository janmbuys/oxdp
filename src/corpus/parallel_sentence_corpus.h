#ifndef _CORPUS_LLS_CORPUS_H_
#define _CORPUS_LLS_CORPUS_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "corpus/dict.h"
#include "corpus/parallel_sentence.h"
#include "corpus/corpus_interface.h"

namespace oxlm {

class ParallelSentenceCorpus: public CorpusInterface {
  public:
  ParallelSentenceCorpus();
  
  Words convertWhitespaceDelimitedLine(const std::string& line, const boost::shared_ptr<Dict>& dict, 
                                             bool frozen);
  
  Indices convertWhitespaceDelimitedNumberLine(const std::string& line);

  void readInFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen);
  
  void readMonoFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen);
  
  void readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) override;

  ParallelSentence sentence_at(unsigned i) const {
    return sentences_.at(i);
  }

  size_t size() const override;

  size_t numTokens() const override;
  
  size_t numTokensS() const;
  
  std::vector<int> unigramCounts() const override;
 
  private:
  std::vector<ParallelSentence> sentences_;
  int vocab_size_;

};

}

#endif
