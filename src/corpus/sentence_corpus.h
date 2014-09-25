#ifndef _CORPUS_S_CORPUS_H_
#define _CORPUS_S_CORPUS_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "corpus/dict.h"
#include "corpus/sentence.h"
#include "corpus/corpus_interface.h"

namespace oxlm {

class SentenceCorpus: public CorpusInterface {
  public:
  SentenceCorpus();
  
  SentenceCorpus(Words sent);

  Words convertWhitespaceDelimitedLine(const std::string& line, const boost::shared_ptr<Dict>& dict, 
                                             bool frozen);

  void readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) override;

  Sentence sentence_at(unsigned i) const {
    return sentences_.at(i);
  }

  size_t size() const override;

  size_t numTokens() const override;
  
  std::vector<int> unigramCounts() const override;
 
  private:
  std::vector<Sentence> sentences_;
  int vocab_size_;

};

}

#endif
