#ifndef _CORPUS_CORPUS_H_
#define _CORPUS_CORPUS_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>

#include <boost/shared_ptr.hpp>

#include "corpus/sentence.h"

namespace oxlm {

class Corpus {
  public:
  Corpus();

  Words convertWhitespaceDelimitedLine(const std::string& line, boost::shared_ptr<Dict>& dict, 
                                             bool frozen);

  virtual void readFile(const std::string& filename, boost::shared_ptr<Dict>& dict, bool frozen);

  Sentence sentence_at(unsigned i) const {
    return sentences_.at(i);
  }

  size_t size() const {
    return sentences_.size();
  }

  private:
  std::vector<Sentence> sentences_;


};

}

#endif
