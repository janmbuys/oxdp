#ifndef _CORPUS_CORPUS_H_
#define _CORPUS_CORPUS_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <boost/serialization/vector.hpp>

#include "corpus/dict.h"
#include "corpus/sentence.h"
#include "corpus/corpus_interface.h"

namespace oxlm {

class Corpus: public CorpusInterface {
  public:
  Corpus();

  Corpus(Words corpus);

  void readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) override;

  WordId at(unsigned i) const {
    return corpus_.at(i);
  }

  size_t size() const override;

  size_t numTokens() const override;
 
  std::vector<int> unigramCounts() const override;

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & corpus_;
  }

  private:
  Words corpus_;
  int vocab_size_;

};

}

#endif
