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

// A corpus of Sentences. 
class SentenceCorpus : public CorpusInterface {
 public:
  SentenceCorpus();

  SentenceCorpus(Words sent, int vocab_size);

  // Parses one line as a sentence. 
  Words convertWhitespaceDelimitedLine(const std::string& line,
                                       const boost::shared_ptr<Dict>& dict,
                                       bool frozen);

  // Reads in sentences from a text file.
  void readFile(const std::string& filename,
                const boost::shared_ptr<Dict>& dict, bool frozen) override;

  Sentence sentence_at(unsigned i) const { return sentences_.at(i); }

  size_t size() const override;

  size_t numTokens() const override;

  // Number of tokens, excluding the EOS token.
  size_t numTokensS() const;

  std::vector<int> unigramCounts() const override;

 private:
  std::vector<Sentence> sentences_;
  int vocab_size_;
};

}  // namespace oxlm

#endif
