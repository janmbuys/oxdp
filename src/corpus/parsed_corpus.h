#ifndef _CORPUS_PARSED_CORPUS_H_
#define _CORPUS_PARSED_CORPUS_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "corpus/dict.h"
#include "corpus/parsed_sentence.h"
#include "corpus/corpus_interface.h"

namespace oxlm {

//NB this is not at the moment officially related to corpus
class ParsedCorpus: public CorpusInterface {
  public:
  ParsedCorpus();

  void convertWhitespaceDelimitedConllLine(const std::string& line, 
      const boost::shared_ptr<Dict>& dict, Words* sent_out, Words* tags_out, Indices* arcs_out, bool frozen);

  void readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) override;

  ParsedSentence sentence_at(unsigned i) const {
    return sentences_.at(i);
  }

  size_t size() const override;

  size_t numTokens() const override;

  private:
  std::vector<ParsedSentence> sentences_;

};

}

#endif
