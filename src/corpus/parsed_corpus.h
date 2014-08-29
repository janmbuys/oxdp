#ifndef _CORPUS_PARSED_CORPUS_H_
#define _CORPUS_PARSED_CORPUS_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>

#include <boost/shared_ptr.hpp>

#include "corpus/parsed_sentence.h"

namespace oxlm {

//NB this is not at the moment officially related to corpus
class ParsedCorpus {
  public:
  ParsedCorpus();

  void convertWhitespaceDelimitedConllLine(const std::string& line, 
      boost::shared_ptr<Dict>& dict, Words* sent_out, Words* tags_out, Indices* arcs_out, bool frozen);

  void readFile(const std::string& filename, boost::shared_ptr<Dict>& dict, bool frozen);

  Sentence sentence_at(unsigned i) const {
    return sentences_.at(i);
  }

  private:
  std::vector<ParsedSentence> sentences_;

};

}

#endif
