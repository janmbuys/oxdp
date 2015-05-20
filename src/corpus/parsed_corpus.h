#ifndef _CORPUS_PARSED_CORPUS_H_
#define _CORPUS_PARSED_CORPUS_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "corpus/dict.h"
#include "corpus/model_config.h"
#include "corpus/parsed_sentence.h"
#include "corpus/corpus_interface.h"

namespace oxlm {

class ParsedCorpus: public CorpusInterface {
 public:
  ParsedCorpus(const boost::shared_ptr<ModelConfig>& config);

  void convertWhitespaceDelimitedConllLine(const std::string& line, 
      const boost::shared_ptr<Dict>& dict, Words* sent_out, Words* tags_out, WordsList* features_out, Indices* arcs_out, Words* labels_out, bool frozen);

  void convertWhitespaceDelimitedTxtLine(const std::string& line, 
      const boost::shared_ptr<Dict>& dict, Words* sent_out, WordsList* features_out, bool frozen);

  void readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) override;

  void readTxtFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen);

  ParsedSentence sentence_at(unsigned i) const {
    //std::cout << i << " " << sentences_.at(i).word_at(1) << std::endl;
    return sentences_.at(i);
  }

  void add_sentence(ParsedSentence sent) {
    sentences_.push_back(sent);
  }

  void set_arcs_at(unsigned j, const ParsedSentence& parse) {
    for (unsigned i = 0; i < parse.size(); ++i) {
      sentences_.at(j).set_arc(i, parse.arc_at(i));
    }
  }

  void set_labels_at(unsigned j, const ParsedSentence& parse) {
    for (unsigned i = 0; i < parse.size(); ++i) {
      sentences_.at(j).set_label(i, parse.label_at(i));
    }
  }


  size_t size() const override;

  size_t numTokens() const override;

  std::vector<int> unigramCounts() const override;

  std::vector<int> actionCounts() const;

  std::vector<int> tagCounts() const;

 private:
  std::vector<ParsedSentence> sentences_;
  boost::shared_ptr<ModelConfig> config_;
};

}

#endif
