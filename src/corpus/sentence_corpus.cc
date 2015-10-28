#include "corpus/sentence_corpus.h"

namespace oxlm {

SentenceCorpus::SentenceCorpus() : sentences_() {}

SentenceCorpus::SentenceCorpus(Words sent, int vocab_size)
    : sentences_(1, Sentence(sent)), vocab_size_(vocab_size) {}

Words SentenceCorpus::convertWhitespaceDelimitedLine(
    const std::string& line, const boost::shared_ptr<Dict>& dict, bool frozen) {
  Words out;

  size_t cur = 0;
  size_t last = 0;
  int state = 0;

  // Don't add a start-of-sentence symbol.

  while (cur < line.size()) {
    if (Dict::is_ws(line[cur++])) {
      if (state == 0) continue;
      out.push_back(dict->convert(line.substr(last, cur - last - 1), frozen));
      state = 0;
    } else {
      if (state == 1) continue;
      last = cur - 1;
      state = 1;
    }
  }

  if (state == 1)
    out.push_back(dict->convert(line.substr(last, cur - last), frozen));

  // Add an end-of-sentence symbol if defined.
  if (dict->eos() != -1) out.push_back(dict->eos());

  return out;
}

void SentenceCorpus::readFile(const std::string& filename,
                              const boost::shared_ptr<Dict>& dict,
                              bool frozen) {
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  while (getline(in, line)) {
    Words sents = convertWhitespaceDelimitedLine(line, dict, frozen);
    sentences_.push_back(Sentence(sents));
  }
  vocab_size_ = dict->size();
}

size_t SentenceCorpus::size() const { return sentences_.size(); }

size_t SentenceCorpus::numTokensS() const {
  size_t total = 0;
  for (auto sent : sentences_) {
    total += sent.size();
  }

  return total;
}

size_t SentenceCorpus::numTokens() const {
  size_t total = 0;
  for (auto sent : sentences_) {
    total += sent.size() - 1;
  }

  return total;
}

std::vector<int> SentenceCorpus::unigramCounts() const {
  std::vector<int> counts(vocab_size_, 0);
  for (auto sent : sentences_) {
    for (size_t j = 0; j < sent.size(); ++j) {
      counts[sent.word_at(j)] += 1;
    }
  }

  return counts;
}

}  // namespace oxlm

