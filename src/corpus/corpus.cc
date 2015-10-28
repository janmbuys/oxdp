
#include "corpus/dict.h"
#include "corpus/corpus.h"

namespace oxlm {

Corpus::Corpus() : corpus_() {}

Corpus::Corpus(Words corpus) : corpus_(corpus) {}

void Corpus::readFile(const std::string& filename,
                      const boost::shared_ptr<Dict>& dict, bool frozen) {
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  while (getline(in, line)) {
    std::stringstream line_stream(line);
    std::string token;
    while (line_stream >> token) {
      corpus_.push_back(dict->convert(token, frozen));
    }
    corpus_.push_back(dict->eos());
  }

  vocab_size_ = dict->size();
}

size_t Corpus::size() const { return corpus_.size(); }

size_t Corpus::numTokens() const { return size(); }

std::vector<int> Corpus::unigramCounts() const {
  std::vector<int> counts(vocab_size_, 0);
  for (size_t i = 0; i < size(); ++i) {
    counts[corpus_.at(i)] += 1;
  }

  return counts;
}

}  // namespace oxlm

