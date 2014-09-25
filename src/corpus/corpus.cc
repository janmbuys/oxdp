
#include "corpus/dict.h"
#include "corpus/corpus.h"

namespace oxlm {

Corpus::Corpus():
  corpus_()
{
}

Corpus::Corpus(Words corpus):
  corpus_(corpus)
{
}

void Corpus::readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) {
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
}


size_t Corpus::size() const {
  return corpus_.size();
}

size_t Corpus::numTokens() const {
  return size();
}

}

