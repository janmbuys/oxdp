#include "corpus/parallel_sentence_corpus.h"

namespace oxlm {

ParallelSentenceCorpus::ParallelSentenceCorpus():
  sentences_()
{
}

Words ParallelSentenceCorpus::convertWhitespaceDelimitedLine(const std::string& line, 
        const boost::shared_ptr<Dict>& dict, bool frozen) {
  Words out;

  size_t cur = 0;
  size_t last = 0;
  int state = 0;
     
  //don't add start of sentence symbol

  while (cur < line.size()) {
    if (Dict::is_ws(line[cur++])) {
      if (state == 0) 
        continue;
      out.push_back(dict->convert(line.substr(last, cur - last - 1), frozen));
      dict->convertFeature(line.substr(last, cur - last - 1), frozen);
      state = 0;
    } else {
      if (state == 1) 
        continue;
      last = cur - 1;
      state = 1;
    }
  }

  if (state == 1) {
    out.push_back(dict->convert(line.substr(last, cur - last), frozen));
    dict->convertFeature(line.substr(last, cur - last), frozen);
  }

  //add end of sentence symbol if defined 
  if (dict->eos() != -1) 
    out.push_back(dict->eos()); 

  return out;
}

Indices ParallelSentenceCorpus::convertWhitespaceDelimitedNumberLine(const std::string& line) {
  Indices out;

  size_t cur = 0;
  size_t last = 0;
  int state = 0;
     
  //don't add start of sentence symbol

  while (cur < line.size()) {
    if (Dict::is_ws(line[cur++])) {
      if (state == 0) 
        continue;
      out.push_back(std::stoi(line.substr(last, cur - last - 1)));
      state = 0;
    } else {
      if (state == 1) 
        continue;
      last = cur - 1;
      state = 1;
    }
  }

  if (state == 1)
    out.push_back(std::stoi(line.substr(last, cur - last)));

  return out;
}

void ParallelSentenceCorpus::readInFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) {
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  while(getline(in, line)) {
    Words in_sent = convertWhitespaceDelimitedLine(line, dict, frozen);
    sentences_.push_back(ParallelSentence(in_sent));
  }
  vocab_size_ = dict->size();
}

void ParallelSentenceCorpus::readMonoFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) {
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  while(getline(in, line)) {
    Words in_sent = convertWhitespaceDelimitedLine(line, dict, frozen);
    Words out_sent = convertWhitespaceDelimitedLine(line, dict, frozen);
    Indices align(in_sent.size());
    iota(align.begin(), align.end(), 0);
    sentences_.push_back(ParallelSentence(in_sent, out_sent, align));
  }
  vocab_size_ = dict->size();
}

void ParallelSentenceCorpus::readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) {
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  while(getline(in, line)) {
    Words in_sent = convertWhitespaceDelimitedLine(line, dict, frozen);
    getline(in, line);
    Words out_sent = convertWhitespaceDelimitedLine(line, dict, frozen);
    getline(in, line);
    Indices align = convertWhitespaceDelimitedNumberLine(line);
    if (dict->eos() != -1) 
      align.push_back(in_sent.size() - 1); //add EOS alignment
    sentences_.push_back(ParallelSentence(in_sent, out_sent, align));
  }
  vocab_size_ = dict->size();
}


size_t ParallelSentenceCorpus::size() const {
  return sentences_.size();
}

//for output side
size_t ParallelSentenceCorpus::numTokensS() const {
  size_t total = 0;
  for (auto sent: sentences_)
    total += sent.out_size(); //exclude EOS

  return total;
}

size_t ParallelSentenceCorpus::numTokens() const {
  size_t total = 0;
  for (auto sent: sentences_)
    total += sent.out_size() - 1; //exclude EOS

  return total;
}

std::vector<int> ParallelSentenceCorpus::unigramCounts() const {
  std::vector<int> counts(vocab_size_, 0);
  for (auto sent: sentences_) {
    for (size_t j = 0; j < sent.out_size(); ++j)
      counts[sent.out_word_at(j)] += 1;
  }

  return counts;
}


}

