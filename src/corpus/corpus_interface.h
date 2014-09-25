#ifndef _CORPUS_CORPUS_I_H_
#define _CORPUS_CORPUS_I_H_

namespace oxlm {

class CorpusInterface {
  public:
  virtual void readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) = 0;

  virtual size_t size() const = 0;

  virtual size_t numTokens() const = 0;

  virtual std::vector<int> unigramCounts() const = 0;

  virtual ~CorpusInterface() {}
};

}

#endif
