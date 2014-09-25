#ifndef _CORPUS_DATA_SET_I_H_
#define _CORPUS_DATA_SET_I_H_

#include "corpus/data_point.h"

namespace oxlm {

class DataSetInterface {
  public:
  virtual void addExample(DataPoint example) = 0;

  virtual DataPoint exampleAt(unsigned i) const = 0;

  virtual int wordAt(unsigned i) const = 0;

  virtual std::vector<int> contextAt(unsigned i) const = 0;

  virtual size_t size() const = 0;

  virtual ~DataSetInterface() {}
};

}

#endif
