#ifndef _CORPUS_DATA_SET_H_
#define _CORPUS_DATA_SET_H_

#include "corpus/data_point.h"
#include "corpus/dict.h"

namespace oxlm {

class DataSet {
  public:

  void addExample(DataPoint example);

  void clear();

  DataPoint exampleAt(unsigned i) const;

  WordId wordAt(unsigned i) const;

  Words contextAt(unsigned i) const;

  size_t size() const;

  private:
  std::vector<DataPoint> examples_;
};


} // namespace oxlm

#endif


