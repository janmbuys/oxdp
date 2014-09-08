#ifndef _CORPUS_DATA_SET_H_
#define _CORPUS_DATA_SET_H_

#include "corpus/data_point.h"
#include "corpus/dict.h"
#include "corpus/data_set_interface.h"

namespace oxlm {

class DataSet: public DataSetInterface {
  public:

  void addExample(DataPoint example) override;

  DataPoint exampleAt(unsigned i) const override;

  WordId wordAt(unsigned i) const override;

  Words contextAt(unsigned i) const override;

  size_t size() const override;

  private:
  std::vector<DataPoint> examples_;
};


} // namespace oxlm

#endif


