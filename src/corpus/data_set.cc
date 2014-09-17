#include "corpus/data_set.h"

namespace oxlm {

void  DataSet::addExample(DataPoint example) {
  examples_.push_back(example);
}

DataPoint DataSet::exampleAt(unsigned i) const {
  return examples_.at(i);
}

WordId DataSet::wordAt(unsigned i) const {
  return examples_.at(i).word;
}

Words DataSet::contextAt(unsigned i) const {
  return examples_.at(i).context;
}

size_t DataSet::size() const {
  return examples_.size();
}

}