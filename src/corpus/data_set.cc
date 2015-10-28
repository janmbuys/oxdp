#include "corpus/data_set.h"

namespace oxlm {

void DataSet::addExample(DataPoint example) { examples_.push_back(example); }

void DataSet::clear() { examples_.clear(); }

DataPoint DataSet::exampleAt(unsigned i) const { return examples_.at(i); }

WordId DataSet::wordAt(unsigned i) const { return examples_.at(i).word; }

WordId DataSet::tagAt(unsigned i) const { return examples_.at(i).tag; }

int DataSet::sentenceIdAt(unsigned i) const {
  return examples_.at(i).sentence_id;
}

Context DataSet::contextAt(unsigned i) const { return examples_.at(i).context; }

size_t DataSet::size() const { return examples_.size(); }

}  // namespace oxlm
