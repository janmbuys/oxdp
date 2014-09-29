#include "corpus/parse_data_set.h"

namespace oxlm {

void ParseDataSet::addExample(DataPoint example) {
  add_word_example(example);
}

DataPoint ParseDataSet::exampleAt(unsigned i) const {
  return word_example_at(i);
}

//need better naming
WordId ParseDataSet::wordAt(unsigned i) const {
  return word_at(i);
}

Words ParseDataSet::contextAt(unsigned i) const {
  return word_context_at(i);
}

size_t ParseDataSet::size() const {
  return tag_examples_.size();
}

}

