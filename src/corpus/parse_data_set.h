#ifndef _CORPUS_PARSE_DATA_SET_H_
#define _CORPUS_PARSE_DATA_SET_H_

#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "corpus/data_set.h"

namespace oxlm {

class ParseDataSet {
  public:

  ParseDataSet();

  void add_word_example(DataPoint example) {
    word_examples_->addExample(example);
  }

  void add_tag_example(DataPoint example) {
    tag_examples_->addExample(example);
  }

  void add_action_example(DataPoint example) {
    action_examples_->addExample(example);
  }

  void extend(const boost::shared_ptr<ParseDataSet>& examples) {
    for (unsigned i = 0; i < examples->word_example_size(); ++i) 
      add_word_example(examples->word_example_at(i));
    for (unsigned i = 0; i < examples->tag_example_size(); ++i) 
      add_tag_example(examples->tag_example_at(i));
    for (unsigned i = 0; i < examples->action_example_size(); ++i) 
      add_action_example(examples->action_example_at(i));
  }

  void clear() {
    word_examples_->clear();
    tag_examples_->clear();
    action_examples_->clear();
  }

  DataPoint word_example_at(unsigned i) const {
    return word_examples_->exampleAt(i);
  }

  DataPoint tag_example_at(unsigned i) const {
    return tag_examples_->exampleAt(i);
  }

  DataPoint action_example_at(unsigned i) const {
    return action_examples_->exampleAt(i);
  }

  WordId word_at(unsigned i) const {
    return word_examples_->wordAt(i);
  }

  WordId tag_at(unsigned i) const {
    return tag_examples_->wordAt(i);
  }

  WordId action_at(unsigned i) const {
    return action_examples_->wordAt(i);
  }

  Context word_context_at(unsigned i) const {
    return word_examples_->contextAt(i);
  }

  Context tag_context_at(unsigned i) const {
    return tag_examples_->contextAt(i);
  }

  Context action_context_at(unsigned i) const {
    return action_examples_->contextAt(i);
  }

  boost::shared_ptr<DataSet> word_examples() const {
    return word_examples_;
  }

  boost::shared_ptr<DataSet> tag_examples() const {
    return tag_examples_;
  }

  boost::shared_ptr<DataSet> action_examples() const {
    return action_examples_;
  }

  size_t word_example_size() const {
     return word_examples_->size();
  }

  size_t tag_example_size() const {
     return tag_examples_->size();
  }

  size_t action_example_size() const {
     return action_examples_->size();
  }

  private:
  boost::shared_ptr<DataSet> word_examples_;
  boost::shared_ptr<DataSet> tag_examples_;
  boost::shared_ptr<DataSet> action_examples_;
};


}

#endif
