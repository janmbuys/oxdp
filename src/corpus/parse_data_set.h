#ifndef _CORPUS_PARSE_DATA_SET_H_
#define _CORPUS_PARSE_DATA_SET_H_

#include "corpus/dict.h"
#include "corpus/data_point.h"
#include "corpus/data_set_interface.h"

namespace oxlm {

class ParseDataSet: public DataSetInterface {
  public:

  void addExample(DataPoint example) override;

  DataPoint exampleAt(unsigned i) const override;

  WordId wordAt(unsigned i) const override;

  Words contextAt(unsigned i) const override;

  size_t size() const override;

  void add_word_example(DataPoint example) {
    word_examples_.push_back(example);
  }

  void add_tag_example(DataPoint example) {
    tag_examples_.push_back(example);
  }

  void add_action_example(DataPoint example) {
    action_examples_.push_back(example);
  }

  void extend(const boost::shared_ptr<ParseDataSet>& examples) {
    for (unsigned i = 0; i < examples->size(); ++i) {
      add_word_example(examples->word_example_at(i));
      add_tag_example(examples->tag_example_at(i));
      add_action_example(examples->action_example_at(i));
    }
  }

  void clear() {
    word_examples_.clear();
    tag_examples_.clear();
    action_examples_.clear();
  }

  DataPoint word_example_at(unsigned i) const {
    return word_examples_.at(i);
  }

  DataPoint tag_example_at(unsigned i) const {
    return tag_examples_.at(i);
  }

  DataPoint action_example_at(unsigned i) const {
    return action_examples_.at(i);
  }

  WordId word_at(unsigned i) const {
    return word_examples_.at(i).word;
  }

  WordId tag_at(unsigned i) const {
    return tag_examples_.at(i).word;
  }

  WordId action_at(unsigned i) const {
    return action_examples_.at(i).word;
  }

  Words word_context_at(unsigned i) const {
    return word_examples_.at(i).context;
  }

  Words tag_context_at(unsigned i) const {
    return tag_examples_.at(i).context;
  }

  Words action_context_at(unsigned i) const {
    return action_examples_.at(i).context;
  }

  DataPoints word_examples() const {
    return word_examples_;
  }

  DataPoints tag_examples() const {
    return tag_examples_;
  }

  DataPoints action_examples() const {
    return action_examples_;
  }

  size_t word_example_size() const {
     return word_examples_.size();
  }

  size_t tag_example_size() const {
     return tag_examples_.size();
  }

  size_t action_example_size() const {
     return action_examples_.size();
  }

  private:
  DataPoints word_examples_;
  DataPoints tag_examples_;
  DataPoints action_examples_;
};


}

#endif
