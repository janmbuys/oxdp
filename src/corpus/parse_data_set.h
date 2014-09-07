#ifndef _CORPUS_PARSE_DATA_SET_H_
#define _CORPUS_PARSE_DATA_SET_H_

#include "data_point.h"

namespace oxlm {

class ParseDataSet {
  public:

  ParseDataSet();

  void add_word_example(DataPoint example) {
    word_examples_.push_back(example);
  }

  void add_tag_example(DataPoint example) {
    tag_examples_.push_back(example);
  }

  void add_action_example(DataPoint example) {
    action_examples_.push_back(example);
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
  DataSet word_examples_;
  DataSet tag_examples_;
  DataSet action_examples_;

}


}

#endif
