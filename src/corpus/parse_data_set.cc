#include "corpus/parse_data_set.h"

namespace oxlm {

ParseDataSet::ParseDataSet() {
  word_examples_ = boost::make_shared<DataSet>();
  tag_examples_ = boost::make_shared<DataSet>();
  action_examples_ = boost::make_shared<DataSet>();
}

}  //namespace oxlm

