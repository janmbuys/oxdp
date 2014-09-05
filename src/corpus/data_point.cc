#include "corpus/data_point.h"

namespace oxlm {

DataPoint::DataPoint(int word, const std::vector<int>& context): 
  word(word), 
  context(context) {}

} // namespace oxlm
