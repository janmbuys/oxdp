#include "corpus/data_point.h"

namespace oxlm {

DataPoint::DataPoint(const std::vector<int>& context, int word): 
  context(context), 
  word(word) {}

} // namespace oxlm
