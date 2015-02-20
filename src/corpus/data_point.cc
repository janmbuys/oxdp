#include "corpus/data_point.h"

namespace oxlm {

DataPoint::DataPoint(int word, const Context& context): 
  word(word), 
  context(context) {}

} // namespace oxlm
