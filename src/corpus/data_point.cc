#include "data_point.h"

namespace oxlm {

DataPoint::DataPoint(const vector<int>& context, int word): 
  context(context), 
  word(word)
{
}

} // namespace oxlm
