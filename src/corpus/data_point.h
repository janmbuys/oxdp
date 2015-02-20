#ifndef _CORPUS_DATA_POINT_H_
#define _CORPUS_DATA_POINT_H_

#include <vector>
#include "corpus/context.h"

namespace oxlm {

struct DataPoint {
  DataPoint(int word, const Context& context);

  int word;
  Context context;
};

} // namespace oxlm

#endif
