#ifndef _CORPUS_DATA_POINT_H_
#define _CORPUS_DATA_POINT_H_

#include <vector>
#include "corpus/context.h"

namespace oxlm {

struct DataPoint {
  DataPoint(int word, const Context& context);
  
  DataPoint(int word, const Context& context, int id);
  
  DataPoint(int word, int tag, const Context& context);
  
  DataPoint(int word, const Context& context, const std::vector<int>& features, int id);

  DataPoint(int word, int tag, const Context& context, const std::vector<int>& features, int id);

  int word;
  int tag;
  Context context;
  std::vector<int> features; //output features
  int sentence_id;
};

} // namespace oxlm

#endif
