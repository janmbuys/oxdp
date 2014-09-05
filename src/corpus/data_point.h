#ifndef _CORPUS_DATA_POINT_H_
#define _CORPUS_DATA_POINT_H_

#include <vector>

namespace oxlm {

struct DataPoint {
  DataPoint(int word, const std::vector<int>& context);

  int word;
  std::vector<int> context;
};

//for now
typedef std::vector<DataPoint> DataSet;

} // namespace oxlm

#endif
