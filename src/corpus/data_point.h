#ifndef _CORPUS_DATA_POINT_H_
#define _CORPUS_DATA_POINT_H_

#include <vector>

namespace oxlm {

struct DataPoint {
  DataPoint(const std::vector<int>& context, int word);

  std::vector<int> context;
  int word;
};

//for now
typedef std::vector<DataPoint> DataSet;

} // namespace oxlm

#endif
