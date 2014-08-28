#include <vector>

namespace oxlm {

struct DataPoint {
  DataPoint(const Words& context, WordId word);

  Words context;
  WordId word;
};

//for now
typedef std::vector<DataPoint> DataSet;

} // namespace oxlm

