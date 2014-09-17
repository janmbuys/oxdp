#pragma once

#include <string>
#include <vector>

#include "corpus/dict.h"

using namespace std;

namespace oxlm {

class CdecLBLMapper {
 public:
  CdecLBLMapper(const boost::shared_ptr<Dict>& dict);

  int convert(int cdec_id) const;

 private:
  void add(int lbl_id, int cdec_id);

  boost::shared_ptr<Dict> dict;
  vector<int> cdec2lbl;
  int kUNKNOWN;
};

} // namespace oxlm
