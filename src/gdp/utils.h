#ifndef _GDP_UTILS_H_
#define _GDP_UTILS_H_

#include <vector>
#include "corpus/utils.h"

namespace oxlm {

enum class kAction : WordId { sh, la, ra, re, la2, ra2 };
typedef std::vector<kAction> ActList;

}  // namespace oxlm

#endif
