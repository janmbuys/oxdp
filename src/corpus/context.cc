#include "corpus/context.h"

namespace oxlm {

Context::Context(const std::vector<int>& words):
 words(words), 
 features() {}

Context::Context(const std::vector<int>& words, const std::vector<std::vector<int>> features):
 words(words), 
 features(features) {}

} // namespace oxlm

