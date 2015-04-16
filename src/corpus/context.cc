#include "corpus/context.h"

namespace oxlm {

Context::Context(const std::vector<int>& words):
 words(words), 
 features() {}

Context::Context(const std::vector<int>& words, const std::vector<std::vector<int>> features):
 words(words), 
 features(features) {}

Context::Context(const std::vector<int>& words, const std::vector<int> tags):
 words(words), 
 features(1, tags) {
   //for (auto tag: tags)
   //  features.push_back(std::vector<int>(1, tag));
 }

} // namespace oxlm

