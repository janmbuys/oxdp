#include "corpus/context.h"

namespace oxlm {

Context::Context(const std::vector<int>& words) : words(words), features() {}

Context::Context(const std::vector<int>& words, const std::vector<int>& tags)
    : words(words), tags(tags) {}

Context::Context(const std::vector<int>& words,
                 const std::vector<std::vector<int>>& features)
    : words(words), features(features) {}

Context::Context(const std::vector<int>& words, const std::vector<int>& tags,
                 const std::vector<std::vector<int>>& features)
    : words(words), tags(tags), features(features) {}

}  // namespace oxlm

