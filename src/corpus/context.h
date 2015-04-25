#ifndef _FEATURE_CONTEXT_H_
#define _FEATURE_CONTEXT_H_

#include <vector>

namespace oxlm {

struct Context {
  Context(const std::vector<int>& words);

  Context(const std::vector<int>& words, const std::vector<int>& tags);

  Context(const std::vector<int>& words, const std::vector<std::vector<int>>& features);
  
  Context(const std::vector<int>& words, const std::vector<int>& tags, const std::vector<std::vector<int>>& features);

  std::vector<int> words;
  std::vector<int> tags;
  std::vector<std::vector<int>> features;
};

} // namespace oxlm

#endif

