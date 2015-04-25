#include "corpus/data_point.h"

namespace oxlm {

DataPoint::DataPoint(int word, const Context& context): 
  word(word), 
  context(context) {}

DataPoint::DataPoint(int word, const Context& context, int id): 
  word(word), 
  context(context),
  sentence_id(id) {}

DataPoint::DataPoint(int word, int tag, const Context& context): 
  word(word), 
  tag(tag),
  context(context) {}

DataPoint::DataPoint(int word, const Context& context, const std::vector<int>& features, int id): 
  word(word), 
  context(context),
  features(features),
  sentence_id(id) {}

DataPoint::DataPoint(int word, int tag, const Context& context, const std::vector<int>& features, int id): 
  word(word), 
  tag(tag),
  context(context),
  features(features),
  sentence_id(id) {}

} // namespace oxlm
