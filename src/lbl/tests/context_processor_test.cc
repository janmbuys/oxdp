#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"

namespace oxlm {

TEST(ContextProcessorTest, TestExtract) {
  vector<vector<int>> data = {{0, 2, 3, 4, 1, 5, 6}};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  ContextProcessor processor(corpus, 3, 0, 1);

  vector<WordId> expected_context = {0, 0, 0};
  EXPECT_EQ(expected_context, processor.extract(0, 1));
  expected_context = {2, 0, 0};
  EXPECT_EQ(expected_context, processor.extract(0, 2));
  expected_context = {3, 2, 0};
  EXPECT_EQ(expected_context, processor.extract(0, 3));
  expected_context = {4, 3, 2};
  EXPECT_EQ(expected_context, processor.extract(0, 4));
  expected_context = {0, 0, 0};
  EXPECT_EQ(expected_context, processor.extract(0, 5));
  expected_context = {5, 0, 0};
  EXPECT_EQ(expected_context, processor.extract(0, 6));
}

} // namespace oxlm
