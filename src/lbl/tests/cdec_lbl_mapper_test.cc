#include "gtest/gtest.h"

#include "lbl/cdec_lbl_mapper.h"

#include "hg.h"

namespace oxlm {

TEST(CdecLBLMapperTest, TestBasic) {
  boost::shared_ptr<Dict> dict = boost::make_shared<Dict>();
  dict->convert("<s>", false);
  dict->convert("</s>", false);
  dict->convert("foo", false);
  dict->convert("bar", false);
  CdecLBLMapper mapper(dict);

  EXPECT_EQ(0, mapper.convert(1));
  EXPECT_EQ(1, mapper.convert(2));
  EXPECT_EQ(2, mapper.convert(3));
  EXPECT_EQ(3, mapper.convert(4));
}

} // namespace oxlm
