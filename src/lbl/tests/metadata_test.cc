#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>

#include "corpus/sentence_corpus.h"
#include "lbl/metadata.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(MetadataTest, TestUnigram) {
  boost::shared_ptr<ModelConfig> config = boost::make_shared<ModelConfig>();
  config->vocab_size = 5;
  boost::shared_ptr<Dict> dict = boost::make_shared<Dict>();
  Metadata metadata(config, dict);

  vector<int> data = {2, 3, 2, 4, 1};
  boost::shared_ptr<SentenceCorpus> corpus = boost::make_shared<SentenceCorpus>(data, config->vocab_size);
  metadata.initialize(corpus);
  VectorReal expected_unigram = VectorReal::Zero(5);
  expected_unigram << 0, 0.2, 0.4, 0.2, 0.2;
  EXPECT_MATRIX_NEAR(expected_unigram, metadata.getUnigram(), EPS);
}

TEST(MetadataTest, TestSerialization) {
  boost::shared_ptr<ModelConfig> config = boost::make_shared<ModelConfig>();
  boost::shared_ptr<Dict> dict = boost::make_shared<Dict>();
  Metadata metadata(config, dict), metadata_copy;

  stringstream stream(ios_base::binary | ios_base::in | ios_base::out);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << metadata;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> metadata_copy;

  EXPECT_EQ(metadata, metadata_copy);
}

} // namespace oxlm
