#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>

#include "corpus/ngram_model.h"
#include "corpus/sentence_corpus.h"
#include "lbl/context_processor.h"
#include "lbl/factored_weights.h"
#include "utils/constants.h"

namespace ar = boost::archive;

namespace oxlm {

class FactoredWeightsTest : public testing::Test {
 protected:
  void SetUp() {
    config = boost::make_shared<ModelData>();
    config->word_representation_size = 3;
    config->vocab_size = 5;
    config->ngram_order = 3;
    config->sigmoid = true;

    vector<int> data = {2, 3, 4, 1};
    vector<int> classes = {0, 2, 4, 5};
    corpus = boost::make_shared<SentenceCorpus>(data);
    index = boost::make_shared<WordToClassIndex>(classes);
    dict = boost::make_shared<Dict>();
    metadata = boost::make_shared<FactoredMetadata>(config, dict, index);
    metadata->initialize(corpus);
    ngram_model = boost::make_shared<NGramModel>(config->ngram_order, dict->sos(), dict->eos());
  }

  Real getPredictions(
      const boost::shared_ptr<FactoredWeights>& weights, const vector<int>& indices) const {
    Real ret = ngram_model->evaluateSentence(corpus->sentence_at(0), weights);
    return ret;
  }

  boost::shared_ptr<ModelData> config;
  boost::shared_ptr<Dict> dict;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<FactoredMetadata> metadata;
  boost::shared_ptr<SentenceCorpus> corpus;
  boost::shared_ptr<NGramModel> ngram_model;
};

TEST_F(FactoredWeightsTest, TestCheckGradient) {
  FactoredWeights weights(config, metadata, true);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;

  boost::shared_ptr<FactoredWeights> gradient =
      boost::make_shared<FactoredWeights>(config, metadata, false);
  boost::shared_ptr<DataSet> examples = boost::make_shared<DataSet>();
  ngram_model->extractSentence(corpus->sentence_at(0), examples);

  weights.getGradient(examples, gradient, objective, words);
  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(examples, gradient, 1e-3));
}

TEST_F(FactoredWeightsTest, TestCheckGradientDiagonal) {
  config->diagonal_contexts = true;
  FactoredWeights weights(config, metadata, true);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<FactoredWeights> gradient =
      boost::make_shared<FactoredWeights>(config, metadata, false);
  boost::shared_ptr<DataSet> examples = boost::make_shared<DataSet>();
  ngram_model->extractSentence(corpus->sentence_at(0), examples);
  weights.getGradient(examples, gradient, objective, words);

  // See the comment in weights_test.cc if you suspect the gradient is not
  // computed correctly.
  EXPECT_TRUE(weights.checkGradient(examples, gradient, 1e-3));
}

TEST_F(FactoredWeightsTest, TestPredict) {
  boost::shared_ptr<FactoredWeights> weights = boost::make_shared<FactoredWeights>(config, metadata, 
           true);
  vector<int> indices = {0, 1, 2, 3};
  boost::shared_ptr<DataSet> examples = boost::make_shared<DataSet>();
  ngram_model->extractSentence(corpus->sentence_at(0), examples);

  Real objective = weights->getObjective(examples);

  EXPECT_NEAR(objective, getPredictions(weights, indices), EPS);
  // Check cache values.
  EXPECT_NEAR(objective, getPredictions(weights, indices), EPS);
}

TEST_F(FactoredWeightsTest, TestSerialization) {
  FactoredWeights weights(config, metadata, true), weights_copy;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << weights;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> weights_copy;

  EXPECT_EQ(weights, weights_copy);
}

} // namespace oxlm
