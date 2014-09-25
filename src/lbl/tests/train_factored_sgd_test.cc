#include "gtest/gtest.h"

#include "lbl/factored_weights.h"
#include "lbl/model.h"
#include "lbl/model_utils.h"
#include "lbl/tests/sgd_test.h"
#include "utils/constants.h"

namespace oxlm {

TEST_F(FactoredSGDTest, TestTrainFactoredSGD) {
  Model<FactoredWeights, FactoredWeights, FactoredMetadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  boost::shared_ptr<Dict> dict = model.getDict();
  boost::shared_ptr<SentenceCorpus> test_corpus = boost::make_shared<SentenceCorpus>();
  test_corpus->readFile(config->test_file, dict, true);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(61.6428337,perplexity(log_likelihood, test_corpus->numTokens()), EPS); // 61.6428031 
}

TEST_F(FactoredSGDTest, TestTrainFactoredNCE) {
  config->noise_samples = 10;
  Model<FactoredWeights, FactoredWeights, FactoredMetadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  boost::shared_ptr<Dict> dict = model.getDict();
  boost::shared_ptr<SentenceCorpus> test_corpus = boost::make_shared<SentenceCorpus>();
  test_corpus->readFile(config->test_file, dict, true);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  EXPECT_NEAR(66.0728988, perplexity(log_likelihood, test_corpus->numTokens()), EPS); //66.0725250
}

} // namespace oxlm
