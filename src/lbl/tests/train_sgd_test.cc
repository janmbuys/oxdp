#include "gtest/gtest.h"

#include "lbl/model.h"
#include "lbl/model_utils.h"
#include "lbl/weights.h"
#include "lbl/tests/sgd_test.h"
#include "utils/constants.h"

namespace oxlm {

TEST_F(SGDTest, TestBasic) {
  Model<Weights, Weights, Metadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  boost::shared_ptr<Dict> dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = boost::make_shared<Corpus>();
  test_corpus->readFile(config->test_file, dict, true);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  std::cout << "  Test Likelihood: " << log_likelihood 
           << "  Test Size: " << test_corpus->numTokens() 
           << "  Test Perplexity: " << perplexity(log_likelihood, test_corpus->numTokens()) << std::endl;
  EXPECT_NEAR(72.2445220, perplexity(log_likelihood, test_corpus->size()), EPS);
}

TEST_F(SGDTest, TestNCE) {
  config->noise_samples = 10;
  Model<Weights, Weights, Metadata> model(config);
  model.learn();
  config->test_file = "test.txt";
  boost::shared_ptr<Dict> dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = boost::make_shared<Corpus>();
  test_corpus->readFile(config->test_file, dict, true);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  std::cout << "  Test Likelihood: " << log_likelihood 
           << "  Test Size: " << test_corpus->numTokens() 
           << "  Test Perplexity: " << perplexity(log_likelihood, test_corpus->numTokens()) << std::endl;
  EXPECT_NEAR(66.8699874, perplexity(log_likelihood, test_corpus->size()), EPS); //67.7361526
}

} // namespace oxlm
