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
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = boost::make_shared<Corpus>();
  dict.read_from_file(config->test_file, test_corpus, true);
  Real log_likelihood = 0;
  model.evaluate(test_corpus, log_likelihood);
  std::cout << log_likelihood << std::endl;
  size_t test_size = 0;
  for (unsigned i = 0; i < test_corpus->size(); ++i)
    test_size += test_corpus->at(i).size() - 1;
  //std::cout << log_likelihood << " " << test_size << std::endl;

  //EXPECT_NEAR(61.6428031, perplexity(log_likelihood, test_size), EPS);
  //for sentence-level evaluation
  EXPECT_NEAR(68.46865, perplexity(log_likelihood, test_size), EPS);
}

} // namespace oxlm
