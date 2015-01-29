#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>

#include "corpus/sentence_corpus.h"
#include "lbl/context_processor.h"
#include "lbl/weights.h"

#include "utils/constants.h"
#include "utils/testing.h"

#include "gdp/ngram_model.h"

namespace ar = boost::archive;

namespace oxlm {

class TestWeights : public testing::Test {
 protected:
  void SetUp() {
    config = boost::make_shared<ModelConfig>();
    config->word_representation_size = 3;
    config->vocab_size = 5;
    config->ngram_order = 3;
    config->activation = Activation::sigmoid;

    vector<int> data = {2, 3, 4, 1};
    corpus = boost::make_shared<SentenceCorpus>(data, config->vocab_size);
    boost::shared_ptr<Dict> dict = boost::make_shared<Dict>();
    metadata = boost::make_shared<Metadata>(config, dict);
    metadata->initialize(corpus);
    ngram_model = boost::make_shared<NGramModel<Weights>>(config->ngram_order, dict->sos(), dict->eos());
  }

  Real getPredictions(
      const boost::shared_ptr<Weights>& weights, const vector<int>& indices) const {
    Real ret = ngram_model->evaluateSentence(corpus->sentence_at(0), weights);
    return ret;
  }

  boost::shared_ptr<ModelConfig> config;
  boost::shared_ptr<Metadata> metadata;
  boost::shared_ptr<SentenceCorpus> corpus;
  boost::shared_ptr<NGramModel<Weights>> ngram_model;
};

TEST_F(TestWeights, TestGradientCheck) {
  Weights weights(config, metadata, true);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;

  boost::shared_ptr<Weights> gradient =
      boost::make_shared<Weights>(config, metadata, false);
  boost::shared_ptr<DataSet> examples = boost::make_shared<DataSet>();
  ngram_model->extractSentence(corpus->sentence_at(0), examples);
  
  weights.getGradient(examples, gradient, objective, words);

  // In truth, using float for model parameters instead of double seriously
  // degrades the gradient computation, but has no negative effect on the
  // performance of the model and gives a 2x speed up and reduces memory by 2x.
  //
  // If you suspect there might be something off with the gradient, change
  // typedef Real to double and set a lower accepted error (e.g. 1e-5) when
  // checking the gradient.
  EXPECT_TRUE(weights.checkGradient(examples, gradient, 1e-3));
}

TEST_F(TestWeights, TestGradientCheckDiagonal) {
  config->diagonal_contexts = true;

  Weights weights(config, metadata, true);
  vector<int> indices = {0, 1, 2, 3};
  Real objective;
  MinibatchWords words;
  boost::shared_ptr<Weights> gradient =
      boost::make_shared<Weights>(config, metadata, false);
  boost::shared_ptr<DataSet> examples = boost::make_shared<DataSet>();
  ngram_model->extractSentence(corpus->sentence_at(0), examples);
  weights.getGradient(examples, gradient, objective, words);

  // See the comment above if you suspect the gradient is not computed
  // correctly.
  EXPECT_TRUE(weights.checkGradient(examples, gradient, 1e-3));
}

TEST_F(TestWeights, TestPredict) {
  boost::shared_ptr<Weights> weights = boost::make_shared<Weights>(config, metadata, true);
  vector<int> indices = {0, 1, 2, 3};
  boost::shared_ptr<DataSet> examples = boost::make_shared<DataSet>();
  ngram_model->extractSentence(corpus->sentence_at(0), examples);
  Real objective = weights->getObjective(examples);

  EXPECT_NEAR(objective, getPredictions(weights, indices), EPS);
  // Check cache values.
  EXPECT_NEAR(objective, getPredictions(weights, indices), EPS);
}

TEST_F(TestWeights, TestSerialization) {
  Weights weights(config, metadata, true), weights_copy;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << weights;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> weights_copy;

  EXPECT_EQ(weights, weights_copy);
}

} // namespace oxlm
