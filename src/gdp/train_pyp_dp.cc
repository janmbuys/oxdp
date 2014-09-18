#include "corpus/utils.h"
#include "corpus/dict.h"

#include "pyp/model_config.h"
#include "gdp/pyp_dp_model.h"

using namespace oxlm;

int main(int argc, char** argv) {
  //TODO make configuration readable from command file
  std::string training_file = "english-wsj/english_wsj_train.conll";
  std::string test_file = "english-wsj/english_wsj_dev.conll";
  
  boost::shared_ptr<ModelConfig> config = boost::make_shared<ModelConfig>();

  config->training_file = training_file;
  config->test_file = test_file;

  config->parser_type = ParserType::arcstandard; 
  //lexalization also influences context functions and sizes...
  config->lexicalised = false;

  config->randomise = true;
  config->iterations = 5;
  config->minibatch_size = 1;

  config->beam_sizes = {1, 2, 4}; //, 8, 16, 32, 64};

  PypDpModel model(config); 
  model.learn();
  //model.evaluate();

  return 0;
}

