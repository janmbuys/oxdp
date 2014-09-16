#include "corpus/utils.h"
#include "corpus/dict.h"
#include "gdp/model_config.h"
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
  config->lexicalised = false;

  config->randomise = true;
  config->iterations = 1;
  config->minibatch_size = 1;
  
  std::cerr << "start" << std::endl;

  PypDpModel model(config); 
  model.learn();
  //model.evaluate();

  return 0;
}

