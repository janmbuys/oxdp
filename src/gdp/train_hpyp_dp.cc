#include "corpus/dict.h"
#include "corpus/parsed_sentence.h"
#include "gdp/arc_standard_parse_model.h"
#include "gdp/arc_eager_parse_model.h"
#include "gdp/eisner_parse_model.h"
#include "gdp/model_config.h"

using namespace oxlm;

int main(int argc, char** argv) {
  //TODO make configuration readable from command file
  std::string training_file = "english-wsj/english_wsj_train.conll";
  std::string test_file = "english-wsj/english_wsj_dev.conll";
  
  boost::shared_ptr<ModelConfig> config = boost::make_shared<ModelConfig> (training_file, 
          test_file, 1, 1, false, ParserType::arcstandard, false);
  
  PypModel model(config); 
  
  model.learn();
 
  model.evaluate();

  return 0;
}

