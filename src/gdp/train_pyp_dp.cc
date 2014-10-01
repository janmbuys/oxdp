#include "corpus/utils.h"
#include "corpus/dict.h"

#include "pyp/model_config.h"
#include "pyp/pyp_model.h"

#include "gdp/pyp_dp_model.h"

using namespace oxlm;

int main(int argc, char** argv) {
  //TODO make configuration readable from command file
  std::string training_file = "english-wsj-stanford-unk/english_wsj_train.conll";
  //std::string training_file = "question-bank-unk/qbank_train.conll";
  //std::string training_file = "question-bank-unk/wsj_qbank_train.conll";
  
  std::string training_file_unsup = "question-bank-unk/qbank_train.conll";
  //std::string training_file_unsup = "question-bank-unk/wikianswers_questions_100k.conll";
  
  //std::string test_file = "english-wsj-stanford-unk/english_wsj_dev.conll";
  std::string test_file = "question-bank-unk/qbank_dev.conll";

  boost::shared_ptr<ModelConfig> config = boost::make_shared<ModelConfig>();

  config->training_file = training_file;
  config->training_file_unsup = training_file_unsup;
  config->test_file = test_file;

  //config->parser_type = ParserType::ngram; 
  config->parser_type = ParserType::arcstandard; 
  //lexalization also influences context functions and sizes...
  config->lexicalised = true;
  config->semi_supervised = true;

  config->randomise = true;
  config->iterations = 10;
  config->minibatch_size = 1;

  config->beam_sizes = {1, 2, 4, 8}; //1, 2, 4, 8, 16, 32, 64};
  //config->beam_sizes = {1}; //, 8, 16, 32, 64};

  if (config->parser_type == ParserType::ngram) {
    PypModel model(config); 
    model.learn();
  } else {
    PypDpModel model(config); 
    if (config->semi_supervised)
      model.learn_semi_supervised();
    else
      model.learn();
    //model.evaluate();
  }

  return 0;
}

