#include "gdp/pyp_model.h"

namespace oxlm {

PypModel::PypModel() num_actions_{1} {
  dict_ = boost::make_shared<Dict>(true, false);
} 

PypModel::PypModel(const boost::shared_ptr<ModelConfig>& config): 
    config_(config),
    num_actions_{1} {
  dict_ = boost::make_shared<Dict>(true, config->model_type==ModelType::arceager);
  if (config->model_type == ModelType::arcstandard)
    num_actions_ = 3;
  else if (config->model_type == ModelType::arceager)
    num_actions_ = 4;
}

void PypModel::learn() {
  //read training data
  std::cerr << "Reading training corpus...\n";
  boost::shared_ptr<Corpus> training_corpus = boost::make_shared<Corpus>();
  training_corpus.readFile(config_->training_file, dict_, false);

  std::cerr << "Corpus size: " << training_corpus->size() << " sentences\t (" 
            << dict->size() << " word types, " << dict->tag_size() << " tags)\n";

  size_t num_actions = 1;

  //don't worry about working with an existing model now

  //read test data 
  boost::shared_ptr<Corpus> test_corpus = boost::make_shared<Corpus>();
  if (config->test_file.size()) {
    Corpus.readFile(config_->training_file, dict_, false);
    test_corpus.readFile(config->test_file, dict_, true);
    cout << "Done reading test corpus..." << endl;
  }

  //instantiate weights
  if (config->lexicalised)
    weights_ = boost::make_shared<ParsedLexPypWeights>(dict->size(), dict->tag_size(), num_actions);
  else
    weights_ = boost::make_shared<ParsedPypWeights>(dict->tag_size(), num_actions);
 
  std::vector<int> indices(training_corpus->size());
  //double test_objective = 0; //use later

  //TODO implement per-minibach, randomized order training, and context extractor

  /*switch (config->model_type) {
    case ModelType::eisner:
     = boost::make_shared<EisnerParseModel>( )
  } */


}

void PypModel::evaluate() const {
 //read test data 
  boost::shared_ptr<Corpus> test_corpus = boost::make_shared<Corpus>();
  Corpus.readFile(config_->training_file, dict_, false);
  test_corpus.readFile(config->test_file, dict_, true);
  cout << "Done reading test corpus..." << endl;
  
  evaluate(test_corpus);
}

void PypModel::evaluate(const boost::shared_ptr<Corpus>& corpus) const {


}


}

