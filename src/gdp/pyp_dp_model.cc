#include "gdp/pyp_dp_model.h"

namespace oxlm {

PypDpModel::PypDpModel(): 
    num_actions_{1} {
  dict_ = boost::make_shared<Dict>(true, false);
} 

PypDpModel::PypDpModel(const boost::shared_ptr<ModelConfig>& config): 
    config_(config),
    num_actions_{1} {
  dict_ = boost::make_shared<Dict>(true, config->parser_type==ParserType::arceager);
  if (config->parser_type == ParserType::arcstandard)
    num_actions_ = 3;
  else if (config->parser_type == ParserType::arceager)
    num_actions_ = 4;
}

void PypDpModel::learn() {
  //read training data
  std::cerr << "Reading training corpus...\n";
  boost::shared_ptr<ParsedCorpus> training_corpus = boost::make_shared<ParsedCorpus>();
  training_corpus->readFile(config_->training_file, dict_, false);

  std::cerr << "Corpus size: " << training_corpus->size() << " sentences\t (" 
            << dict_->size() << " word types, " << dict_->tag_size() << " tags)\n";

  //don't worry about working with an existing model now

  //read test data 
  boost::shared_ptr<ParsedCorpus> test_corpus = boost::make_shared<ParsedCorpus>();
  if (config_->test_file.size()) {
    test_corpus->readFile(config_->test_file, dict_, true);
    std::cout << "Done reading test corpus..." << std::endl;
  }

  //instantiate weights

  if (config_->lexicalised)
    weights_ = boost::make_shared<ParsedLexPypWeights<wordLMOrder, tagLMOrder, actionLMOrder>>(
            dict_->size(), dict_->tag_size(), num_actions_);
  else
    weights_ = boost::make_shared<ParsedPypWeights<tagLMOrder, actionLMOrder>>(
            dict_->tag_size(), num_actions_);
 
  std::vector<int> indices(training_corpus->size());
  std::iota(indices.begin(), indices.end(), 0);

  double best_perplexity = std::numeric_limits<double>::infinity();
  double test_objective = 0; //use later
  int minibatch_counter = 1;
  int minibatch_size = config_->minibatch_size;

  //TODO parallelize

  //need to record per sentence ...
  //TODO need a proper way to keep track of indices
  // one way is to randomize only once, not for each iteration
  // else keep sentence index while building examples for a minibatch
  //std::vector<DataSet> word_examples_list(training_corpus->size(), DataSet());
  //std::vector<DataSet> tag_examples_list(training_corpus->size(), DataSet());
  //std::vector<DataSet> action_examples_list(trainings_corpus->size(), DataSet());

  for (int iter = 0; iter < config_->iterations; ++iter) {
    //auto iteretion_start = GetTime(); //possibly clashing definitions
    if (config_->randomise)
      std::random_shuffle(indices.begin(), indices.end());
   
    size_t start = 0;
    //loop over minibatches
    while (start < training_corpus->size()) {
      size_t end = std::min(training_corpus->size(), start + minibatch_size);

      
      std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
      
      //index boundaries for splitting among threads
      //std::vector<int> minibatch = scatterMinibatch(start, end, indices);
      //TODO implement training
      //for now don't implement seperate context extractor 


      if ((minibatch_counter % 1000 == 0 && minibatch_counter <= 10000) || 
           minibatch_counter % 10000 == 0) {
        evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);
      }

      ++minibatch_counter;
      start = end;
    }     

    evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);

    //double iteration_time = GetDuration(iteration_start, GetTime());
   //TODO update
    std::cout << "Iteration: " << iter << ", "
         //    << "Time: " << iteration_time << " seconds, "
             << "Training Objective: " << weights_->likelihood() / training_corpus->numTokens() 
           << "\n\n";
  }

  std::cout << "Overall minimum perplexity: " << best_perplexity << std::endl;

  /*switch (config_->parserl_type) {
    case ParserType::eisner:
     = boost::make_shared<EisnerParseModel>( )
  } */


}

void PypDpModel::evaluate() const {
 //read test data 
  boost::shared_ptr<ParsedCorpus> test_corpus = boost::make_shared<ParsedCorpus>();
  test_corpus->readFile(config_->test_file, dict_, true);
  std::cout << "Done reading test corpus..." << std::endl;
  
  double log_likelihood = 0;
  evaluate(test_corpus, log_likelihood);
    
  size_t test_size = test_corpus->numTokens(); //TODO should actually be number of examples
  double test_perplexity = std::exp(log_likelihood/test_size); //TODO use perplexity function
  std::cout << "Test Perplexity: " << test_perplexity << std::endl;
}

void PypDpModel::evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, int minibatch_counter, 
                   double& log_likelihood, double& best_perplexity) const {
  if (test_corpus != nullptr) {
    evaluate(test_corpus, log_likelihood);
    
    size_t test_size = test_corpus->numTokens(); //TODO should actually be number of examples
    double test_perplexity = std::exp(log_likelihood/test_size); //TODO use perplexity function
    std::cout << "\tMinibatch " << minibatch_counter << ", "
         << "Test Perplexity: " << test_perplexity << std::endl;

    if (test_perplexity < best_perplexity) 
      best_perplexity = test_perplexity;
  }
}

void PypDpModel::evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, double& accumulator) 
    const {
  if (test_corpus != nullptr) {
    accumulator = 0;
    
    std::vector<int> indices(test_corpus->size());
    std::iota(indices.begin(), indices.end(), 0);
   
    //for now this is hard-coded 
    std::vector<unsigned> beam_sizes{1, 2, 4, 8, 16, 32, 64};

    for (unsigned beam_size: beam_sizes) {
      AccuracyCounts acc_counts;

      size_t start = 0;
      while (start < test_corpus->size()) {
        size_t end = std::min(start + config_->minibatch_size, test_corpus->size());

        //std::vector<int> minibatch = scatterMinibatch(start, end, indices);
        std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
        double objective = 0;
        boost::shared_ptr<TransitionParseModelInterface> parse_model;
        //TransitionParseModelInterface parse_model;
        if (config_->parser_type==ParserType::arcstandard)
          parse_model = boost::make_shared<ArcStandardParseModel>();
          //parse_model = ArcStandardParseModel();
        else if (config_->parser_type==ParserType::arceager)
          parse_model = boost::make_shared<ArcEagerParseModel>();
          //parse_model = ArcEagerParseModel();
        //TODO parallize, maybe move
        for (size_t j = start; j < end; ++j) {
          TransitionParser parse = parse_model->beamParseSentence(test_corpus->sentence_at(j), weights_, beam_size);

          if (config_->parser_type==ParserType::arcstandard)
            acc_counts.countArcStandardAccuracy(parse, test_corpus->sentence_at(j));
          else if (config_->parser_type==ParserType::arceager)
            acc_counts.countArcEagerAccuracy(parse, test_corpus->sentence_at(j));
          objective += parse.particle_weight();
          //TODO calculate gold likelihood
        }

        accumulator += objective;
        start = end;
      } 

      acc_counts.printAccuracy();
    }
  }
}


}

