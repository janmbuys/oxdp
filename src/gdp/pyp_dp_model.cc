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
  if (config->parser_type == ParserType::arcstandard) {
    parse_model_ = boost::make_shared<ArcStandardParseModel>();
    num_actions_ = 3;
  } else if (config->parser_type == ParserType::arceager) {
    parse_model_ = boost::make_shared<ArcEagerParseModel>();
    num_actions_ = 4;
  } else {
    parse_model_ = boost::make_shared<EisnerParseModel>();
    num_actions_ = 1;
  }
}

void PypDpModel::learn() {
  MT19937 eng;
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

  if (config_->lexicalised) {
    if (config_->parser_type == ParserType::arcstandard)
      weights_ = boost::make_shared<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>(
            dict_->size(), dict_->tag_size(), num_actions_);
    else if (config_->parser_type == ParserType::arceager)
      weights_ = boost::make_shared<ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>(
            dict_->size(), dict_->tag_size(), num_actions_);
    else
      weights_ = boost::make_shared<ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>(
            dict_->size(), dict_->tag_size(), num_actions_);

  } else {
    if (config_->parser_type == ParserType::arcstandard)
      weights_ = boost::make_shared<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>(dict_->tag_size(), 
            num_actions_);
    else if (config_->parser_type == ParserType::arceager)
      weights_ = boost::make_shared<ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>(dict_->tag_size(), 
            num_actions_);
    else
      weights_ = boost::make_shared<ParsedPypWeights<tagLMOrderE, 1>>(dict_->tag_size(), 
            num_actions_);
  }

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
  std::vector<boost::shared_ptr<ParseDataSet>> examples_list(training_corpus->size(), 
          boost::make_shared<ParseDataSet>());
       
  for (int iter = 0; iter < config_->iterations; ++iter) {
    //auto iteretion_start = GetTime(); //possibly clashing definitions
    if (config_->randomise)
      std::random_shuffle(indices.begin(), indices.end());
   
    //first implement version where we don't store examples from previous iteration

    size_t start = 0;
    //loop over minibatches
    while (start < training_corpus->size()) {
      size_t end = std::min(training_corpus->size(), start + minibatch_size);
      
      std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
      boost::shared_ptr<ParseDataSet> minibatch_examples = boost::make_shared<ParseDataSet>();
      boost::shared_ptr<ParseDataSet> old_minibatch_examples = boost::make_shared<ParseDataSet>();

      //index boundaries for splitting among threads
      //std::vector<int> minibatch = scatterMinibatch(start, end, indices);
       
      //critical loop
      for (auto j: minibatch) {
        if (iter > 0) {
          old_minibatch_examples->extend(examples_list.at(j));
        }            
        examples_list.at(j)->clear();
        parse_model_->extractSentence(training_corpus->sentence_at(j), examples_list.at(j));
                                                                     //, weights_);
        minibatch_examples->extend(examples_list.at(j));
        //parse_model_->extractSentence(training_corpus->sentence_at(j), minibatch_examples);
      }     

      weights_->updateRemove(*old_minibatch_examples, eng); 
      weights_->updateInsert(*minibatch_examples, eng); 

      //for now, only evaluate at end of iteration
      /* if ((minibatch_counter % 1000 == 0 && minibatch_counter <= 10000) || 
           minibatch_counter % 10000 == 0) {
        evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);
      } */

      ++minibatch_counter;
      start = end;
    }     
    //std::cout << std::endl;

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
      std::cerr << "parsing with beam size " << beam_size << ":\n";
      boost::shared_ptr<AccuracyCounts> acc_counts = boost::make_shared<AccuracyCounts>();

      size_t start = 0;
      while (start < test_corpus->size()) {
        size_t end = std::min(start + config_->minibatch_size, test_corpus->size());

        //std::vector<int> minibatch = scatterMinibatch(start, end, indices);
        std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
        double objective = 0;
            
        //TODO parallize, maybe move
        for (size_t j = start; j < end; ++j) {
          objective += parse_model_->evaluateSentence(test_corpus->sentence_at(j), weights_, acc_counts, 
                                                     beam_size);
          //TODO calculate gold likelihood
        }

        accumulator += objective;
        start = end;
      } 

      acc_counts->printAccuracy();
    }
  }
}

}

