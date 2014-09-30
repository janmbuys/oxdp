#include "gdp/pyp_dp_model.h"

namespace oxlm {

PypDpModel::PypDpModel() {
  dict_ = boost::make_shared<Dict>(true, false);
} 

PypDpModel::PypDpModel(const boost::shared_ptr<ModelConfig>& config): 
    config_(config) {
  dict_ = boost::make_shared<Dict>(true, config->parser_type==ParserType::arceager);
  if (config->parser_type == ParserType::arcstandard) {
    parse_model_ = boost::make_shared<ArcStandardParseModel>();
    config_->num_actions = 3;
  } else if (config->parser_type == ParserType::arceager) {
    parse_model_ = boost::make_shared<ArcEagerParseModel>();
    config_->num_actions = 4;
  } else {
    parse_model_ = boost::make_shared<EisnerParseModel>();
    config_->num_actions = 1;
    dict_->convert("STOP", false);
  }
}

//TODO learn_semi_supervised()

void PypDpModel::learn() {
  MT19937 eng;
  //read training data
  std::cerr << "Reading training corpus...\n";
  boost::shared_ptr<ParsedCorpus> training_corpus = boost::make_shared<ParsedCorpus>();
  training_corpus->readFile(config_->training_file, dict_, false);
  config_->vocab_size = dict_->size();
  config_->num_tags = dict_->tag_size();

  std::cerr << "Corpus size: " << training_corpus->size() << " sentences\t (" 
            << dict_->size() << " word types, " << dict_->tag_size() << " tags)\n";  


  //don't worry about working with an existing model now

  //read test data 
  std::cerr << "Reading test corpus...\n";
  boost::shared_ptr<ParsedCorpus> test_corpus = boost::make_shared<ParsedCorpus>();
  if (config_->test_file.size()) {
    test_corpus->readFile(config_->test_file, dict_, true);
    std::cerr << "Done reading test corpus..." << std::endl;
  }

  //instantiate weights

  if (config_->lexicalised) {
    if (config_->parser_type == ParserType::arcstandard)
      weights_ = boost::make_shared<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>(
            dict_->size(), dict_->tag_size(), config_->num_actions);
    else if (config_->parser_type == ParserType::arceager)
      weights_ = boost::make_shared<ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>(
            dict_->size(), dict_->tag_size(), config_->num_actions);
    else
      weights_ = boost::make_shared<ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>(
            dict_->size(), dict_->tag_size(), config_->num_actions);

  } else {
    if (config_->parser_type == ParserType::arcstandard)
      weights_ = boost::make_shared<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>(dict_->tag_size(), 
            config_->num_actions);
    else if (config_->parser_type == ParserType::arceager)
      weights_ = boost::make_shared<ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>(dict_->tag_size(), 
            config_->num_actions);
    else
      weights_ = boost::make_shared<ParsedPypWeights<tagLMOrderE, 1>>(dict_->tag_size(), 
            config_->num_actions);
  }

  std::vector<int> indices(training_corpus->size());
  std::iota(indices.begin(), indices.end(), 0);

  Real best_perplexity = std::numeric_limits<Real>::infinity();
  Real test_objective = 0; //use later
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
    std::cerr << "Training iteration " << iter << std::endl;
    auto iteration_start = get_time(); 
    int non_projective_count = 0;

    //std::cout << indices.size() << " indices\n";
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
      //std::cout << minibatch[0] << std::endl;
      for (auto j: minibatch) {
        //std::cout << j << " Gold arcs: " << std::endl;
        //training_corpus->sentence_at(j).print_arcs();
        if (iter > 0) {
          old_minibatch_examples->extend(examples_list.at(j));
        }            
        examples_list.at(j)->clear();
        parse_model_->extractSentence(training_corpus->sentence_at(j), examples_list.at(j));
                                                                     //, weights_);
        if (!training_corpus->sentence_at(j).is_projective_dependency())
          ++non_projective_count;            
        minibatch_examples->extend(examples_list.at(j));
        //std::cout << "done " << examples_list.at(j)->size() << std::endl;
        //parse_model_->extractSentence(training_corpus->sentence_at(j), minibatch_examples);
      }     

      weights_->updateRemove(*old_minibatch_examples, eng); 
      weights_->updateInsert(*minibatch_examples, eng); 
      //std::cout << "updated weights" << std::endl;

      //for now, only evaluate at end of iteration
      /* if ((minibatch_counter % 1000 == 0 && minibatch_counter <= 10000) || 
           minibatch_counter % 10000 == 0) {
        evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);
      } */

      ++minibatch_counter;
      start = end;
    }     
    //std::cout << std::endl;

    Real iteration_time = get_duration(iteration_start, get_time());

    std::cerr << "Iteration: " << iter << ", "
             << "Training Time: " << iteration_time << " seconds, "
             << "Non-projective: " << (non_projective_count + 0.0) / training_corpus->size() << ", "
             << "Training Objective: " << weights_->likelihood() / training_corpus->numTokens() 
           << "\n\n";
    
    evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);
  }

  std::cerr << "Overall minimum perplexity: " << best_perplexity << std::endl;
  
}

void PypDpModel::evaluate() const {
 //read test data 
  boost::shared_ptr<ParsedCorpus> test_corpus = boost::make_shared<ParsedCorpus>();
  test_corpus->readFile(config_->test_file, dict_, true);
  std::cerr << "Done reading test corpus..." << std::endl;
  
  Real log_likelihood = 0;
  evaluate(test_corpus, log_likelihood);
    
  size_t test_size = test_corpus->numTokens(); //TODO should actually be number of examples
  Real test_perplexity = perplexity(log_likelihood, test_size); //TODO use perplexity function
  std::cerr << "Test Perplexity: " << test_perplexity << std::endl;
}

void PypDpModel::evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, int minibatch_counter, 
                   Real& log_likelihood, Real& best_perplexity) const {
  if (test_corpus != nullptr) {
    evaluate(test_corpus, log_likelihood);
    
    size_t test_size = test_corpus->numTokens(); //TODO should actually be number of examples
    Real test_perplexity = perplexity(log_likelihood, test_size); //TODO use perplexity function
    std::cerr << "\tMinibatch " << minibatch_counter << ", "
         << "Test Perplexity: " << test_perplexity << std::endl;

    if (test_perplexity < best_perplexity) 
      best_perplexity = test_perplexity;
  }
}

void PypDpModel::evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, Real& accumulator) 
    const {
  if (test_corpus != nullptr) {
    accumulator = 0;
    
    std::vector<int> indices(test_corpus->size());
    std::iota(indices.begin(), indices.end(), 0);
   
    for (unsigned beam_size: config_->beam_sizes) {
      std::cerr << "parsing with beam size " << beam_size << ":\n";
      auto beam_start = get_time();
      boost::shared_ptr<AccuracyCounts> acc_counts = boost::make_shared<AccuracyCounts>();

      size_t start = 0;
      while (start < test_corpus->size()) {
        size_t end = std::min(start + config_->minibatch_size, test_corpus->size());

        //std::vector<int> minibatch = scatterMinibatch(start, end, indices);
        std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
        Real objective = 0;
            
        //TODO parallize, maybe move
        for (auto j: minibatch) {
          objective += parse_model_->evaluateSentence(test_corpus->sentence_at(j), weights_, acc_counts, 
                                                     beam_size);
          //TODO calculate gold likelihood
        }

        accumulator += objective;
        start = end;
      } 

      Real beam_time = get_duration(beam_start, get_time());
      Real sents_per_sec = static_cast<int>(test_corpus->size())/beam_time;
      std::cerr << "(" << static_cast<int>(sents_per_sec) << " sentences per second)\n"; 
      acc_counts->printAccuracy();
    }
  }
}

}

