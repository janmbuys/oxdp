#include "gdp/pyp_model.h"

namespace oxlm {

PypModel::PypModel() {
  dict_ = boost::make_shared<Dict>();
} 

PypModel::PypModel(const boost::shared_ptr<ModelConfig>& config): 
    config_(config) {
  dict_ = boost::make_shared<Dict>();
  model_ = boost::make_shared<NGramModel<PypWeights<wordLMOrder>>>(wordLMOrder, dict_->sos(), dict_->eos());
}

void PypModel::learn() {
  MT19937 eng;
  //read training data
  std::cerr << "Reading training corpus...\n";
  boost::shared_ptr<SentenceCorpus> training_corpus = boost::make_shared<SentenceCorpus>();
  training_corpus->readFile(config_->training_file, dict_, false);
  config_->vocab_size = dict_->size();

  std::cerr << "Corpus size: " << training_corpus->size() << " sentences\t (" 
            << dict_->size() << " word types)\n";  


  //don't worry about working with an existing model now

  //read test data 
  std::cerr << "Reading test corpus...\n";
  boost::shared_ptr<SentenceCorpus> test_corpus = boost::make_shared<SentenceCorpus>();
  if (config_->test_file.size()) {
    test_corpus->readFile(config_->test_file, dict_, true);
    std::cerr << "Done reading test corpus..." << std::endl;
  }

  //instantiate weights
  weights_ = boost::make_shared<PypWeights<wordLMOrder>>(dict_->size());

  //Here indices are still over sentences
  std::vector<int> indices(training_corpus->size());
  std::iota(indices.begin(), indices.end(), 0);

  Real best_perplexity = std::numeric_limits<Real>::infinity();
  Real test_objective = 0; 
  int minibatch_counter = 1;
  int minibatch_size = config_->minibatch_size;

  //TODO parallelize
      
  for (int iter = 0; iter < config_->iterations; ++iter) {
    std::cerr << "Training iteration " << iter << std::endl;
    auto iteration_start = get_time(); 

    if (config_->randomise)
      std::random_shuffle(indices.begin(), indices.end());
   
    //first implement version where we don't store examples from previous iteration

    size_t start = 0;
    //loop over minibatches
    while (start < training_corpus->size()) {
      size_t end = std::min(training_corpus->size(), start + minibatch_size);
      
      std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
      boost::shared_ptr<DataSet> minibatch_examples = boost::make_shared<DataSet>();

      //index boundaries for splitting among threads
      //std::vector<int> minibatch = scatterMinibatch(start, end, indices);
       
      //firstly extract all the training examples
      for (auto j: minibatch) 
        model_->extractSentence(training_corpus->sentence_at(j), minibatch_examples);

      //update weights (can ||s)
      if (iter > 0) {
        weights_->updateRemove(minibatch_examples, eng); 
      }            

      weights_->updateInsert(minibatch_examples, eng); 

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
             << "Training Objective: " << weights_->likelihood() / training_corpus->numTokens() 
           << "\n\n";
    
    evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);
  }

  std::cerr << "Overall minimum perplexity: " << best_perplexity << std::endl;
   
  //generate from model
  for (int i = 0; i < 100; ++i) {
    Sentence sent = model_->generateSentence(weights_, eng);
    sent.print_sentence(dict_);
  }
}

void PypModel::evaluate() const {
 //read test data 
  boost::shared_ptr<SentenceCorpus> test_corpus = boost::make_shared<SentenceCorpus>();
  test_corpus->readFile(config_->test_file, dict_, true);
  std::cerr << "Done reading test corpus..." << std::endl;
  
  Real log_likelihood = 0;
  evaluate(test_corpus, log_likelihood);
    
  Real test_perplexity = perplexity(log_likelihood, test_corpus->numTokens());
     
  std::cerr << "Test Perplexity: " << test_perplexity << std::endl;
}

void PypModel::evaluate(const boost::shared_ptr<SentenceCorpus>& test_corpus, int minibatch_counter, 
                   Real& log_likelihood, Real& best_perplexity) const {
  if (test_corpus != nullptr) {
    evaluate(test_corpus, log_likelihood);
    
    Real test_perplexity = perplexity(log_likelihood, test_corpus->numTokens());
    std::cerr << "\tMinibatch " << minibatch_counter << ", "
           << "Test Likelihood: " << log_likelihood << std::endl
           << "Test Size: " << test_corpus->numTokens() << std::endl
         << "Test Perplexity: " << test_perplexity << std::endl;

    if (test_perplexity < best_perplexity) 
      best_perplexity = test_perplexity;
  }
}

void PypModel::evaluate(const boost::shared_ptr<SentenceCorpus>& test_corpus, Real& accumulator) 
    const {
  if (test_corpus != nullptr) {
    accumulator = 0;
    
    std::vector<int> indices(test_corpus->size());
    std::iota(indices.begin(), indices.end(), 0);
   
    std::cerr << "Evaluating:\n";
    auto eval_start = get_time();
    size_t start = 0;

    while (start < test_corpus->size()) {
      size_t end = std::min(start + config_->minibatch_size, test_corpus->size());

      //std::vector<int> minibatch = scatterMinibatch(start, end, indices);
      std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
      Real objective = 0;
            
      //TODO parallize, maybe move
      for (auto j: minibatch) 
        objective += model_->evaluateSentence(test_corpus->sentence_at(j), weights_);

      accumulator += objective;
      start = end;
    } 

    Real eval_time = get_duration(eval_start, get_time());
    std::cerr << "Time: " << eval_time << " seconds\n"; 
  }
}

}

