#include "gdp/pyp_dp_model.h"

namespace oxlm {

template<class ParseModel, class ParsedWeights>
PypDpModel<ParseModel, ParsedWeights>::PypDpModel() {
  dict_ = boost::make_shared<Dict>(true, false);
} 

template<class ParseModel, class ParsedWeights>
PypDpModel<ParseModel, ParsedWeights>::PypDpModel(const boost::shared_ptr<ModelConfig>& config): 
    config_(config) {
  dict_ = boost::make_shared<Dict>(true, config_->parser_type==ParserType::arceager);
  
  if (config_->parser_type == ParserType::arcstandard) {
    config_->num_actions = 3;
  } else if (config_->parser_type == ParserType::arceager) {
    config_->num_actions = 4;
  } else {
    config_->num_actions = 1;
    dict_->convert("<stop>", false);
  }

  parse_model_ = boost::make_shared<ParseModel>(config);
}

template<class ParseModel, class ParsedWeights>
void PypDpModel<ParseModel, ParsedWeights>::learn_semi_supervised() {
  MT19937 eng;
  //read training data
  
  boost::shared_ptr<ParsedCorpus> sup_training_corpus = boost::make_shared<ParsedCorpus>();
  boost::shared_ptr<ParsedCorpus> unsup_training_corpus = boost::make_shared<ParsedCorpus>();
 
  if (config_->training_file.size()) { 
    std::cerr << "Reading supervised training corpus...\n";
    sup_training_corpus->readFile(config_->training_file, dict_, false);
     std::cerr << "Corpus size: " << sup_training_corpus->size() << " sentences\t (" 
              << dict_->size() << " word types, " << dict_->tag_size() << " tags, " 
	      << dict_->label_size() << " labels)\n";  
  } else {
    std::cerr << "No supervised training corpus.\n";
  } 

  if (config_->training_file_unsup.size()) { 
    std::cerr << "Reading unsupervised training corpus...\n";
    unsup_training_corpus->readFile(config_->training_file_unsup, dict_, false);
     std::cerr << "Corpus size: " << unsup_training_corpus->size() << " sentences\t (" 
              << dict_->size() << " word types, " << dict_->tag_size() << " tags, "  
	      << dict_->label_size() << " labels)\n";  
  } else {
    std::cerr << "No unsupervised training corpus.\n";
  }

  config_->vocab_size = dict_->size();
  config_->num_tags = dict_->tag_size();
  if (config_->labelled_parser) {
    config_->num_labels = dict_->label_size();
    config_->num_actions += 2*(dict_->label_size()-1); //add labelled actions
  }

  //read test data 
  std::cerr << "Reading test corpus...\n";
  boost::shared_ptr<ParsedCorpus> test_corpus = boost::make_shared<ParsedCorpus>();
  if (config_->test_file.size()) {
    test_corpus->readFile(config_->test_file, dict_, true);
    std::cerr << "Done reading test corpus..." << std::endl;
  }

  //instantiate weights
  weights_ = boost::make_shared<ParsedWeights>(dict_->size(), dict_->tag_size(), config_->num_actions);

  std::vector<int> sup_indices(sup_training_corpus->size());
  std::iota(sup_indices.begin(), sup_indices.end(), 0);

  std::vector<int> unsup_indices(unsup_training_corpus->size());
  std::iota(unsup_indices.begin(), unsup_indices.end(), 0);

  Real best_perplexity = std::numeric_limits<Real>::infinity();
  Real test_objective = 0; //use later
  int minibatch_counter = 1;
  int minibatch_size = config_->minibatch_size;
  int minibatch_size_unsup = config_->minibatch_size_unsup;

  //TODO parallelize

  //need to record per sentence ...
  //TODO need a proper way to keep track of indices
  // one way is to randomize only once, not for each iteration
  // else keep sentence index while building examples for a minibatch
  std::vector<boost::shared_ptr<ParseDataSet>> sup_examples_list(sup_training_corpus->size(), 
          boost::make_shared<ParseDataSet>());
  std::vector<boost::shared_ptr<ParseDataSet>> unsup_examples_list(unsup_training_corpus->size(), 
          boost::make_shared<ParseDataSet>());
       
  for (int iter = 0; iter < config_->iterations; ++iter) {
    std::cerr << "Training iteration " << iter << std::endl;
    auto iteration_start = get_time(); 
    //int non_projective_count = 0;

    //std::cout << indices.size() << " indices\n";
    if (config_->randomise) {
      std::random_shuffle(sup_indices.begin(), sup_indices.end());
      std::random_shuffle(unsup_indices.begin(), unsup_indices.end());
    }

    size_t start = 0;
    //loop over supervised minibatches
    while (start < sup_training_corpus->size()) {
      size_t end = std::min(sup_training_corpus->size(), start + minibatch_size);
      
      std::vector<int> minibatch(sup_indices.begin() + start, sup_indices.begin() + end);
      boost::shared_ptr<ParseDataSet> minibatch_examples = boost::make_shared<ParseDataSet>();
      boost::shared_ptr<ParseDataSet> old_minibatch_examples = boost::make_shared<ParseDataSet>();
       
      //critical loops
      //FIRST remove old examples
      if (iter > 0) {
        for (auto j: minibatch) 
          old_minibatch_examples->extend(sup_examples_list.at(j));
        weights_->updateRemove(old_minibatch_examples, eng); 
      }

      //THEN add new examples
      for (auto j: minibatch) {
        sup_examples_list.at(j)->clear();
        parse_model_->extractSentence(sup_training_corpus->sentence_at(j), sup_examples_list.at(j));
        //TODO if (iter > 0) alternatively sample from distribution
        minibatch_examples->extend(sup_examples_list.at(j));
      }     

      weights_->updateInsert(minibatch_examples, eng); 

      ++minibatch_counter;
      start = end;
    }     

    start = 0;
    //loop over unsupervised minibatches
    while (start < unsup_training_corpus->size()) {
      if (minibatch_counter % 10000 == 0)
        std::cout << minibatch_counter << std::endl;
      size_t end = std::min(unsup_training_corpus->size(), start + minibatch_size_unsup);
      
      std::vector<int> minibatch(unsup_indices.begin() + start, unsup_indices.begin() + end);
      boost::shared_ptr<ParseDataSet> minibatch_examples = boost::make_shared<ParseDataSet>();
      boost::shared_ptr<ParseDataSet> old_minibatch_examples = boost::make_shared<ParseDataSet>();
      
      //critical loops
      //FIRST remove old examples
      if (iter > 0) {
        for (auto j: minibatch) {
          old_minibatch_examples->extend(unsup_examples_list.at(j));
          //std::cout << j << " ";
        }
        //std::cout << std::endl;
        weights_->updateRemove(old_minibatch_examples, eng); 
      }

      //THEN add new examples
      for (auto j: minibatch) {
        unsup_examples_list.at(j)->clear();
        if (iter == 0)
          parse_model_->extractSentenceUnsupervised(unsup_training_corpus->sentence_at(j),
                            weights_, unsup_examples_list.at(j));
        else 
          parse_model_->extractSentenceUnsupervised(unsup_training_corpus->sentence_at(j),
                            weights_, eng, unsup_examples_list.at(j));
        minibatch_examples->extend(unsup_examples_list.at(j));
      }     

      weights_->updateInsert(minibatch_examples, eng); 

      ++minibatch_counter;
      start = end;

      /*if ((iter == 0) && (minibatch_counter % 10000 == 0)) {
        evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);
      } */
    }

    if ((iter > 0) && (iter % 5 == 0))
      weights_->resampleHyperparameters(eng);

    Real iteration_time = get_duration(iteration_start, get_time());

    std::cerr << "Iteration: " << iter << ", "
             << "Training Time: " << iteration_time << " seconds, "
             << "Training Objective: " << weights_->likelihood() / 
             (sup_training_corpus->numTokens() + unsup_training_corpus->numTokens())
           << "\n\n";
   
    //if (iter%10 == 0)
    evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);
  }

  std::cerr << "Overall minimum perplexity: " << best_perplexity << std::endl;
  
}



template<class ParseModel, class ParsedWeights>
void PypDpModel<ParseModel, ParsedWeights>::learn() {
  MT19937 eng;
  //read training data
  std::cerr << "Reading training corpus...\n";
  boost::shared_ptr<ParsedCorpus> training_corpus = boost::make_shared<ParsedCorpus>();
  training_corpus->readFile(config_->training_file, dict_, false);
  config_->vocab_size = dict_->size();
  config_->num_tags = dict_->tag_size();
  config_->num_labels = dict_->label_size();
  if (config_->labelled_parser) {
    config_->num_actions += 2*(dict_->label_size()-1); //add labelled actions
  }

  std::cerr << "Corpus size: " << training_corpus->size() << " sentences\t (" 
            << dict_->size() << " word types, " << dict_->tag_size() << " tags, "  
	    << dict_->label_size() << " labels)\n";  

  //print tags
  //for (int i = 0; i < dict_->tag_size(); ++i)
  //  std::cout << dict_->lookupTag(i) << "\n";

  //don't worry about working with an existing model now

  //read test data 
  std::cerr << "Reading test corpus...\n";
  boost::shared_ptr<ParsedCorpus> test_corpus = boost::make_shared<ParsedCorpus>();
  if (config_->test_file.size()) {
    test_corpus->readFile(config_->test_file, dict_, true);
    std::cerr << "Done reading test corpus..." << std::endl;
  }

  //instantiate weights

  //instantiate weights
  weights_ = boost::make_shared<ParsedWeights>(dict_->size(), dict_->tag_size(), config_->num_actions);

  /*
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
  }  */

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
       
      //critical loops
      //FIRST remove old examples
      if (iter > 0) {
        for (auto j: minibatch) 
          old_minibatch_examples->extend(examples_list.at(j));
        weights_->updateRemove(old_minibatch_examples, eng); 
      }

      //std::cout << start << std::endl;
      //THEN add new examples
      for (auto j: minibatch) {
        //if (iter == 0) {  //this only takes 1 sec per iteration
          examples_list.at(j)->clear();
          //print labels
          //training_corpus->sentence_at(j).print_arcs();
          //training_corpus->sentence_at(j).print_labels();
          //if (iter == 0)
          parse_model_->extractSentence(training_corpus->sentence_at(j), examples_list.at(j));
          //else
          //  parse_model_->extractSentence(training_corpus->sentence_at(j), weights_, eng, examples_list.at(j));
          if (!training_corpus->sentence_at(j).is_projective_dependency())
            ++non_projective_count;
        //}
        minibatch_examples->extend(examples_list.at(j));
      }     

      weights_->updateInsert(minibatch_examples, eng); 
     
      ++minibatch_counter;
      start = end;
    }     
    //std::cout << std::endl;
   
    if ((iter > 0) && (iter % 5 == 0))
      weights_->resampleHyperparameters(eng);

    Real iteration_time = get_duration(iteration_start, get_time());

    std::cerr << "Iteration: " << iter << ", "
             << "Training Time: " << iteration_time << " seconds, "
             << "Non-projective: " << (non_projective_count + 0.0) / training_corpus->size() << ", "
             << "Training Objective: " << weights_->likelihood() / training_corpus->numTokens() 
           << "\n\n";
    
    if (iter%5 == 0)
      evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);
  }

  std::cerr << "Overall minimum perplexity: " << best_perplexity << std::endl;
  
}

template<class ParseModel, class ParsedWeights>
void PypDpModel<ParseModel, ParsedWeights>::evaluate() const {
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

template<class ParseModel, class ParsedWeights>
void PypDpModel<ParseModel, ParsedWeights>::evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, int minibatch_counter, 
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

template<class ParseModel, class ParsedWeights>
void PypDpModel<ParseModel, ParsedWeights>::evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, Real& accumulator) 
    const {
  MT19937 eng; //should not actually redefine, but it should be ok
  if (test_corpus != nullptr) {
    
    std::vector<int> indices(test_corpus->size());
    std::iota(indices.begin(), indices.end(), 0);
  
    for (unsigned beam_size: config_->beam_sizes) {
      std::ofstream outs;
      outs.open("system.out" + std::to_string(beam_size)); //config_->test_file

      std::cerr << "parsing with beam size " << beam_size << ":\n";
      accumulator = 0;
      auto beam_start = get_time();
      boost::shared_ptr<AccuracyCounts> acc_counts = boost::make_shared<AccuracyCounts>(dict_);

      size_t start = 0;
      while (start < test_corpus->size()) {
        size_t end = std::min(start + config_->minibatch_size, test_corpus->size());

        //std::vector<int> minibatch = scatterMinibatch(start, end, indices);
        std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
        Real objective = 0;
            
        //TODO parallize, maybe move
        for (auto j: minibatch) {
          Parser parse = parse_model_->evaluateSentence(test_corpus->sentence_at(j), weights_, acc_counts, beam_size);
          objective += parse.weight();

          //write output to conll-format file
          for (unsigned i = 1; i < parse.size(); ++i) { 
            outs << i << "\t" << dict_->lookup(parse.word_at(i)) << "\t_\t_\t" 
                 << dict_->lookupTag(parse.tag_at(i)) << "\t_\t" << parse.arc_at(i) << "\t"
                 << dict_->lookupLabel(parse.label_at(i)) << "\t_\t_\n";
          }
          outs << "\n";
        }

        accumulator += objective;
        start = end;
      } 

      outs.close();
      Real beam_time = get_duration(beam_start, get_time());
      Real sents_per_sec = static_cast<int>(test_corpus->size())/beam_time;
      Real tokens_per_sec = static_cast<int>(test_corpus->numTokens())/beam_time;
      std::cerr << "(" << beam_time << "s, " <<
                   static_cast<int>(sents_per_sec) << " sentences per second, " <<
                   static_cast<int>(tokens_per_sec) << " tokens per second)\n";
      acc_counts->printAccuracy();
    }
  }
}

template class PypDpModel<ArcStandardParseModel<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>, ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>;
template class PypDpModel<ArcStandardLabelledParseModel<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>, ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>;
template class PypDpModel<ArcEagerParseModel<ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>, ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>;
template class PypDpModel<EisnerParseModel<ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>, ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>;

template class PypDpModel<ArcStandardParseModel<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>, ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>;
template class PypDpModel<ArcStandardLabelledParseModel<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>, ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>;
template class PypDpModel<ArcEagerParseModel<ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>, ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>;
template class PypDpModel<EisnerParseModel<ParsedPypWeights<tagLMOrderE, 1>>, ParsedPypWeights<tagLMOrderE, 1>>;

} //namespace oxlm

