#include "gdp/pyp_dp_model.h"

namespace oxlm {

template<class ParseModel, class ParsedWeights>
PypDpModel<ParseModel, ParsedWeights>::PypDpModel() {
  dict_ = boost::make_shared<Dict>(true);
  ch_dict_ = boost::make_shared<Dict>("<s>", " ");
} 

template<class ParseModel, class ParsedWeights>
PypDpModel<ParseModel, ParsedWeights>::PypDpModel(const boost::shared_ptr<ModelConfig>& config): 
    config_(config) {
  dict_ = boost::make_shared<Dict>(config->root_first);
  ch_dict_ = boost::make_shared<Dict>("<s>", " ");
  
  if (config_->parser_type == ParserType::eisner) {
    dict_->convert("<stop>", false);  //add terminating symbol
  }

  parse_model_ = boost::make_shared<ParseModel>(config);
}

template<class ParseModel, class ParsedWeights>
void PypDpModel<ParseModel, ParsedWeights>::learn() {
  MT19937 eng;
  //read training data
  boost::shared_ptr<ParsedCorpus> sup_training_corpus = boost::make_shared<ParsedCorpus>(config_);
  boost::shared_ptr<ParsedCorpus> unsup_training_corpus = boost::make_shared<ParsedCorpus>(config_);
 
  if (config_->training_file.size()) { 
    std::cerr << "Reading supervised training corpus...\n";
    sup_training_corpus->readFile(config_->training_file, dict_, false);
     std::cerr << "Corpus size: " << sup_training_corpus->size() << " sentences\t (" 
              << dict_->size() << " word types, " << dict_->tag_size() << " tags, " 
	      << dict_->label_size() << " labels)\n";  
  } else {
    std::cerr << "No supervised training corpus.\n";
  } 

  if (config_->training_file_ques.size()) { 
    boost::shared_ptr<ParsedCorpus> ques_training_corpus = boost::make_shared<ParsedCorpus>(config_);
    std::cerr << "Reading question training corpus...\n";
    ques_training_corpus->readFile(config_->training_file_ques, dict_, false);
    std::cerr << "Corpus size: " << ques_training_corpus->size() << " sentences\t (" 
              << dict_->size() << " word types, " << dict_->tag_size() << " tags, "  
	      << dict_->label_size() << " labels)\n";  

    //duplicate sup data x3
    unsigned sup_size = sup_training_corpus->size();
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < sup_size; ++j)
        sup_training_corpus->add_sentence(sup_training_corpus->sentence_at(j));
  
    //insert ques data x50 into sup corpus
    unsigned ques_size = ques_training_corpus->size();
    for (int k = 0; k < 100; ++k)
      for (int j = 0; j < ques_size; ++j)
        sup_training_corpus->add_sentence(ques_training_corpus->sentence_at(j));
  } 

  if (config_->semi_supervised && config_->training_file_unsup.size()) { 
    std::cerr << "Reading unsupervised training corpus...\n";
    unsup_training_corpus->readFile(config_->training_file_unsup, dict_, false);
     std::cerr << "Corpus size: " << unsup_training_corpus->size() << " sentences\t (" 
              << dict_->size() << " word types, " << dict_->tag_size() << " tags, "  
	      << dict_->label_size() << " labels)\n";  
  } else {
    std::cerr << "No unsupervised training corpus.\n";
  }

  //construct character dictionary
  for (size_t j = 0; j < dict_->size(); ++j) {
    const Word w = dict_->lookup(j);
    for (size_t i = 0; i < w.size(); ++i)
        ch_dict_->convert(w.substr(i,1), false);
  }

  //read test data 
  std::cerr << "Reading test corpus...\n";
  boost::shared_ptr<ParsedCorpus> test_corpus = boost::make_shared<ParsedCorpus>(config_);
  if (config_->test_file.size()) {
    test_corpus->readFile(config_->test_file, dict_, true);
    std::cerr << "Done reading test corpus..." << std::endl;
    std::cerr << "Corpus size: " << test_corpus->size() << " sentences\t (" 
                 << test_corpus->numTokens() << " tokens\t (" 
              << dict_->size() << " word types, " << dict_->tag_size() << " tags, "  
	      << dict_->label_size() << " labels)\n";  
  }

  bool write_data = false;

  if (write_data) {
    std::ofstream outs;
    //write vocabularies
    outs.open("conll-dataset/words.list"); 
    for (int i = 0; i < dict_->size(); ++i)
     outs << i << " " << dict_->lookup(i) << "\n";
    outs.close();
    outs.open("conll-dataset/pos.list"); 
    for (int i = 0; i < dict_->tag_size(); ++i)
     outs << i << " " << dict_->lookupTag(i) << "\n";
    outs.close();
    outs.open("conll-dataset/labels.list"); 
    for (int i = 0; i < dict_->label_size(); ++i)
     outs << i << " " << dict_->lookupLabel(i) << "\n";
    outs.close(); 

    //extract sup training data
    outs.open("conll-dataset/conll.train.data"); 
    for (int j = 0; j < sup_training_corpus->size(); ++j) {
      for (int i = 0; i < sup_training_corpus->sentence_at(j).size(); ++i)
        outs << sup_training_corpus->sentence_at(j).word_at(i) << " ";
        //outs << sup_training_corpus->sentence_at(j).tag_at(i) << " ";
      outs << "\n";
      for (int i = 0; i < sup_training_corpus->sentence_at(j).size(); ++i)
        outs << sup_training_corpus->sentence_at(j).arc_at(i) << " ";
      outs << "\n";
      for (int i = 0; i < sup_training_corpus->sentence_at(j).size(); ++i)
        outs << sup_training_corpus->sentence_at(j).label_at(i) << " ";
      outs << "\n";

      boost::shared_ptr<ParseDataSet> examples = boost::make_shared<ParseDataSet>(); 
      parse_model_->extractSentence(sup_training_corpus->sentence_at(j), examples);
      for (int i = 0; i < examples->action_example_size(); ++i) {
        outs << examples->action_at(i) << "\t";
        for (WordId cw: examples->action_context_at(i)) 
          outs << cw << " ";
        outs << "\n";
      }
      outs << "\n"; 
    }
    outs.close(); 

    //extract dev training data
    outs.open("conll-dataset/conll.dev.data"); 
    for (int j = 0; j < test_corpus->size(); ++j) {
      for (int i = 0; i < test_corpus->sentence_at(j).size(); ++i)
        outs << test_corpus->sentence_at(j).word_at(i) << " ";
        //outs << test_corpus->sentence_at(j).tag_at(i) << " ";
      outs << "\n";
      for (int i = 0; i < test_corpus->sentence_at(j).size(); ++i)
        outs << test_corpus->sentence_at(j).arc_at(i) << " ";
      outs << "\n";
      for (int i = 0; i < test_corpus->sentence_at(j).size(); ++i)
        outs << test_corpus->sentence_at(j).label_at(i) << " ";
      outs << "\n";
      outs << "\n"; 
    }

    outs.close();
  }

  //instantiate weights
  weights_ = boost::make_shared<ParsedWeights>(dict_, ch_dict_, config_->numActions());

  std::vector<int> sup_indices(sup_training_corpus->size());
  std::iota(sup_indices.begin(), sup_indices.end(), 0);

  std::vector<int> unsup_indices(unsup_training_corpus->size());
  std::iota(unsup_indices.begin(), unsup_indices.end(), 0);

  Real best_perplexity = std::numeric_limits<Real>::infinity();
  Real test_objective = 0; 
  int minibatch_counter = 1;
  int minibatch_size = config_->minibatch_size;
  int minibatch_size_unsup = config_->minibatch_size_unsup;

  std::vector<boost::shared_ptr<ParseDataSet>> sup_examples_list;
  for (int i = 0; i < sup_training_corpus->size(); ++i)
    sup_examples_list.push_back(boost::make_shared<ParseDataSet>());
          
  std::vector<boost::shared_ptr<ParseDataSet>> unsup_examples_list;
  for (int i = 0; i < unsup_training_corpus->size(); ++i)
    unsup_examples_list.push_back(boost::make_shared<ParseDataSet>());
       
  for (int iter = 0; iter < config_->iterations; ++iter) {
    std::cerr << "Training iteration " << iter << std::endl;
    auto iteration_start = get_time(); 
    int non_projective_count = 0;

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
       
      //FIRST remove old examples
      if (iter > 0) {
        for (auto j: minibatch) {
          old_minibatch_examples->extend(sup_examples_list.at(j));
        }

        weights_->updateRemove(old_minibatch_examples, eng); 
      }

      //THEN add new examples
      for (auto j: minibatch) {
        sup_examples_list.at(j)->clear();
        if (!config_->bootstrap || (iter == 0))
          parse_model_->extractSentence(sup_training_corpus->sentence_at(j), sup_examples_list.at(j));
        else
          parse_model_->extractSentence(sup_training_corpus->sentence_at(j), weights_, eng, sup_examples_list.at(j));
        if (!sup_training_corpus->sentence_at(j).projective_dependency())
          ++non_projective_count;
        minibatch_examples->extend(sup_examples_list.at(j));
      }     

      weights_->updateInsert(minibatch_examples, eng); 

      ++minibatch_counter;
      start = end;
    }     

    start = 0;
    //loop over unsupervised minibatches
    while (start < unsup_training_corpus->size()) {
      if (minibatch_counter % 5000 == 0)
        std::cerr << minibatch_counter << std::endl;
      size_t end = std::min(unsup_training_corpus->size(), start + minibatch_size_unsup);
      
      std::vector<int> minibatch(unsup_indices.begin() + start, unsup_indices.begin() + end);
      boost::shared_ptr<ParseDataSet> minibatch_examples = boost::make_shared<ParseDataSet>();
      boost::shared_ptr<ParseDataSet> old_minibatch_examples = boost::make_shared<ParseDataSet>();
      
      //FIRST remove old examples
      if (iter > 0) {
        for (auto j: minibatch) {
          old_minibatch_examples->extend(unsup_examples_list.at(j));
        }
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

      if ((iter == 0) && (minibatch_counter % 20000 == 0)) {
        evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);
      } 
    }

    if (iter % 5 == 0)
      weights_->resampleHyperparameters(eng);

    Real iteration_time = get_duration(iteration_start, get_time());

    std::cerr << "Iteration: " << iter << ", "
             << "Training Time: " << iteration_time << " seconds, "
             << "Non-projective: " << (non_projective_count + 0.0) / sup_training_corpus->size() << ", "
             << "Training Objective: " << weights_->likelihood() / 
             (sup_training_corpus->numTokens() + unsup_training_corpus->numTokens())
           << "\n\n";
   
    if (iter%5 == 0)
      evaluate(test_corpus, minibatch_counter, test_objective, best_perplexity);
  }

  std::cerr << "Overall minimum perplexity: " << best_perplexity << std::endl;

  //generate from model
  for (int i = 0; i < config_->generate_samples; ++i) {
    Parser parse = parse_model_->generateSentence(weights_, eng);
    std::cout << parse.weight() << "  ";
    parse.print_sentence(dict_);
    parse.print_arcs();
    parse.print_labels();
  }  
}

template<class ParseModel, class ParsedWeights>
void PypDpModel<ParseModel, ParsedWeights>::evaluate() const {
 //read test data 
  boost::shared_ptr<ParsedCorpus> test_corpus = boost::make_shared<ParsedCorpus>(config_);
  test_corpus->readFile(config_->test_file, dict_, true);
  std::cerr << "Done reading test corpus..." << std::endl;
  
  Real log_likelihood = 0;
  evaluate(test_corpus, log_likelihood);
    
  size_t test_size = 3*test_corpus->numTokens(); //approx
  Real test_perplexity = perplexity(log_likelihood, test_size); 
  std::cerr << "Test Perplexity: " << test_perplexity << std::endl;
}

template<class ParseModel, class ParsedWeights>
void PypDpModel<ParseModel, ParsedWeights>::evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, int minibatch_counter, 
                   Real& log_likelihood, Real& best_perplexity) const {
  if (test_corpus != nullptr) {
    evaluate(test_corpus, log_likelihood);
    
    size_t test_size = 3*test_corpus->numTokens(); //approx
    Real test_perplexity = perplexity(log_likelihood, test_size); 
    std::cerr << "\tMinibatch " << minibatch_counter << ", "
         << "Test Perplexity: " << test_perplexity << std::endl;

    if (test_perplexity < best_perplexity) 
      best_perplexity = test_perplexity;
  }
}

template<class ParseModel, class ParsedWeights>
void PypDpModel<ParseModel, ParsedWeights>::evaluate(const boost::shared_ptr<ParsedCorpus>& test_corpus, Real& accumulator) 
    const {
  MT19937 eng; 
  if (test_corpus != nullptr) {
    std::vector<int> indices(test_corpus->size());
    std::iota(indices.begin(), indices.end(), 0);
  
    for (unsigned beam_size: config_->beam_sizes) {
      std::ofstream outs;
      outs.open(config_->test_output_file); 

      std::cerr << "parsing with beam size " << beam_size << ":\n";
      accumulator = 0;
      auto beam_start = get_time();
      boost::shared_ptr<AccuracyCounts> acc_counts = boost::make_shared<AccuracyCounts>(dict_);

      size_t start = 0;
      while (start < test_corpus->size()) {
        size_t end = std::min(start + config_->minibatch_size, test_corpus->size());

        std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
        Real objective = 0;
            
        for (auto j: minibatch) {
          //Parser parse = parse_model_->evaluateSentence(test_corpus->sentence_at(j), weights_, eng, acc_counts, beam_size); 
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

//template class PypDpModel<ArcStandardParseModel<ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>, ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>;
template class PypDpModel<ArcStandardLabelledParseModel<ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>, ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>;
//template class PypDpModel<ArcEagerParseModel<ParsedChLexPypWeights<wordLMOrderAE, charLMOrder, tagLMOrderAE, actionLMOrderAE>>, ParsedChLexPypWeights<wordLMOrderAE, charLMOrder, tagLMOrderAE, actionLMOrderAE>>;
template class PypDpModel<ArcEagerLabelledParseModel<ParsedChLexPypWeights<wordLMOrderAE, charLMOrder, tagLMOrderAE, actionLMOrderAE>>, ParsedChLexPypWeights<wordLMOrderAE, charLMOrder, tagLMOrderAE, actionLMOrderAE>>;
template class PypDpModel<EisnerParseModel<ParsedChLexPypWeights<wordLMOrderE, charLMOrder, tagLMOrderE, 1>>, ParsedChLexPypWeights<wordLMOrderE, charLMOrder, tagLMOrderE, 1>>;

//template class PypDpModel<ArcStandardParseModel<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>, ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>;
template class PypDpModel<ArcStandardLabelledParseModel<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>, ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>;
//template class PypDpModel<ArcEagerParseModel<ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>, ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>;
template class PypDpModel<ArcEagerLabelledParseModel<ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>, ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>;
template class PypDpModel<EisnerParseModel<ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>, ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>;

//template class PypDpModel<ArcStandardParseModel<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>, ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>;
template class PypDpModel<ArcStandardLabelledParseModel<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>, ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>;
//template class PypDpModel<ArcEagerParseModel<ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>, ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>;
template class PypDpModel<ArcEagerLabelledParseModel<ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>, ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>;
template class PypDpModel<EisnerParseModel<ParsedPypWeights<tagLMOrderE, 1>>, ParsedPypWeights<tagLMOrderE, 1>>;

} //namespace oxlm

