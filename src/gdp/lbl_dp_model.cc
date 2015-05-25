#include <iomanip>

#include <boost/make_shared.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/factored_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/metadata.h"
#include "lbl/model_utils.h"
#include "lbl/operators.h"
#include "lbl/weights.h"
#include "utils/conditional_omp.h"

#include "gdp/lbl_dp_model.h"

namespace oxlm {

template<class ParseModel, class ParsedWeights, class Metadata>
LblDpModel<ParseModel, ParsedWeights, Metadata>::LblDpModel() {
  dict = boost::make_shared<Dict>(true);
}

template<class ParseModel, class ParsedWeights, class Metadata>
LblDpModel<ParseModel, ParsedWeights, Metadata>::LblDpModel(
    const boost::shared_ptr<ModelConfig>& config)
    : config(config) {
  dict = boost::make_shared<Dict>(true);
  parse_model = boost::make_shared<ParseModel>(config);
  
  if (config->parser_type == ParserType::eisner) {
    dict->convert("<stop>", false); //add terminating symbol
  }

  metadata = boost::make_shared<Metadata>(config, dict);
  srand(1);
}

template<class ParseModel, class ParsedWeights, class Metadata>
boost::shared_ptr<Dict> LblDpModel<ParseModel, ParsedWeights, Metadata>::getDict() const {
  return dict;
}

template<class ParseModel, class ParsedWeights, class Metadata>
boost::shared_ptr<ModelConfig> LblDpModel<ParseModel, ParsedWeights, Metadata>::getConfig() const {
  return config;
}


template<class ParseModel, class ParsedWeights, class Metadata>
MatrixReal LblDpModel<ParseModel, ParsedWeights, Metadata>::getWordVectors() const {
  return weights->getWordVectors();
}

template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::learn() {
  MT19937 eng;
  boost::shared_ptr<ParsedCorpus> training_corpus = boost::make_shared<ParsedCorpus>(config);
  boost::shared_ptr<ParsedCorpus> unsup_training_corpus = boost::make_shared<ParsedCorpus>(config);
  training_corpus->readFile(config->training_file, dict, false);
  std::cerr << "Done reading training corpus..." << std::endl;
  std::cerr << "Corpus size: " << training_corpus->size() << " sentences" << std::endl; 

  if (config->semi_supervised && config->training_file_unsup.size()) { 
    std::cerr << "Reading unsupervised training corpus...\n";
    //unsup_training_corpus->readTxtFile(config->training_file + ".txt", dict, false);
    unsup_training_corpus->readTxtFile(config->training_file_unsup, dict, false);
    std::cerr << "Corpus size: " << unsup_training_corpus->size() << " sentences" << std::endl;

    
  } else {
    std::cerr << "No unsupervised training corpus.\n";
  }

  //copy tag feature map
  for (unsigned i = 0; i < dict->tag_size(); ++i) 
    config->tag_to_feature.push_back(dict->tagToFeature(i));

  //add transitions as features
  dict->convertFeature("ACTION_NULL", false);
  config->action_feature_index = dict->feature_size();
  for (unsigned i = 0; i < config->numActions(); ++i) 
    dict->convertFeature("ACTION_" + std::to_string(i), false);


  if (config->label_features) {
    //add labels to feature vocab
    config->label_feature_index = dict->feature_size();
    for (unsigned i = 0; i < dict->label_size(); ++i) 
      dict->convertFeature("LABEL_" + dict->lookupLabel(i), false);
  }

  if (config->distance_features) {
    config->distance_feature_index = dict->feature_size();
    size_t range = config->distance_range;
    for (size_t i = 0; i < range; ++i)
      dict->convertFeature("LV_" + std::to_string(i), false);
    for (size_t i = 0; i < range; ++i)
      dict->convertFeature("RV_" + std::to_string(i), false);
    for (size_t i = 1; i <= range; ++i)
      dict->convertFeature("BufDist_" + std::to_string(i), false);
    for (size_t i = 1; i <= range; ++i)
      dict->convertFeature("HeadDist_" + std::to_string(i), false);
  }

  //update vocab sizes
  config->num_features = dict->feature_size();
  config->num_train_sentences = training_corpus->size();

  //copy word feature map
  for (unsigned i = 0; i < dict->size(); ++i) {
    config->addWordFeatures(dict->getWordFeatures(i));
  }

  std::cerr << dict->size() << " word types, " << dict->feature_size() << " features, "  
	    << dict->label_size() << " labels)\n";  

  boost::shared_ptr<ParsedCorpus> test_corpus; 
  if (config->test_file.size()) {
    test_corpus = boost::make_shared<ParsedCorpus>(config);
    test_corpus->readFile(config->test_file, dict, true);

    std::cerr << "Done reading test corpus..." << endl;
  }

  std::cerr << "Reading test corpus 2...\n";
  boost::shared_ptr<ParsedCorpus> test_corpus2 = boost::make_shared<ParsedCorpus>(config);
  if (config->test_file2.size()) {
    test_corpus2->readFile(config->test_file2, dict, true);
    std::cerr << "Done reading test corpus 2..." << std::endl;
    std::cerr << "Corpus size: " << test_corpus2->size() << " sentences,\t" 
                 << test_corpus2->numTokens() << " tokens\n";
  }

  std::cerr << "Reading test corpus unsup...\n";
  boost::shared_ptr<ParsedCorpus> test_corpus_unsup = boost::make_shared<ParsedCorpus>(config);
  if (config->test_file_unsup.size()) {
    test_corpus_unsup->readTxtFile(config->test_file_unsup, dict, true);
    std::cerr << "Done reading unsup test corpus..." << std::endl;
    std::cerr << "Corpus size: " << test_corpus_unsup->size() << " sentences,\t" 
                 << test_corpus_unsup->numTokens() << " tokens\n";
  }
  
  if (config->model_input_file.size() == 0) {
    metadata->initialize(training_corpus);
    weights = boost::make_shared<ParsedWeights>(config, metadata, true);
  } else {
    Real log_likelihood = 0;
    evaluate(test_corpus, log_likelihood);
    std::cerr << "Initial perplexity: "
         << perplexity(log_likelihood, test_corpus->numTokens()) << endl;
  }

  vector<int> indices(training_corpus->size());
  iota(indices.begin(), indices.end(), 0);

  vector<int> unsup_indices(unsup_training_corpus->size());
  iota(unsup_indices.begin(), unsup_indices.end(), 0);

  Real best_perplexity = numeric_limits<Real>::infinity();
  Real best_perplexity2 = numeric_limits<Real>::infinity();
  Real best_perplexity_unsup = numeric_limits<Real>::infinity();
  Real test_objective = 0;
  Real test_objective2 = 0;
  Real test_objective_unsup = 0;

  Real best_global_objective = numeric_limits<Real>::infinity();
  bool objective_improved = true;
  bool stop_training = false;
  bool stop_sv_training = false;
  Real global_objective = 0;

  boost::shared_ptr<ParsedWeights> global_gradient =
      boost::make_shared<ParsedWeights>(config, metadata, false);
  boost::shared_ptr<ParsedWeights> adagrad =
      boost::make_shared<ParsedWeights>(config, metadata, false);
  MinibatchWords global_words;

  int shared_index = 0;
  // For no particular reason. It just looks like this works best.
  int task_size = sqrt(config->minibatch_size) / 4; //to imitate word-level behaviour
 
  /* boost::shared_ptr<AccuracyCounts> temp_acc_counts = boost::make_shared<AccuracyCounts>(dict);
  for (unsigned j = 0; j < unsup_training_corpus->size(); ++j) {
        //std::cout << j << std::endl;
        ParsedSentence temp = unsup_training_corpus->sentence_at(j);
        //std::cout << temp.size() << std::endl;
        Parser parse = parse_model->evaluateSentence(temp, weights, temp_acc_counts, false, config->num_particles); 
        //std::cout << parse.size() << std::endl;
        //if (j > 50) {
        //for (unsigned i = 0; i < parse.size(); ++i)
        //  std::cout << " " << parse.arc_at(i) << std::endl;
          //if (j != 38)
          unsup_training_corpus->set_arcs_at(j, parse);
        //}
        std::cout << "parsed" << std::endl;
    } */ 

  omp_set_num_threads(config->threads);
  #pragma omp parallel
  {
    int minibatch_counter = 1;
    int minibatch_size = config->minibatch_size;
    boost::shared_ptr<ParsedWeights> gradient =
        boost::make_shared<ParsedWeights>(config, metadata, false);

    for (int iter = 0; (iter < config->iterations) && !stop_training; ++iter) {
      Real best_iter_objective = numeric_limits<Real>::infinity();

      for (int sv_iter = 0; (sv_iter < config->iterations_sv + 1) && !stop_sv_training; ++sv_iter) {
        auto iteration_start = get_time();
        bool sentences_only = (sv_iter > 0);

        #pragma omp master
        {
          std::cerr << "Training iteration " << iter << " (" << sv_iter << ")" << std::endl;
          if (config->randomise) {
            random_shuffle(indices.begin(), indices.end());
          }
          global_objective = 0;
        }
        // Wait until the master thread finishes shuffling the indices.
        #pragma omp barrier

        size_t start = 0;
        while (start < training_corpus->size()) {
          //std::cout << start << std::endl;
          size_t end = min(training_corpus->size(), start + minibatch_size);

          vector<int> minibatch(
              indices.begin() + start,
              min(indices.begin() + end, indices.end()));

          // Reset the set of minibatch words shared across all threads.
          #pragma omp master
          {
            global_words = MinibatchWords();
            shared_index = 0;
          }

          // Wait until the global gradient is initialized. Otherwise, some
          // gradient updates may be ignored.
          #pragma omp barrier

          Real objective = 0;
          int num_examples = 0;
          MinibatchWords words;
          size_t task_start;
          while (true) {
            #pragma omp critical
            {
              task_start = shared_index;
              shared_index += task_size;
            }

            if (task_start < minibatch.size()) {
              size_t task_end = min(task_start + task_size, minibatch.size());
              vector<int> task(
                  minibatch.begin() + task_start, minibatch.begin() + task_end);
              //collect the training examples for the minibatch
              boost::shared_ptr<ParseDataSet> task_examples = boost::make_shared<ParseDataSet>();
                          
              for (int j: task) {
                if ((!config->bootstrap && (config->bootstrap_iter == 0)) || 
                       (iter < config->bootstrap_iter))
                  parse_model->extractSentence(training_corpus->sentence_at(j), task_examples);
                else if (config->bootstrap)
                  parse_model->extractSentence(training_corpus->sentence_at(j), weights, task_examples);
                else 
                  parse_model->extractSentenceUnsupervised(training_corpus->sentence_at(j), weights, task_examples);
              }
              num_examples += task_examples->word_example_size() + task_examples->action_example_size();
              //std::cout << task_examples->tag_example_size() << std::endl;
              if (config->predict_pos)
                num_examples += task_examples->tag_example_size();

              if (config->noise_samples > 0) {
                weights->estimateGradient(
                    task_examples, gradient, objective, words);
              } else {
                //std::cout << " " << task_start;
                weights->getGradient(
                    task_examples, gradient, objective, words);
                
             /* if (!weights->checkGradient(task_examples, global_gradient, EPS))  
                 std::cout << "gradient check failed" << std::endl;
                else
                 std::cout << "gradient OK" << std::endl; */
              }
            } else {
              break;
            }
          }
          //std::cout << std::endl;

          global_gradient->syncUpdate(words, gradient, sentences_only);
          #pragma omp critical
          {
            global_objective += objective;
            global_words.merge(words);
          }

          // Wait until the global gradient is fully updated by all threads and
          // the global words are fully merged.
          #pragma omp barrier

          // Prepare minibatch words for parallel processing.
          #pragma omp master
          global_words.transform();

          // Wait until the minibatch words are fully prepared for parallel
          // processing.
          #pragma omp barrier
          update(global_words, global_gradient, adagrad, sentences_only);

          // Wait for all threads to finish making the model gradient update.
          #pragma omp barrier
          Real minibatch_factor =
              static_cast<Real>(num_examples) / (3*training_corpus->numTokens());
          if (config->predict_pos)
            minibatch_factor =
              static_cast<Real>(num_examples) / (4*training_corpus->numTokens());
          //approx total number of predictions
          
          objective = regularize(global_gradient, minibatch_factor, sentences_only);
          #pragma omp critical
          global_objective += objective;

          // Clear gradients.
          gradient->clear(words, false);
          global_gradient->clear(global_words, true);

          // Wait the regularization update to finish and make sure the global
          // words are reset only after the global gradient is fully cleared.
          #pragma omp barrier

          /* if (minibatch_counter % 1000 == 0) {
            evaluate(test_corpus, iteration_start, minibatch_counter,
                     test_objective, best_perplexity);
          } */

          ++minibatch_counter;
          start = end;
        }

        #pragma omp master
        {
          Real iteration_time = get_duration(iteration_start, get_time());
          std::cerr << "Iteration: " << iter << " (" << sv_iter << "), "
               << "Time: " << iteration_time << " seconds, "
               << "  Likelihood: " << global_objective 
               << "  Size: " << training_corpus->numTokens()
               << "  Perplexity: " << perplexity(global_objective, training_corpus->numTokens())
               << "  Objective: " << global_objective / training_corpus->numTokens()
               << endl << endl;
       
          if (global_objective <= best_iter_objective) {
            best_iter_objective = global_objective;
          } else {
            stop_sv_training = true;
          }

          
          if (global_objective <= best_global_objective) {
            best_global_objective = global_objective;
            objective_improved = true;
          } else if (stop_sv_training || (sv_iter == config->iterations_sv))  {
            if (!objective_improved)
              stop_training = true;
            objective_improved = false;
          }

          //if (iter%5 == 0)
          evaluate(test_corpus, iteration_start, minibatch_counter,
                 test_objective, best_perplexity);
          if (config->semi_supervised)
            evaluate(test_corpus_unsup, iteration_start, minibatch_counter,
                 test_objective_unsup, best_perplexity_unsup);
          else if (config->test_file2.size()) 
            evaluate(test_corpus2, iteration_start, minibatch_counter,
                 test_objective2, best_perplexity2);
        }
      }
    }

    if (config->semi_supervised) {
      #pragma omp master
      {
        std::cerr << "Parsing unlabelled data" << std::endl;
      }

      boost::shared_ptr<AccuracyCounts> temp_acc_counts = boost::make_shared<AccuracyCounts>(dict);
      for (unsigned j = 0; j < unsup_training_corpus->size(); ++j) {
        //ParsedSentence temp = unsup_training_corpus->sentence_at(j);
        //Parser parse = parse_model->evaluateSentence(temp, weights, temp_acc_counts, false, config->num_particles); 
        Parser parse = parse_model->evaluateSentence(unsup_training_corpus->sentence_at(j), weights, temp_acc_counts, false, config->num_particles); 
        unsup_training_corpus->set_arcs_at(j, parse);
        unsup_training_corpus->set_labels_at(j, parse);
        
        //unsup_training_corpus->sentence_at(j).print_arcs();
        //unsup_training_corpus->sentence_at(j).print_sentence(dict);
        //for (unsigned i = 0; i < parse.size(); ++i)
        //  std::cout << unsup_training_corpus->sentence_at(j).arc_at(i) << " ";
        //std::cout << std::endl;
      }
  
      #pragma omp master
      Real best_global_objective = numeric_limits<Real>::infinity();
    }

    for (int iter = 0; (iter < config->iterations_unsup) && config->semi_supervised; ++iter) {
      auto iteration_start = get_time();

      #pragma omp master
      {
        std::cerr << "Unsup Training iteration " << iter << std::endl;
        if (config->randomise) {
          random_shuffle(unsup_indices.begin(), unsup_indices.end());
        }
        global_objective = 0;
      }
      // Wait until the master thread finishes shuffling the indices.
      #pragma omp barrier

      size_t start = 0;
      while (start < unsup_training_corpus->size()) {
        //std::cout << start << std::endl;
        size_t end = min(unsup_training_corpus->size(), start + minibatch_size);

        vector<int> minibatch(
            unsup_indices.begin() + start,
            min(unsup_indices.begin() + end, unsup_indices.end()));

        // Reset the set of minibatch words shared across all threads.
        #pragma omp master
        {
          global_words = MinibatchWords();
          shared_index = 0;
        }

        // Wait until the global gradient is initialized. Otherwise, some
        // gradient updates may be ignored.
        #pragma omp barrier

        Real objective = 0;
        int num_examples = 0;
        MinibatchWords words;
        size_t task_start;
        while (true) {
          #pragma omp critical
          {
            task_start = shared_index;
            shared_index += task_size;
          }

          if (task_start < minibatch.size()) {
            size_t task_end = min(task_start + task_size, minibatch.size());
            vector<int> task(
                minibatch.begin() + task_start, minibatch.begin() + task_end);
            //collect the training examples for the minibatch
            boost::shared_ptr<ParseDataSet> task_examples = boost::make_shared<ParseDataSet>();
            
            for (int j: task) {
              //unsup_training_corpus->sentence_at(j).print_arcs();
              parse_model->extractSentence(unsup_training_corpus->sentence_at(j), task_examples);
              //std::cout << std::endl;
              //for (unsigned i = 0; i < task_examples->action_example_size(); ++i) 
              //  std::cout << task_examples->action_at(i) << " ";
              //std::cout << std::endl;
            }
            num_examples += task_examples->word_example_size() + task_examples->action_example_size();
            if (config->predict_pos)
              num_examples += task_examples->tag_example_size();

            if (config->noise_samples > 0) {
              weights->estimateGradient(
                  task_examples, gradient, objective, words);
            } else {
              //std::cout << " " << task_start;
              weights->getGradient(
                  task_examples, gradient, objective, words);
              
           /* if (!weights->checkGradient(task_examples, global_gradient, EPS))  
               std::cout << "gradient check failed" << std::endl;
              else
               std::cout << "gradient OK" << std::endl; */
            }
          } else {
            break;
          }
        }
        //std::cout << std::endl;

        global_gradient->syncUpdate(words, gradient, false);
        #pragma omp critical
        {
          global_objective += objective;
          global_words.merge(words);
        }

        // Wait until the global gradient is fully updated by all threads and
        // the global words are fully merged.
        #pragma omp barrier

        // Prepare minibatch words for parallel processing.
        #pragma omp master
        global_words.transform();

        // Wait until the minibatch words are fully prepared for parallel
        // processing.
        #pragma omp barrier
        update(global_words, global_gradient, adagrad, false);

        // Wait for all threads to finish making the model gradient update.
        #pragma omp barrier
        Real minibatch_factor =
            static_cast<Real>(num_examples) / (3*unsup_training_corpus->numTokens());
        if (config->predict_pos)
          minibatch_factor =
            static_cast<Real>(num_examples) / (4*unsup_training_corpus->numTokens());

        //approx total number of predictions
        
        objective = regularize(global_gradient, minibatch_factor, false);
        #pragma omp critical
        global_objective += objective;

        // Clear gradients.
        gradient->clear(words, false);
        global_gradient->clear(global_words, true);

        // Wait the regularization update to finish and make sure the global
        // words are reset only after the global gradient is fully cleared.
        #pragma omp barrier
        if ((iter == 0) && (minibatch_counter % 10000 == 0)) {
          evaluate(test_corpus_unsup, iteration_start, minibatch_counter,
             test_objective_unsup, best_perplexity_unsup);
        } 

        ++minibatch_counter;
        start = end;
      }

      #pragma omp master
      {
        Real iteration_time = get_duration(iteration_start, get_time());
        std::cerr << "Unsup Iteration: " << iter << ", "
             << "Time: " << iteration_time << " seconds, "
             << "  Likelihood: " << global_objective 
             << "  Size: " << unsup_training_corpus->numTokens()
             << "  Perplexity: " << perplexity(global_objective, unsup_training_corpus->numTokens())
             << "  Objective: " << global_objective / unsup_training_corpus->numTokens()
             << endl << endl;
     
        if (global_objective <= best_global_objective) {
          best_global_objective = global_objective;
        } /*else {
          improve_objective = false;
        } */

        //if (iter%5 == 0)
        evaluate(test_corpus, iteration_start, minibatch_counter,
               test_objective, best_perplexity);
        evaluate(test_corpus_unsup, iteration_start, minibatch_counter,
             test_objective_unsup, best_perplexity_unsup);
      }
    }
  }

  std::cerr << "Overall minimum perplexity: " << best_perplexity << endl;

  //generate from model
  for (int i = 0; i < config->generate_samples; ++i) {
    Parser parse = parse_model->generateSentence(weights, eng);
    //std::cout << parse.weight() << "  ";
    parse.print_sentence(dict);
    parse.print_arcs();
    //parse.print_labels();
  }  
}

template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::update(
    const MinibatchWords& global_words,
    const boost::shared_ptr<ParsedWeights>& global_gradient,
    const boost::shared_ptr<ParsedWeights>& adagrad,
    bool sentences_only) {
  adagrad->updateSquared(global_words, global_gradient, sentences_only);
  weights->updateAdaGrad(global_words, global_gradient, adagrad, sentences_only);
}

template<class ParseModel, class ParsedWeights, class Metadata>
Real LblDpModel<ParseModel, ParsedWeights, Metadata>::regularize(
    const boost::shared_ptr<ParsedWeights>& global_gradient,
    Real minibatch_factor, bool sentences_only) {
  return weights->regularizerUpdate(global_gradient, minibatch_factor, sentences_only);
}

template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::evaluate() const {
  #pragma omp master
  {
  //read test data 
  boost::shared_ptr<ParsedCorpus> test_corpus = boost::make_shared<ParsedCorpus>(config);
  test_corpus->readFile(config->test_file, dict, true);
  std::cerr << "Done reading test corpus..." << std::endl;
  
  Real log_likelihood = 0;
  evaluate(test_corpus, log_likelihood);
    
  size_t test_size = 3*test_corpus->numTokens(); //approx
  Real test_perplexity = perplexity(log_likelihood, test_size); 
  std::cerr << "Test Perplexity: " << test_perplexity << std::endl;
  }
}

template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::evaluate(
    const boost::shared_ptr<ParsedCorpus>& test_corpus, Real& accumulator) const {
  if (test_corpus != nullptr) {
  #pragma omp master
  {
    vector<int> indices(test_corpus->size());
    iota(indices.begin(), indices.end(), 0);
    
    for (int i_test = 1; i_test <= config->iterations_test; ++i_test) {
    //for (unsigned beam_size: config->beam_sizes) {
      unsigned beam_size = config->beam_sizes[0];
      std::ofstream outs;
      outs.open(config->test_output_file);
      //#pragma omp master
      {
        std::cerr << "parsing with beam size " << beam_size << ", iter " << i_test << ":\n";
        accumulator = 0;
      }

      // Each thread must wait until the perplexity is set to 0.
      // Otherwise, partial results might get overwritten.
      //#pragma omp barrier
      
      auto beam_start = get_time();
      boost::shared_ptr<AccuracyCounts> acc_counts = boost::make_shared<AccuracyCounts>(dict);

      for (int j = 0; j < test_corpus->size(); ++j) {

      //size_t start = 0;
      //while (start < test_corpus->size()) {
      //  size_t end = std::min(start + config->minibatch_size, test_corpus->size());
        //std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
        //minibatch = scatterMinibatch(minibatch);
        Real objective = 0;
      //  for (auto j: minibatch) {
        Parser parse;
          
        if (config->sentence_vector) {
          Real best_sentence_objective = numeric_limits<Real>::infinity();
            //boost::shared_ptr<ParsedWeights> global_gradient =
            //  boost::make_shared<ParsedWeights>(config, metadata, false);
            //boost::shared_ptr<ParsedWeights> adagrad =
            //  boost::make_shared<ParsedWeights>(config, metadata, false);
            //boost::shared_ptr<ParsedWeights> gradient =
            //  boost::make_shared<ParsedWeights>(config, metadata, false);

          weights->resetSentenceVector();
          for (int i = 0; i < i_test; ++i) {
            VectorReal sentence_vector_gradient = VectorReal::Zero(config->representation_size);
            parse = parse_model->evaluateSentence(test_corpus->sentence_at(j), weights, acc_counts, (i == i_test - 1), beam_size);

            if (i < i_test - 1) {
              boost::shared_ptr<ParseDataSet> examples = boost::make_shared<ParseDataSet>();

              parse_model->extractSentence(parse, weights, examples);
              //int num_examples = task_examples->word_example_size() + task_examples->action_example_size();

              Real sentence_objective = 0;
              sentence_vector_gradient = weights->getSentenceVectorGradient(examples, sentence_objective);
              //std::cout << AccuracyCounts::sentence_uas(parse, test_corpus->sentence_at(j)) << " " << parse.weight() << " " <<  sentence_vector_gradient.norm() << std::endl;
              weights->updateSentenceVectorGradient(sentence_vector_gradient);

              //global_gradient->syncUpdate(words, gradient);
              //update(global_words, global_gradient, adagrad);
              //Real minibatch_factor = static_cast<Real>(num_examples) / (3*training_corpus->numTokens());
              //objective = regularize(global_gradient, 1);

              if (sentence_objective <= best_sentence_objective) 
                best_sentence_objective = sentence_objective;
            }
          }
        } else {
          parse = parse_model->evaluateSentence(test_corpus->sentence_at(j), weights, acc_counts, true, beam_size);
        }

        objective += parse.weight();

        //write output to conll-format file
        for (unsigned i = 1; i < parse.size(); ++i) { 
          outs << i << "\t" << dict->lookup(parse.word_at(i)) << "\t_\t_\t" 
               << dict->lookupTag(parse.tag_at(i)) << "\t_\t" << parse.arc_at(i) << "\t"
               << dict->lookupLabel(parse.label_at(i)) << "\t_\t_\n";
        }
        outs << "\n";
      
        accumulator += objective;
      }

        //#pragma omp critical
      //start = end;
      //} 
    
      // Wait for all the threads to compute the perplexity for their slice of
      // test data.
      //#pragma omp barrier

      //#pragma omp master
      //{
      outs.close();
      Real beam_time = get_duration(beam_start, get_time());
      Real sents_per_sec = static_cast<int>(test_corpus->size())/beam_time;
      Real tokens_per_sec = static_cast<int>(test_corpus->numTokens())/beam_time;
      std::cerr << "(" << beam_time << "s, " <<
                 static_cast<int>(sents_per_sec) << " sentences per second, " <<
                 static_cast<int>(tokens_per_sec) << " tokens per second)\n";
      acc_counts->printAccuracy();
      //}
    }
    
    //#pragma omp barrier
    weights->clearCache();
  }
  }
}

template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::evaluate(
    const boost::shared_ptr<ParsedCorpus>& test_corpus, const Time& iteration_start,
    int minibatch_counter, Real& log_likelihood, Real& best_perplexity) const {
  if (test_corpus != nullptr) {
    evaluate(test_corpus, log_likelihood);

    #pragma omp master
    {
      Real test_perplexity = perplexity(log_likelihood, test_corpus->numTokens());
      Real iteration_time = get_duration(iteration_start, get_time());
      std::cerr << "\tMinibatch " << minibatch_counter << ", "
           << "Time: " << get_duration(iteration_start, get_time()) << " seconds, "
           << "  Test Likelihood: " << log_likelihood 
           << "  Test Size: " << test_corpus->numTokens() 
           << "  Test Perplexity: " << test_perplexity << endl;

      if (test_perplexity < best_perplexity) {
        best_perplexity = test_perplexity;
        save();
      }
    }
  } else {
    #pragma omp master
    save();
  }
}

template<class ParseModel, class ParsedWeights, class Metadata>
Real LblDpModel<ParseModel, ParsedWeights, Metadata>::predict(
    int word_id, const vector<int>& context) const {
  return weights->predict(word_id, context);
}

template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::save() const {
  if (config->model_output_file.size()) {
      std::cerr << "Writing model to " << config->model_output_file << "..." << endl;
    ofstream fout(config->model_output_file);
    boost::archive::binary_oarchive oar(fout);
    oar << config;
    oar << dict;
    oar << weights;
    oar << metadata;
    std::cerr << "Done..." << endl;
  }
}

template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::load(const string& filename) {
  if (filename.size() > 0) {
    auto start_time = get_time();
    cerr << "Loading model from " << filename << "..." << endl;
    ifstream fin(filename);
    boost::archive::binary_iarchive iar(fin);
    iar >> config;
    iar >> dict;
    iar >> weights;
    iar >> metadata;
    cerr << "Reading model took " << get_duration(start_time, get_time())
         << " seconds..." << endl;
  }
}

template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::clearCache() {
  weights->clearCache();
}

template<class ParseModel, class ParsedWeights, class Metadata>
bool LblDpModel<ParseModel, ParsedWeights, Metadata>::operator==(
    const LblDpModel<ParseModel, ParsedWeights, Metadata>& other) const {
  return *config == *other.config
      && *metadata == *other.metadata
      && *weights == *other.weights;
}

template class LblDpModel<ArcStandardLabelledParseModel<TaggedParsedFactoredWeights>, TaggedParsedFactoredWeights, TaggedParsedFactoredMetadata>;
//template class LblDpModel<ArcStandardParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>;
template class LblDpModel<ArcStandardLabelledParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>;
//template class LblDpModel<ArcEagerParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>;
template class LblDpModel<ArcEagerLabelledParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>;
//template class LblDpModel<EisnerParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>;

//template class LblDpModel<ArcStandardParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>;
template class LblDpModel<ArcStandardLabelledParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>;
//template class LblDpModel<ArcEagerParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>;
template class LblDpModel<ArcEagerLabelledParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>;
//template class LblDpModel<EisnerParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>;

template class LblDpModel<ArcStandardLabelledParseModel<DiscriminativeWeights>, DiscriminativeWeights, DiscriminativeMetadata>;
} // namespace oxlm

