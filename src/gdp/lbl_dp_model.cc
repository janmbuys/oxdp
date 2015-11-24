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

template <class ParseModel, class ParsedWeights, class Metadata>
LblDpModel<ParseModel, ParsedWeights, Metadata>::LblDpModel() {
  dict = boost::make_shared<Dict>(true);
}

template <class ParseModel, class ParsedWeights, class Metadata>
LblDpModel<ParseModel, ParsedWeights, Metadata>::LblDpModel(
    const boost::shared_ptr<ModelConfig>& config)
    : config(config) {
  dict = boost::make_shared<Dict>(true);
  parse_model = boost::make_shared<ParseModel>(config);

  metadata = boost::make_shared<Metadata>(config, dict);
  srand(1);
}

template <class ParseModel, class ParsedWeights, class Metadata>
boost::shared_ptr<Dict>
LblDpModel<ParseModel, ParsedWeights, Metadata>::getDict() const {
  return dict;
}

template <class ParseModel, class ParsedWeights, class Metadata>
boost::shared_ptr<ModelConfig>
LblDpModel<ParseModel, ParsedWeights, Metadata>::getConfig() const {
  return config;
}

template <class ParseModel, class ParsedWeights, class Metadata>
MatrixReal LblDpModel<ParseModel, ParsedWeights, Metadata>::getWordVectors()
    const {
  return weights->getWordVectors();
}

template <class ParseModel, class ParsedWeights, class Metadata>
MatrixReal LblDpModel<ParseModel, ParsedWeights, Metadata>::getFeatureVectors()
    const {
  return weights->getFeatureVectors();
}

template <class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::learn() {
  MT19937 eng;
  boost::shared_ptr<ParsedCorpus> training_corpus =
      boost::make_shared<ParsedCorpus>(config);
  boost::shared_ptr<ParsedCorpus> unsup_training_corpus =
      boost::make_shared<ParsedCorpus>(config);
  training_corpus->readFile(config->training_file, dict, false);
  std::cerr << "Done reading training corpus..." << std::endl;
  std::cerr << "Corpus size: " << training_corpus->size() << " sentences"
            << std::endl;

  if (config->semi_supervised && config->training_file_unsup.size()) {
    std::cerr << "Reading unsupervised training corpus...\n";
    unsup_training_corpus->readTxtFile(config->training_file_unsup, dict,
                                       false);
    std::cerr << "Corpus size: " << unsup_training_corpus->size()
              << " sentences" << std::endl;

  } else {
    std::cerr << "No unsupervised training corpus.\n";
  }

  // Copy tag feature map.
  for (unsigned i = 0; i < dict->tag_size(); ++i) {
    config->tag_to_feature.push_back(dict->tagToFeature(i));
  }

  // Add transitions as features.
  dict->convertFeature("ACTION_NULL", false);
  config->action_feature_index = dict->feature_size();
  for (unsigned i = 0; i < config->numActions(); ++i) {
    dict->convertFeature("ACTION_" + std::to_string(i), false);
  }

  // Add labels to feature vocab.
  if (config->label_features) {
    config->label_feature_index = dict->feature_size();
    for (unsigned i = 0; i < dict->label_size(); ++i) {
      dict->convertFeature("LABEL_" + dict->lookupLabel(i), false);
    }
  }

  if (config->distance_features) {
    config->distance_feature_index = dict->feature_size();
    size_t range = config->distance_range;
    for (size_t i = 0; i < range; ++i) {
      dict->convertFeature("LV_" + std::to_string(i), false);
    }
    for (size_t i = 0; i < range; ++i) {
      dict->convertFeature("RV_" + std::to_string(i), false);
    }
    for (size_t i = 1; i <= range; ++i) {
      dict->convertFeature("BufDist_" + std::to_string(i), false);
    }
    for (size_t i = 1; i <= range; ++i) {
      dict->convertFeature("HeadDist_" + std::to_string(i), false);
    }
  }

  // Update vocabulary sizes.
  config->num_features = dict->feature_size();
  config->num_train_sentences = training_corpus->size();  

  // Copy word feature map.
  for (unsigned i = 0; i < dict->size(); ++i) {
    config->addWordFeatures(dict->getWordFeatures(i));
  }

  std::cerr << dict->size() << " word types, " << dict->feature_size()
            << " features, " << dict->label_size() << " labels)\n";

  boost::shared_ptr<ParsedCorpus> test_corpus;
  if (config->test_file.size()) {
    test_corpus = boost::make_shared<ParsedCorpus>(config);
    test_corpus->readFile(config->test_file, dict, true);

    std::cerr << "Done reading test corpus..." << endl;
  }

  std::cerr << "Reading test corpus 2...\n";
  boost::shared_ptr<ParsedCorpus> test_corpus2 =
      boost::make_shared<ParsedCorpus>(config);
  if (config->test_file2.size()) {
    test_corpus2->readFile(config->test_file2, dict, true);
    std::cerr << "Done reading test corpus 2..." << std::endl;
    std::cerr << "Corpus size: " << test_corpus2->size() << " sentences,\t"
              << test_corpus2->numTokens() << " tokens\n";
  }

  std::cerr << "Reading test corpus unsup...\n";
  boost::shared_ptr<ParsedCorpus> test_corpus_unsup =
      boost::make_shared<ParsedCorpus>(config);
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
  Real global_objective = 0;

  boost::shared_ptr<ParsedWeights> global_gradient =
      boost::make_shared<ParsedWeights>(config, metadata, false);
  boost::shared_ptr<ParsedWeights> adagrad =
      boost::make_shared<ParsedWeights>(config, metadata, false);
  MinibatchWords global_words;

  int shared_index = 0;
  // For no particular reason. It just looks like this works best.
  int task_size =
      sqrt(config->minibatch_size) / 4;  // to imitate word-level behaviour

  omp_set_num_threads(config->threads);
  #pragma omp parallel
  {
    int minibatch_counter = 1;
    int minibatch_size = config->minibatch_size;
    boost::shared_ptr<ParsedWeights> gradient =
        boost::make_shared<ParsedWeights>(config, metadata, false);

    for (int iter = 0; (iter < config->iterations) && !stop_training; ++iter) {
      Real best_iter_objective = numeric_limits<Real>::infinity();
      auto iteration_start = get_time();

      #pragma omp master
      {
        std::cerr << "Training iteration " << iter << std::endl;
        if (config->randomise) {
          random_shuffle(indices.begin(), indices.end());
        }
        global_objective = 0;
      }

      // Wait until the master thread finishes shuffling the indices.
      #pragma omp barrier

      size_t start = 0;
      while (start < training_corpus->size()) {
        size_t end = min(training_corpus->size(), start + minibatch_size);

        vector<int> minibatch(indices.begin() + start,
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
            vector<int> task(minibatch.begin() + task_start,
                             minibatch.begin() + task_end);
            // Collect the training examples for the minibatch.
            boost::shared_ptr<ParseDataSet> task_examples =
                boost::make_shared<ParseDataSet>();

            for (int j : task) {
              if ((!config->bootstrap && (config->bootstrap_iter == 0)) ||
                  (iter < config->bootstrap_iter)) {
                parse_model->extractSentence(training_corpus->sentence_at(j),
                                             task_examples);
              } else if (config->bootstrap) {
                parse_model->extractSentence(training_corpus->sentence_at(j),
                                             weights, task_examples);
              } else {
                parse_model->extractSentenceUnsupervised(
                    training_corpus->sentence_at(j), weights, task_examples);
              }
            }

            num_examples += task_examples->word_example_size() +
                            task_examples->action_example_size();
            if (config->predict_pos) {
              num_examples += task_examples->tag_example_size();
            }

            if (config->noise_samples > 0) {
              weights->estimateGradient(task_examples, gradient, objective,
                                        words);
            } else {
              weights->getGradient(task_examples, gradient, objective, words);
            }
          } else {
            break;
          }
        }

        global_gradient->syncUpdate(words, gradient);
        
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
        update(global_words, global_gradient, adagrad);

        // Wait for all threads to finish making the model gradient update.
        #pragma omp barrier
        Real minibatch_factor = static_cast<Real>(num_examples) /
                                (3 * training_corpus->numTokens());
        if (config->predict_pos)
          minibatch_factor = static_cast<Real>(num_examples) /
                             (4 * training_corpus->numTokens());

        objective = regularize(global_words, global_gradient, minibatch_factor);
        #pragma omp critical
        global_objective += objective;

        // Clear gradients.
        gradient->clear(words, false);
        global_gradient->clear(global_words, true);

        // Wait the regularization update to finish and make sure the global
        // words are reset only after the global gradient is fully cleared.
        #pragma omp barrier
        ++minibatch_counter;
        start = end;
      }

      #pragma omp master
      {
        Real iteration_time = get_duration(iteration_start, get_time());
        std::cerr << "Iteration: " << iter << ", "
                  << "Time: " << iteration_time << " seconds, "
                  << "  Likelihood: " << global_objective
                  << "  Size: " << training_corpus->numTokens()
                  << "  Perplexity: "
                  << perplexity(global_objective, training_corpus->numTokens())
                  << "  Objective: "
                  << global_objective / training_corpus->numTokens() << endl
                  << endl;

        if (global_objective <= best_global_objective) {
          best_global_objective = global_objective;
          objective_improved = true;
        } else {
          if (!objective_improved) stop_training = true;
          objective_improved = false;
        }
      }
    
      evaluate(test_corpus, iteration_start, minibatch_counter, test_objective,
               best_perplexity);
      if (config->semi_supervised) {
        evaluate(test_corpus_unsup, iteration_start, minibatch_counter,
                 test_objective_unsup, best_perplexity_unsup);
      } else if (config->test_file2.size()) {
        evaluate(test_corpus2, iteration_start, minibatch_counter,
                 test_objective2, best_perplexity2);
      }

      #pragma omp master
      std::cerr << "Done with training iteration " << iter << std::endl;
    }

    if (config->semi_supervised) {
      #pragma omp master
      std::cerr << "Parsing unlabelled data" << std::endl;

      boost::shared_ptr<AccuracyCounts> temp_acc_counts =
          boost::make_shared<AccuracyCounts>(dict);
      for (unsigned j = 0; j < unsup_training_corpus->size(); ++j) {
        Parser parse = parse_model->evaluateSentence(
            unsup_training_corpus->sentence_at(j), weights, temp_acc_counts,
            false, config->num_particles);
        unsup_training_corpus->set_arcs_at(j, parse);
        unsup_training_corpus->set_labels_at(j, parse);
      }

      #pragma omp master
      Real best_global_objective = numeric_limits<Real>::infinity();
    }

    for (int iter = 0;
         (iter < config->iterations_unsup) && config->semi_supervised; ++iter) {
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
            vector<int> task(minibatch.begin() + task_start,
                             minibatch.begin() + task_end);
            // Collect the training examples for the minibatch.
            boost::shared_ptr<ParseDataSet> task_examples =
                boost::make_shared<ParseDataSet>();

            //TODO make inference rather than pre-parsing an option
            for (int j : task) {
              parse_model->extractSentence(
                  unsup_training_corpus->sentence_at(j), task_examples);
              // parse_model->extractSentenceUnsupervised(unsup_training_corpus->sentence_at(j),
              //   weights, task_examples);
            }
            num_examples += task_examples->word_example_size() +
                            task_examples->action_example_size();
            if (config->predict_pos) {
              num_examples += task_examples->tag_example_size();
            }

            if (config->noise_samples > 0) {
              weights->estimateGradient(task_examples, gradient, objective,
                                        words);
            } else {
              weights->getGradient(task_examples, gradient, objective, words);
            }
          } else {
            break;
          }
        }

        global_gradient->syncUpdate(words, gradient);
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
        update(global_words, global_gradient, adagrad);

        // Wait for all threads to finish making the model gradient update.
        #pragma omp barrier
        Real minibatch_factor = static_cast<Real>(num_examples) /
                                (3 * unsup_training_corpus->numTokens());
        if (config->predict_pos)
          minibatch_factor = static_cast<Real>(num_examples) /
                             (4 * unsup_training_corpus->numTokens());

        objective = regularize(global_words, global_gradient, minibatch_factor);
        #pragma omp critical
        global_objective += objective;

        // Clear gradients.
        gradient->clear(words, false);
        global_gradient->clear(global_words, true);

        // Wait the regularization update to finish and make sure the global
        // words are reset only after the global gradient is fully cleared.
        #pragma omp barrier
        if ((iter == 0) && (minibatch_counter % 100000 == 0)) {
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
                  << "  Perplexity: "
                  << perplexity(global_objective,
                                unsup_training_corpus->numTokens())
                  << "  Objective: "
                  << global_objective / unsup_training_corpus->numTokens()
                  << endl << endl;

        if (global_objective <= best_global_objective) {
          best_global_objective = global_objective;
        }  
      }
      evaluate(test_corpus, iteration_start, minibatch_counter, test_objective,
               best_perplexity);
      evaluate(test_corpus_unsup, iteration_start, minibatch_counter,
               test_objective_unsup, best_perplexity_unsup);
    }
  }

  std::cerr << "Overall minimum perplexity: " << best_perplexity << endl;

  // generate from model
  std::vector<boost::shared_ptr<Parser>> generated_list;
  for (int i = 0; i < config->generate_samples; ++i) {
    generated_list.push_back(boost::make_shared<Parser>(
        parse_model->generateSentence(weights, eng)));
  }

  std::sort(generated_list.begin(), generated_list.end(), Parser::cmp_weights);
  for (int i = 0; i < config->generate_samples; ++i) {
    generated_list[i]->print_sentence(dict);
  }
}

template <class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::update(
    const MinibatchWords& global_words,
    const boost::shared_ptr<ParsedWeights>& global_gradient,
    const boost::shared_ptr<ParsedWeights>& adagrad) {
  adagrad->updateSquared(global_words, global_gradient);
  weights->updateAdaGrad(global_words, global_gradient, adagrad);
}

template <class ParseModel, class ParsedWeights, class Metadata>
Real LblDpModel<ParseModel, ParsedWeights, Metadata>::regularize(
    const MinibatchWords& global_words,
    const boost::shared_ptr<ParsedWeights>& global_gradient,
    Real minibatch_factor) {
  return weights->regularizerUpdate(global_words, global_gradient,
                                    minibatch_factor);
}

template <class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::evaluate() {
  #pragma omp master
  {
    boost::shared_ptr<ParsedCorpus> test_corpus =
        boost::make_shared<ParsedCorpus>(config);
    test_corpus->readFile(config->test_file, dict, true);
    std::cerr << "Done reading test corpus..." << std::endl;

    Real log_likelihood = 0;
    evaluate(test_corpus, log_likelihood);

    //TODO Make sure that test_size is correct for official perplexity calculation.
    size_t test_size = 3 * test_corpus->numTokens();
    Real test_perplexity = perplexity(log_likelihood, test_size);
    std::cerr << "Test Perplexity: " << test_perplexity << std::endl;
  }
}

template <class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::evaluate(
    const boost::shared_ptr<ParsedCorpus>& test_corpus, Real& accumulator) {
  if (test_corpus == nullptr) return;
  
  // Evaluation is done in a single thread.
  #pragma omp master
  {
    vector<int> indices(test_corpus->size());
    iota(indices.begin(), indices.end(), 0);

    for (unsigned beam_size: config->beam_sizes) {
      std::ofstream outs;
      outs.open(config->test_output_file);
      
      std::cerr << "parsing with beam size " << beam_size << ":\n";
      accumulator = 0;
      
      auto beam_start = get_time();
      boost::shared_ptr<AccuracyCounts> acc_counts =
          boost::make_shared<AccuracyCounts>(dict);

      for (int j = 0; j < test_corpus->size(); ++j) {
        Real objective = 0;
        Parser parse;

        parse = parse_model->evaluateSentence(
            test_corpus->sentence_at(j), weights, acc_counts, true, beam_size);

        objective += parse.weight();

        // Write output to conll-format file.
        for (unsigned i = 1; i < parse.size(); ++i) {
          outs << i << "\t" << dict->lookup(parse.word_at(i)) << "\t_\t_\t"
               << dict->lookupTag(parse.tag_at(i)) << "\t_\t" << parse.arc_at(i)
               << "\t" << dict->lookupLabel(parse.label_at(i)) << "\t_\t_\n";
        }
        outs << "\n";

        accumulator += objective;
      }

      outs.close();
      Real beam_time = get_duration(beam_start, get_time());
      Real sents_per_sec = static_cast<int>(test_corpus->size()) / beam_time;
      Real tokens_per_sec =
          static_cast<int>(test_corpus->numTokens()) / beam_time;
      std::cerr << "(" << beam_time << "s, " << static_cast<int>(sents_per_sec)
                << " sentences per second, " << static_cast<int>(tokens_per_sec)
                << " tokens per second)\n";
      acc_counts->printAccuracy();

      weights->clearCache();
    }
  }
}

template <class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::evaluate(
    const boost::shared_ptr<ParsedCorpus>& test_corpus,
    const Time& iteration_start, int minibatch_counter, Real& log_likelihood,
    Real& best_perplexity) {
  if (test_corpus != nullptr) {
    evaluate(test_corpus, log_likelihood);

    #pragma omp master
    {
      Real test_perplexity =
          perplexity(log_likelihood, test_corpus->numTokens());
      Real iteration_time = get_duration(iteration_start, get_time());
      std::cerr << "\tMinibatch " << minibatch_counter << ", "
                << "Time: " << get_duration(iteration_start, get_time())
                << " seconds, "
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

template <class ParseModel, class ParsedWeights, class Metadata>
Real LblDpModel<ParseModel, ParsedWeights, Metadata>::predict(
    int word_id, const vector<int>& context) const {
  return weights->predict(word_id, context);
}

template <class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::save() const {
  if (config->model_output_file.size()) {
    std::cerr << "Writing model to " << config->model_output_file << "..."
              << endl;
    ofstream fout(config->model_output_file);
    boost::archive::binary_oarchive oar(fout);
    oar << config;
    oar << dict;
    oar << weights;
    oar << metadata;
    std::cerr << "Done..." << endl;
  }
}

template <class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::load(
    const string& filename) {
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

template <class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::clearCache() {
  weights->clearCache();
}

template <class ParseModel, class ParsedWeights, class Metadata>
bool LblDpModel<ParseModel, ParsedWeights, Metadata>::operator==(
    const LblDpModel<ParseModel, ParsedWeights, Metadata>& other) const {
  return *config == *other.config && *metadata == *other.metadata &&
         *weights == *other.weights;
}

template class LblDpModel<
    ArcStandardLabelledParseModel<TaggedParsedFactoredWeights>,
    TaggedParsedFactoredWeights, TaggedParsedFactoredMetadata>;
template class LblDpModel<ArcStandardLabelledParseModel<ParsedFactoredWeights>,
                          ParsedFactoredWeights, ParsedFactoredMetadata>;

}  // namespace oxlm

