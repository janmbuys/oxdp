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
  // Initialize the dictionary now, if it hasn't been initialized when the
  // vocabulary was partitioned in classes. - allways initialize, else miss tags etc
  //bool immutable_dict = config->classes > 0 || config->class_file.size();
  boost::shared_ptr<ParsedCorpus> training_corpus = boost::make_shared<ParsedCorpus>(config);
  training_corpus->readFile(config->training_file, dict, false);
  std::cerr << "Done reading training corpus..." << endl;

  std::cerr << "Corpus size: " << training_corpus->size() << " sentences\t (" 
            << dict->size() << " word types, " << dict->tag_size() << " tags, "  
	    << dict->label_size() << " labels)\n";  

  boost::shared_ptr<ParsedCorpus> test_corpus; 
  if (config->test_file.size()) {
    test_corpus = boost::make_shared<ParsedCorpus>(config);
    test_corpus->readFile(config->test_file, dict, true);
    std::cerr << "Done reading test corpus..." << endl;
  }

  if (config->model_input_file.size() == 0) {
    metadata->initialize(training_corpus);
    weights = boost::make_shared<ParsedWeights>(config, metadata, true);
    //std::cout << "initialized weights" << std::endl;
  } else {
    Real log_likelihood = 0;
    evaluate(test_corpus, log_likelihood);
    std::cerr << "Initial perplexity: "
         << perplexity(log_likelihood, test_corpus->numTokens()) << endl;
  }

  vector<int> indices(training_corpus->size());
  iota(indices.begin(), indices.end(), 0);

  Real best_perplexity = numeric_limits<Real>::infinity();
  Real global_objective = 0, test_objective = 0;
  boost::shared_ptr<ParsedWeights> global_gradient =
      boost::make_shared<ParsedWeights>(config, metadata, false);
  boost::shared_ptr<ParsedWeights> adagrad =
      boost::make_shared<ParsedWeights>(config, metadata, false);
  MinibatchWords global_words;

  int shared_index = 0;
  // For no particular reason. It just looks like this works best.
  int task_size = sqrt(config->minibatch_size) / 4; //to imitate word-level behaviour
  //std::cout << "initialized gradients" << std::endl;
    
  omp_set_num_threads(config->threads);
  #pragma omp parallel
  {
    int minibatch_counter = 1;
    int minibatch_size = config->minibatch_size;
    boost::shared_ptr<ParsedWeights> gradient =
        boost::make_shared<ParsedWeights>(config, metadata, false);

    for (int iter = 0; iter < config->iterations; ++iter) {
      auto iteration_start = get_time();
      //std::cout << "training size: " << training_corpus->size() << std::endl;

      #pragma omp master
      {
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

        vector<int> minibatch(
            indices.begin() + start,
            min(indices.begin() + end, indices.end()));
       //std::cout << "minibatch: " << minibatch.size() << " sentences\n";

        //global_gradient->init(training_corpus, minibatch);
        // Reset the set of minibatch words shared across all threads.
        #pragma omp master
        {
          global_words = MinibatchWords();
          shared_index = 0;
        }

        //gradient->init(training_corpus, minibatch);

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
            //std::cerr << " task " << task_start << std::endl;
            boost::shared_ptr<ParseDataSet> task_examples = boost::make_shared<ParseDataSet>();
            
            // #pragma omp critical
            for (int j: task) 
              parse_model->extractSentence(training_corpus->sentence_at(j), task_examples);
            //std::cout << " (" << task_examples->size() << ") ";
            num_examples += task_examples->word_example_size() + task_examples->action_example_size();

            if (config->noise_samples > 0) {
              weights->estimateGradient(
                  task_examples, gradient, objective, words);
            } else {
              weights->getGradient(
                  task_examples, gradient, objective, words);
              //std::cout << minibatch_counter << " " << minibatch.size() << " " << task_start << std::endl;               
              //if (!weights->checkGradient(task_examples, global_gradient, EPS))
              //  std::cout << "gradient check failed" << std::endl;
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
        Real minibatch_factor =
            static_cast<Real>(num_examples) / (3*training_corpus->numTokens());
        //for now, weight in terms of number of words predicted
        //should actually be total number of predictions, but if the ratio is
        //the same, it should be fine
        
        //std::cout << "\n" << num_examples << " examples " 
        //    << minibatch_factor << " minibatch factor" << std::endl;
        objective = regularize(global_gradient, minibatch_factor);
        #pragma omp critical
        global_objective += objective;

        // Clear gradients.
        gradient->clear(words, false);
        global_gradient->clear(global_words, true);

        // Wait the regularization update to finish and make sure the global
        // words are reset only after the global gradient is fully cleared.
        #pragma omp barrier

        //don't evaluate before end of iteration
        /* if ((minibatch_counter % 100 == 0 && minibatch_counter <= 1000) ||
            minibatch_counter % 1000 == 0) {
          evaluate(test_corpus, iteration_start, minibatch_counter,
                   test_objective, best_perplexity);
        } */

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
             << "  Perplexity: " << perplexity(global_objective, training_corpus->numTokens())
             << "  Objective: " << global_objective / training_corpus->numTokens()
             << endl << endl;
      
        if (iter%5 == 0)
          evaluate(test_corpus, iteration_start, minibatch_counter,
               test_objective, best_perplexity);
      }
    }
  }

  std::cerr << "Overall minimum perplexity: " << best_perplexity << endl;
}

template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::update(
    const MinibatchWords& global_words,
    const boost::shared_ptr<ParsedWeights>& global_gradient,
    const boost::shared_ptr<ParsedWeights>& adagrad) {
  adagrad->updateSquared(global_words, global_gradient);
  weights->updateAdaGrad(global_words, global_gradient, adagrad);
}

template<class ParseModel, class ParsedWeights, class Metadata>
Real LblDpModel<ParseModel, ParsedWeights, Metadata>::regularize(
    const boost::shared_ptr<ParsedWeights>& global_gradient,
    Real minibatch_factor) {
  return weights->regularizerUpdate(global_gradient, minibatch_factor);
}

template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::evaluate() const {
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

//not sure if I should thread this: disable for now
template<class ParseModel, class ParsedWeights, class Metadata>
void LblDpModel<ParseModel, ParsedWeights, Metadata>::evaluate(
    const boost::shared_ptr<ParsedCorpus>& test_corpus, Real& accumulator) const {
  if (test_corpus != nullptr) {
        vector<int> indices(test_corpus->size());
    iota(indices.begin(), indices.end(), 0);
    
    for (unsigned beam_size: config->beam_sizes) {
      std::ofstream outs;
      outs.open("system.out" + std::to_string(beam_size));
      #pragma omp master
      {
        std::cerr << "parsing with beam size " << beam_size << ":\n";
        accumulator = 0;
      }

      // Each thread must wait until the perplexity is set to 0.
      // Otherwise, partial results might get overwritten.
      #pragma omp barrier
      
      auto beam_start = get_time();
      boost::shared_ptr<AccuracyCounts> acc_counts = boost::make_shared<AccuracyCounts>(dict);

      size_t start = 0;
      while (start < test_corpus->size()) {
        size_t end = std::min(start + config->minibatch_size, test_corpus->size());

        std::vector<int> minibatch(indices.begin() + start, indices.begin() + end);
        minibatch = scatterMinibatch(minibatch);

        Real objective = 0;
        for (auto j: minibatch) {
          Parser parse = parse_model->evaluateSentence(test_corpus->sentence_at(j), weights, acc_counts, beam_size);
          objective += parse.weight();
          //parse.print_arcs();

          //write output to conll-format file: may need a lock
          for (unsigned i = 1; i < parse.size(); ++i) { 
            outs << i << "\t" << dict->lookup(parse.word_at(i)) << "\t_\t_\t" 
                 << dict->lookupTag(parse.tag_at(i)) << "\t_\t" << parse.arc_at(i) << "\t"
                 << dict->lookupLabel(parse.label_at(i)) << "\t_\t_\n";
          }
          outs << "\n";
        }

        #pragma omp critical
        accumulator += objective;
        start = end;
      } 
    
      // Wait for all the threads to compute the perplexity for their slice of
      // test data.
      // do we need both a barrier and a master?
      #pragma omp barrier

//      #pragma omp master
      {
        outs.close();
        Real beam_time = get_duration(beam_start, get_time());
        Real sents_per_sec = static_cast<int>(test_corpus->size())/beam_time;
        std::cerr << "(" << static_cast<int>(sents_per_sec) << " sentences per second)\n"; 
        acc_counts->printAccuracy();
      }
    }
    
    #pragma omp barrier
    weights->clearCache();
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

//template class LblDpModel<ArcStandardParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>;
template class LblDpModel<ArcStandardLabelledParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>;
//template class LblDpModel<ArcEagerParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>;
template class LblDpModel<ArcEagerLabelledParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>;
template class LblDpModel<EisnerParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>;

//template class LblDpModel<ArcStandardParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>;
template class LblDpModel<ArcStandardLabelledParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>;
//template class LblDpModel<ArcEagerParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>;
template class LblDpModel<ArcEagerLabelledParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>;
template class LblDpModel<EisnerParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>;

} // namespace oxlm

