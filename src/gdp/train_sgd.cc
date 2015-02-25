#include <boost/program_options.hpp>

#include "lbl/metadata.h"
#include "lbl/weights.h"
#include "lbl/factored_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/parsed_factored_weights.h"
#include "utils/git_revision.h"

#include "gdp/lbl_model.h"
#include "gdp/lbl_dp_model.h"

using namespace boost::program_options;
using namespace oxlm;

#define lblOrderAS 7
#define lblOrderASl 10
#define lblOrderASn 13
#define lblOrderASx 13
#define lblOrderASxx 17
#define lblOrderAE 8
#define lblOrderAEl 12
#define lblOrderAEx 14
#define lblOrderE 6

template<class ParseModel, class ParsedWeights, class LblMetadata>
void train_dp(const boost::shared_ptr<ModelConfig>& config) {
  LblDpModel<ParseModel, ParsedWeights, LblMetadata> model(config);

  model.learn();
  if (config->iterations > 1)
    model.evaluate();
}

int main(int argc, char** argv) {
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", value<string>(),
        "Config file specifying additional command line options");

  options_description generic("Allowed options");
  generic.add_options()
    ("training-set,i", 
        value<std::string>()->default_value("english-wsj-stanford-unk/english_wsj_train.conll"),
        "corpus of parsed sentences for training, conll format")
    ("training-set-unsup,u", value<std::string>(),
        "corpus of unparsed sentences for semi-supervised training, conll format")
    ("test-set,t", value<string>(),
        "corpus of test sentences")
    ("test-out-file,o", value<std::string>()->default_value("system.out.conll"),
        "conll output file for system parsing the test set")
    ("iterations", value<int>()->default_value(1),
        "number of passes through the data")
    ("minibatch-size", value<int>()->default_value(10000),
        "number of sentences per minibatch")
    ("minibatch-size-unsup", value<int>()->default_value(1),
        "number of sentences per minibatch, unsupervised training")
    ("order,n", value<int>()->default_value(4),
        "ngram order")
    ("class-factored,f", value<bool>()->default_value(true),
        "Class-factored vocabulary model.")
    ("model-in", value<string>(),
        "Load initial model from this file")
    ("model-out,o", value<string>(),
        "base filename of model output files")
    ("lambda-lbl,r", value<float>()->default_value(7.0),
        "regularisation strength parameter")
    ("representation-size", value<int>()->default_value(100),
        "Width of representation vectors.")
    ("threads", value<int>()->default_value(1),
        "number of worker threads.")
    ("step-size", value<float>()->default_value(0.05),
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("randomise", value<bool>()->default_value(true),
        "Visit the training tokens in random order.")
    ("parser-type", value<std::string>()->default_value("arcstandard"),
        "Parsing strategy.")    
    ("context-type", value<std::string>()->default_value(""),
        "Conditioning context used.")    
    ("labelled-parser", value<bool>()->default_value(false),
        "Predict arc labels.")
    ("lexicalised", value<bool>()->default_value(true),
        "Predict words in addition to POS tags.")
    ("compositional", value<bool>()->default_value(false),
        "Compositional word representations including POS tags and other features.")
    ("semi-supervised", value<bool>()->default_value(false),
        "Use additional, unlabelled training data.")
    ("root-first", value<bool>()->default_value(true),
        "Add root to the beginning (else end) of the sentence.")
    ("complete-parse", value<bool>()->default_value(true),
        "Enforce complete tree-structured parse.")
    ("bootstrap", value<bool>()->default_value(false),
        "Extract training data with beam search.")
    ("max-beam-size", value<int>()->default_value(8),
        "Maximum beam size for decoding (in powers of 2).")
    ("max-beam-increment", value<int>()->default_value(1),
        "Maximum items to add to beam from one state.")
    ("generate-samples", value<int>()->default_value(0),
        "  Maximum items to add to beam from one state.")
    ("direction-det", value<bool>()->default_value(false),
        "Arc direction always deterministic in beam search.")
    ("sum-over-beam", value<bool>()->default_value(false),
        "Sum over likelihoods of identical parses in final beam.")
    ("diagonal-contexts", value<bool>()->default_value(true),
        "Use diagonal context matrices (usually faster).")
    ("activation", value<std::string>()->default_value("linear"),
        "Activation function for to the projection (hidden) layer.")
    ("noise-samples", value<int>()->default_value(0),
        "Number of noise samples for noise contrastive estimation. "
        "If zero, minibatch gradient descent is used instead.")
    ("classes", value<int>()->default_value(100),
        "Number of classes for factored output using frequency binning.")
    ("class-file", value<string>(),
        "File containing word to class mappings in the format "
        "<class> <word> <frequence>.");
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  variables_map vm;
  store(parse_command_line(argc, argv, cmdline_options), vm);
  if (vm.count("config") > 0) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, cmdline_options), vm);
  }

  if (vm.count("help")) {
    std::cerr << cmdline_options << "\n";
    return 1;
  }

  notify(vm);

  boost::shared_ptr<ModelConfig> config = boost::make_shared<ModelConfig>();

  config->training_file = vm["training-set"].as<std::string>();
  if (vm.count("test-set")) {
    config->test_file = vm["test-set"].as<string>();
  }

  config->pyp_model = false;
  config->iterations = vm["iterations"].as<int>();
  config->minibatch_size = vm["minibatch-size"].as<int>();
  config->ngram_order = vm["order"].as<int>();

  if (vm.count("model-in")) {
    config->model_input_file = vm["model-in"].as<string>();
  }

  if (vm.count("model-out")) {
    config->model_output_file = vm["model-out"].as<string>();
    if (GIT_REVISION) {
      config->model_output_file += "." + string(GIT_REVISION);
    }
  }

  config->context_type = vm["context-type"].as<std::string>();
  std::string parser_type_str = vm["parser-type"].as<std::string>();
  if (parser_type_str == "arcstandard") {
    config->parser_type = ParserType::arcstandard; 
    if (config->context_type == "extended")
      config->ngram_order = lblOrderASx;
    else if (config->context_type == "more-extended")
      config->ngram_order = lblOrderASxx;
    else if (config->context_type == "with-ngram")
      config->ngram_order = lblOrderASn;
    else if (config->context_type == "lookahead")
      config->ngram_order = lblOrderASl;
    else
      config->ngram_order = lblOrderAS;
  } else if (parser_type_str == "arceager") {
    config->parser_type = ParserType::arceager;
    if (config->context_type == "extended")
      config->ngram_order = lblOrderAEx;
    else if (config->context_type == "lookahead")
      config->ngram_order = lblOrderAEl;
    else
      config->ngram_order = lblOrderAE;
  } else if (parser_type_str == "eisner") {
    config->parser_type = ParserType::eisner; 
    config->ngram_order = lblOrderE;
  } else {
    config->parser_type = ParserType::ngram; 
  }

  std::string activation_str = vm["activation"].as<std::string>();
  if (activation_str == "linear") 
    config->activation = Activation::linear;
  else if (activation_str == "sigmoid") 
    config->activation = Activation::sigmoid;
  else if (activation_str == "tanh") 
    config->activation = Activation::tanh;
  else if (activation_str == "rectifier") 
    config->activation = Activation::rectifier;
  else 
    config->activation = Activation::linear;

  config->labelled_parser = vm["labelled-parser"].as<bool>();
  config->lexicalised = vm["lexicalised"].as<bool>();
  config->compositional = vm["compositional"].as<bool>();
  config->direction_deterministic = vm["direction-det"].as<bool>();
  config->sum_over_beam = vm["sum-over-beam"].as<bool>();
  config->semi_supervised = vm["semi-supervised"].as<bool>();
  config->root_first = vm["root-first"].as<bool>();
  config->complete_parse = vm["complete-parse"].as<bool>();
  config->bootstrap = vm["bootstrap"].as<bool>();
  config->max_beam_increment = vm["max-beam-increment"].as<int>();
  config->generate_samples = vm["generate-samples"].as<int>();

  config->beam_sizes = {static_cast<unsigned>(vm["max-beam-size"].as<int>())};
  //for (int i = 2; i <= vm["max-beam-size"].as<int>(); i *= 2)
  //  config->beam_sizes.push_back(i);

  config->l2_lbl = vm["lambda-lbl"].as<float>();
  config->representation_size = vm["representation-size"].as<int>();
  config->threads = vm["threads"].as<int>();
  config->step_size = vm["step-size"].as<float>();
  config->randomise = vm["randomise"].as<bool>();
  config->diagonal_contexts = vm["diagonal-contexts"].as<bool>();
  config->factored = vm["class-factored"].as<bool>();

  config->noise_samples = vm["noise-samples"].as<int>();

  config->classes = vm["classes"].as<int>();
  if (vm.count("class-file")) {
    config->class_file = vm["class-file"].as<string>();
  }

  std::cerr << "################################" << std::endl;
  if (strlen(GIT_REVISION) > 0) {
    std::cerr << "# Git revision: " << GIT_REVISION << std::endl;
  }
  std::cerr << "# Config Summary" << std::endl;
  std::cerr << "# order = " << config->ngram_order << std::endl;
  std::cerr << "# representation_size = " << config->representation_size << std::endl;
  std::cerr << "# class factored = " << config->factored << std::endl;
  std::cerr << "# compositional = " << config->compositional << std::endl;
  std::cerr << "# parser type = " << parser_type_str << std::endl;
  std::cerr << "# context type = " << config->context_type << std::endl;
  std::cerr << "# labelled parser = " << config->labelled_parser << std::endl;
  std::cerr << "# lexicalised parser = " << config->lexicalised << std::endl;
  std::cerr << "# root first = " << config->root_first << std::endl;
  std::cerr << "# bootstrap = " << config->bootstrap << std::endl;
  std::cerr << "# complete parse = " << config->complete_parse << std::endl;
  std::cerr << "# direction deterministic = " << config->direction_deterministic << std::endl;
  std::cerr << "# sum over beam = " << config->sum_over_beam << std::endl;
  std::cerr << "# max beam size = " << config->beam_sizes.back() << std::endl;
  if (config->model_input_file.size()) {
    std::cerr << "# model-in = " << config->model_output_file << std::endl;
  }
  if (config->model_output_file.size()) {
    std::cerr << "# model-out = " << config->model_output_file << std::endl;
  }
  std::cerr << "# supervised training data = " << config->training_file << std::endl;
  std::cerr << "# unsupervised training data = " << config->training_file_unsup << std::endl;
  std::cerr << "# minibatch size = " << config->minibatch_size << std::endl;
  std::cerr << "# lambda = " << config->l2_lbl << std::endl;
  std::cerr << "# step size = " << config->step_size << std::endl;
  std::cerr << "# iterations = " << config->iterations << std::endl;
  std::cerr << "# threads = " << config->threads << std::endl;
  std::cerr << "# randomise = " << config->randomise << std::endl;
  std::cerr << "# diagonal contexts = " << config->diagonal_contexts << std::endl;
  std::cerr << "# noise samples = " << config->noise_samples << std::endl;
  std::cerr << "################################" << std::endl;

  if (config->parser_type == ParserType::ngram) {
    if (config->factored) {
      if (config->model_input_file.size() == 0) {
        LblModel<FactoredWeights, FactoredWeights, FactoredMetadata> model(config);
        model.learn();
      } else {
        LblModel<FactoredWeights, FactoredWeights, FactoredMetadata> model;
        model.load(config->model_input_file);
        boost::shared_ptr<ModelConfig> model_config = model.getConfig();
        model_config->model_input_file = config->model_input_file;
        assert(*config == *model_config);
        model.learn();
      }
    } else {
      LblModel<Weights, Weights, Metadata> model(config);
      model.learn();
    }
  } else if (config->lexicalised) {
    if (config->parser_type == ParserType::arcstandard) {
      train_dp<ArcStandardLabelledParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>(config);
    } else if (config->parser_type == ParserType::arceager) {
      train_dp<ArcEagerLabelledParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>(config);
    } else {
      //train_dp<EisnerParseModel<ParsedFactoredWeights>, ParsedFactoredWeights, ParsedFactoredMetadata>(config);
    }
  } else {
  if (config->parser_type == ParserType::arcstandard) {
      train_dp<ArcStandardLabelledParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>(config);
    } else if (config->parser_type == ParserType::arceager) {
      train_dp<ArcEagerLabelledParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>(config);
    } else {
      //train_dp<EisnerParseModel<ParsedWeights>, ParsedWeights, ParsedMetadata>(config);
    }
  }

  return 0;
}
