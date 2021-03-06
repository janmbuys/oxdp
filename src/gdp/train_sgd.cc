#include <boost/program_options.hpp>

#include "lbl/metadata.h"
#include "lbl/weights.h"
#include "lbl/factored_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/parsed_factored_weights.h"
#include "lbl/tagged_parsed_factored_weights.h"
#include "utils/git_revision.h"

#include "gdp/lbl_model.h"
#include "gdp/lbl_dp_model.h"

using namespace boost::program_options;
using namespace oxlm;

// TODO Remove irrelevant ones.
#define lblOrderASa 12
#define lblOrderAS 7
#define lblOrderAS3 10
#define lblOrderASl 10
#define lblOrderASn 13
#define lblOrderASxn 16
#define lblOrderASxxn 21
#define lblOrderASxl 16
#define lblOrderASx 13
#define lblOrderASx3 18
#define lblOrderASxx 17
#define lblOrderASxx3 24

template <class ParseModel, class ParsedWeights, class LblMetadata>
void train_dp(const boost::shared_ptr<ModelConfig>& config) {
  if (config->model_input_file.size() == 0) {
    LblDpModel<ParseModel, ParsedWeights, LblMetadata> model(config);
    model.learn();
  } else {
    LblDpModel<ParseModel, ParsedWeights, LblMetadata> model;
    model.load(config->model_input_file);
    boost::shared_ptr<ModelConfig> model_config = model.getConfig();
    model_config->model_input_file = config->model_input_file;
    model_config->context_type = config->context_type;
    assert(*config == *model_config);
    model.learn();
  }
}

int main(int argc, char** argv) {
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()("help,h", "print help message")(
      "config,c", value<string>(),
      "Config file specifying additional command line options");

  //TODO Check default values.
  options_description generic("Allowed options");
  generic.add_options()
     ("training-set,i", value<std::string>()->default_value(""),
        "corpus of parsed sentences for training, conll format")
     ("training-set-unsup,u", value<std::string>()->default_value(""),
        "corpus of unparsed sentences for semi-supervised training, conll "
        "format")
     ("test-set,t", value<string>()->default_value(""), "corpus of test sentences")
     ("test-set2", value<string>()->default_value(""), "corpus of test sentences")
     ("test-set-unsup", value<std::string>()->default_value(""),
        "corpus of test sentences to be evaluated at each iteration")
     ("test-out-file",
        value<std::string>()->default_value("system.out.conll"),
        "conll output file for system parsing the test set")
     ("iterations", value<int>()->default_value(10),
      "number of passes through the data")
     ("iterations-unsup", value<int>()->default_value(1),
      "number of passes through the unlabelled data")
     ("minibatch-size", value<int>()->default_value(128),
      "number of sentences per minibatch")
     ("minibatch-size-unsup", value<int>()->default_value(128),
      "number of sentences per minibatch, unsupervised training")
     ("order,n", value<int>()->default_value(5), "ngram order")
     ("class-factored,f", value<bool>()->default_value(true),
        "Class-factored vocabulary model.")
     ("model-in", value<string>()->default_value(""), "Load initial model from this file")
     ("model-out", value<string>()->default_value(""), "base filename of model output files")
     ("lambda-lbl,r", value<float>()->default_value(10.0),
        "regularisation strength parameter")
     ("representation-size", value<int>()->default_value(256),
                                           "Width of representation vectors.")
     ("threads", value<int>()->default_value(1), "number of worker threads.")
     ("step-size", value<float>()->default_value(0.05),
        "SGD batch stepsize, it is normalised by the number of minibatches.")
     ("randomise", value<bool>()->default_value(true),
        "Visit the training tokens in random order.")
     ("parser-type", value<std::string>()->default_value("arcstandard"),
        "Parsing strategy.")
     ("context-type", value<std::string>()->default_value("more-extended"),
                           "Conditioning context used.")
     ("labelled-parser", value<bool>()->default_value(true),
        "Predict arc labels.")
     ("predict-pos", value<bool>()->default_value(true),
                             "Predict POS in model.")
     ("sample-decoding", value<bool>()->default_value(false),
                             "Particle filter sampling during decoding.")
     ("tag-pos", value<bool>()->default_value(false),
        "Tag POS during decoding.")
     ("lexicalised", value<bool>()->default_value(true),
                                  "Predict words in addition to POS tags.")
     ("label-features", value<bool>()->default_value(true),
        "Include arc labels as input feature.")
     ("morph-features", value<bool>()->default_value(false),
        "Include conll morphological features.")
     ("distance-features", value<bool>()->default_value(false),
        "Include distance-based input features.")
     ("semi-supervised", value<bool>()->default_value(false),
        "Use additional, unlabelled training data.")
     ("restricted-semi-supervised", value<bool>()->default_value(false),
        "Don't perform full inference over unlabelled data.")
     ("root-first", value<bool>()->default_value(true),
        "Add root to the beginning (else end) of the sentence.")
     ("stack-context-size", value<int>()->default_value(2),
        "Stack elements context size.")
     ("action-context-size", value<int>()->default_value(0),
        "Action history elements context size.")
     ("child-context-level", value<int>()->default_value(0),
        "Stack elements context size.")("output-context-size",
                                        value<int>()->default_value(1),
                                        "Output sentence context size.")
     ("input-window-size", value<int>()->default_value(1),
        "Input sentence context window size.")
     ("complete-parse", value<bool>()->default_value(true),
        "Enforce complete tree-structured parse.")
     ("bootstrap", value<bool>()->default_value(false),
        "Extract training data with beam search.")
     ("bootstrap-iter", value<int>()->default_value(0),
        "Number of supervised iterations before unsupervised training.")
     ("num-particles", value<int>()->default_value(1000),
        "Number of particles in training.")
     ("max-beam-size", value<int>()->default_value(8),
        "Maximum beam size for decoding (in powers of 2).")
     ("max-beam-increment", value<int>()->default_value(1),
        "Maximum items to add to beam from one state.")
     ("generate-samples", value<int>()->default_value(0),
        "Number of samples to generate.")
     ("direction-det", value<bool>()->default_value(false),
        "Arc direction always deterministic in beam search.")
     ("sum-over-beam", value<bool>()->default_value(false),
        "Sum over likelihoods of identical parses in final beam.")
     ("diagonal-contexts", value<bool>()->default_value(false),
        "Use diagonal context matrices (usually faster).")
     ("activation", value<std::string>()->default_value("sigmoid"),
        "Activation function for to the projection (hidden) layer.")
     ("noise-samples", value<int>()->default_value(0),
        "Number of noise samples for noise contrastive estimation. "
        "If zero, minibatch gradient descent is used instead.")
     ("classes", value<int>()->default_value(100),
        "Number of classes for factored output using frequency binning.")
     ("class-file", value<string>()->default_value(""),
        "File containing word to class mappings in the format "
        "<class> <word> <frequence>.")
     ("lower-class-file", value<string>()->default_value(""),
        "File containing lower word to class mappings in the format "
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
  if (vm.count("training-set-unsup")) {
    config->training_file_unsup = vm["training-set-unsup"].as<std::string>();
  }
  if (vm.count("test-out-file")) {
    config->test_output_file = vm["test-out-file"].as<string>();
  }
  if (vm.count("test-set")) {
    config->test_file = vm["test-set"].as<string>();
  }
  if (vm.count("test-set2")) {
    config->test_file2 = vm["test-set2"].as<string>();
  }
  if (vm.count("test-set-unsup")) {
    config->test_file_unsup = vm["test-set-unsup"].as<std::string>();
  }

  config->pyp_model = false;
  config->distance_range = 5;
  config->iterations = vm["iterations"].as<int>();
  config->iterations_unsup = vm["iterations-unsup"].as<int>();
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
  if (parser_type_str == "arcstandard" || parser_type_str == "arcstandard2") {
    config->parser_type = ParserType::arcstandard;
    if (parser_type_str == "arcstandard2") {
      config->parser_type = ParserType::arcstandard2;
    }
    if (config->context_type == "stack-action") {
      config->stack_ctx_size = vm["stack-context-size"].as<int>();
      config->action_ctx_size = vm["action-context-size"].as<int>();
      config->child_ctx_level = vm["child-context-level"].as<int>();
     // Assume seperate words and tags, no labels. TODO is this true?
      config->ngram_order = 2 * config->stack_ctx_size +
                            config->action_ctx_size +
                            8 * config->child_ctx_level +
                            1; 
    //TODO How much of these we still really need?
    } else if (config->context_type == "extended") {
      config->ngram_order = lblOrderASx;
    } else if (config->context_type == "extended-3rd") {
      config->ngram_order = lblOrderASx3;
    } else if (config->context_type == "more-extended") {
      config->ngram_order = lblOrderASxx;
    } else if (config->context_type == "more-extended-3rd") {
      config->ngram_order = lblOrderASxx3;
    } else if (config->context_type == "with-ngram") {
      config->ngram_order = lblOrderASn;
    } else if (config->context_type == "extended-with-ngram") {
      config->ngram_order = lblOrderASxn;
    } else if (config->context_type == "more-extended-with-ngram") {
      config->ngram_order = lblOrderASxxn;
    } else if (config->context_type == "lookahead") {
      config->ngram_order = lblOrderASl;
    } else if (config->context_type == "extended-lookahead") {
      config->ngram_order = lblOrderASxl;
    } else if (config->context_type == "standard-3rd") {
      config->ngram_order = lblOrderAS3;
    } else {
      config->ngram_order = lblOrderAS;
    }
  } else {
    config->parser_type = ParserType::ngram;
  }

  config->tag_pos = vm["tag-pos"].as<bool>();
  config->predict_pos = vm["predict-pos"].as<bool>();
  config->sample_decoding = vm["sample-decoding"].as<bool>();

  std::string activation_str = vm["activation"].as<std::string>();
  if (activation_str == "linear") {
    config->activation = Activation::linear;
  } else if (activation_str == "sigmoid") {
    config->activation = Activation::sigmoid;
  } else if (activation_str == "tanh") {
    config->activation = Activation::tanh;
  } else if (activation_str == "rectifier") {
    config->activation = Activation::rectifier;
  } else {
    config->activation = Activation::linear;
  }

  config->labelled_parser = vm["labelled-parser"].as<bool>();
  config->lexicalised = vm["lexicalised"].as<bool>();
  config->label_features = vm["label-features"].as<bool>();
  config->morph_features = vm["morph-features"].as<bool>();
  config->distance_features = vm["distance-features"].as<bool>();
  config->direction_deterministic = vm["direction-det"].as<bool>();
  config->sum_over_beam = vm["sum-over-beam"].as<bool>();
  config->semi_supervised = vm["semi-supervised"].as<bool>();
  config->restricted_semi_supervised = vm["restricted-semi-supervised"].as<bool>();
  config->root_first = vm["root-first"].as<bool>();
  config->complete_parse = vm["complete-parse"].as<bool>();
  config->bootstrap = vm["bootstrap"].as<bool>();
  config->bootstrap_iter = vm["bootstrap-iter"].as<int>();
  config->num_particles = vm["num-particles"].as<int>();
  config->max_beam_increment = vm["max-beam-increment"].as<int>();
  config->generate_samples = vm["generate-samples"].as<int>();

  if (config->predict_pos) config->ngram_order += 1;

  config->beam_sizes = {static_cast<unsigned>(vm["max-beam-size"].as<int>())};

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
  if (vm.count("lower-class-file")) {
    config->lower_class_file = vm["lower-class-file"].as<string>();
  }

  //TODO Check that everything is printed out.
  std::cerr << "################################" << std::endl;
  if (strlen(GIT_REVISION) > 0) {
    std::cerr << "# Git revision: " << GIT_REVISION << std::endl;
  }
  std::cerr << "# Config Summary" << std::endl;
  std::cerr << "# order = " << config->ngram_order << std::endl;
  std::cerr << "# representation_size = " << config->representation_size
            << std::endl;
  std::cerr << "# class factored = " << config->factored << std::endl;
  std::cerr << "# predict pos = " << config->predict_pos << std::endl;
  std::cerr << "# label features = " << config->label_features << std::endl;
  std::cerr << "# morph features = " << config->morph_features << std::endl;
  std::cerr << "# distance features = " << config->distance_features
            << std::endl;
  std::cerr << "# parser type = " << parser_type_str << std::endl;
  std::cerr << "# context type = " << config->context_type << std::endl;
  std::cerr << "# labelled parser = " << config->labelled_parser << std::endl;
  std::cerr << "# lexicalised parser = " << config->lexicalised << std::endl;
  std::cerr << "# root first = " << config->root_first << std::endl;
  std::cerr << "# bootstrap = " << config->bootstrap << std::endl;
  std::cerr << "# bootstrap iter = " << config->bootstrap_iter << std::endl;
  std::cerr << "# complete parse = " << config->complete_parse << std::endl;
  std::cerr << "# direction deterministic = " << config->direction_deterministic
            << std::endl;
  std::cerr << "# sum over beam = " << config->sum_over_beam << std::endl;
  std::cerr << "# max beam size = " << config->beam_sizes.back() << std::endl;
  if (config->model_input_file.size()) {
    std::cerr << "# model-in = " << config->model_output_file << std::endl;
  }
  if (config->model_output_file.size()) {
    std::cerr << "# model-out = " << config->model_output_file << std::endl;
  }
  std::cerr << "# supervised training data = " << config->training_file
            << std::endl;
  std::cerr << "# unsupervised training data = " << config->training_file_unsup
            << std::endl;
  std::cerr << "# minibatch size = " << config->minibatch_size << std::endl;
  std::cerr << "# lambda = " << config->l2_lbl << std::endl;
  std::cerr << "# step size = " << config->step_size << std::endl;
  std::cerr << "# iterations = " << config->iterations << std::endl;
  std::cerr << "# threads = " << config->threads << std::endl;
  std::cerr << "# randomise = " << config->randomise << std::endl;
  std::cerr << "# diagonal contexts = " << config->diagonal_contexts
            << std::endl;
  std::cerr << "# noise samples = " << config->noise_samples << std::endl;
  std::cerr << "################################" << std::endl;

  if (config->parser_type == ParserType::ngram) {
    if (config->factored) {
      if (config->model_input_file.size() == 0) {
        LblModel<FactoredWeights, FactoredWeights, FactoredMetadata> model(
            config);
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
    if (config->parser_type == ParserType::arcstandard ||
        config->parser_type == ParserType::arcstandard2) {
      if (config->predict_pos)
        train_dp<ArcStandardLabelledParseModel<TaggedParsedFactoredWeights>,
                 TaggedParsedFactoredWeights, TaggedParsedFactoredMetadata>(
            config);
      else
        train_dp<ArcStandardLabelledParseModel<ParsedFactoredWeights>,
                 ParsedFactoredWeights, ParsedFactoredMetadata>(config);
    }
  } else if (config->parser_type == ParserType::arcstandard ||
             config->parser_type == ParserType::arcstandard2) {
    // TODO For unlexicalised parsing, implicitly factor weights (1 class).
    train_dp<ArcStandardLabelledParseModel<ParsedFactoredWeights>,
             ParsedFactoredWeights, ParsedFactoredMetadata>(config);
  }    
  
  return 0;
}
