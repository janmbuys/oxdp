#include <boost/program_options.hpp>

#include "corpus/utils.h"
#include "corpus/dict.h"
#include "corpus/model_config.h"

#include "gdp/pyp_model.h"
#include "gdp/pyp_dp_model.h"

using namespace boost::program_options;
using namespace oxlm;

template<class ParseModel, class ParsedWeights>
void train_dp(const boost::shared_ptr<ModelConfig>& config) {
  PypDpModel<ParseModel, ParsedWeights> model(config);
  model.learn();
  if (config->iterations > 1)
    model.evaluate();
}

int main(int argc, char** argv) {
 options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", value<std::string>(),
        "config file specifying additional command line options");

  options_description generic("Allowed options");
  generic.add_options()
    ("training-set,i", value<std::string>(),
        "corpus of parsed sentences for training, conll format")
    ("training-set-unsup,u", value<std::string>(),
        "corpus of unparsed sentences for semi-supervised training, conll format")
    ("training-set-ques,q", value<std::string>(),
        "corpus of parsed questions for semi-supervised training, conll format")
    ("test-set,t", value<std::string>(),
        "corpus of test sentences to be evaluated at each iteration")
    ("test-out-file,o", value<std::string>()->default_value("system.out.conll"),
        "conll output file for system parsing the test set")
    ("iterations", value<int>()->default_value(1),
        "number of passes through the data")
    ("minibatch-size", value<int>()->default_value(1),
        "number of sentences per minibatch")
    ("minibatch-size-unsup", value<int>()->default_value(1),
        "number of sentences per minibatch, unsupervised training")
    ("randomise", value<bool>()->default_value(true),
        "Visit the training tokens in random order.")
    ("parser-type", value<std::string>()->default_value("arcstandard"),
        "Parsing strategy.")
    ("context-type", value<std::string>()->default_value(""),
        "Conditioning context used.")    
    ("labelled-parser", value<bool>()->default_value(false),
        "Predict arc labels.")
    ("predict-pos", value<bool>()->default_value(false),
        "Predict POS during decoding.")
    ("lexicalised", value<bool>()->default_value(true),
        "Predict words in addition to POS tags.")
    ("pos-annotated", value<bool>()->default_value(false),
        "Use word_POS as input feature.")
    ("root-first", value<bool>()->default_value(true),
        "Add root to the beginning (else end) of the sentence.")
    ("bootstrap", value<bool>()->default_value(false),
        "Extract training data with beam search.")
    ("bootstrap-iter", value<int>()->default_value(0),
        "Number of supervised iterations before unsupervised training.")
    ("complete-parse", value<bool>()->default_value(true),
        "Enforce complete tree-structured parse.")
    ("char-lexicalised", value<bool>()->default_value(false),
        "Predict words with character-based LM.")
    ("semi-supervised", value<bool>()->default_value(false),
        "Use additional, unlabelled training data.")
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
    ("particle-resample", value<bool>()->default_value(false),
        "Resample after generating each word in particle filter sampling.")
    ("num-particles", value<int>()->default_value(100),
        "Number of particles in particle filter.")
    ("model-out,m", value<std::string>(),  //not used now
        "base filename of model output files")
    ("threads", value<int>()->default_value(1), //not used now
        "number of worker threads.");
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  variables_map vm;
  store(parse_command_line(argc, argv, cmdline_options), vm);
  if (vm.count("config") > 0) {
      std::ifstream config(vm["config"].as<std::string>().c_str());
    store(parse_config_file(config, cmdline_options), vm);
  }

  if (vm.count("help")) {
    std::cerr << cmdline_options << "\n";
    return 1;
  }

  notify(vm);
  
  boost::shared_ptr<ModelConfig> config = boost::make_shared<ModelConfig>();
 
  if (vm.count("training-set")) {
    config->training_file = vm["training-set"].as<std::string>();
  }
  if (vm.count("training-set-unsup")) {
    config->training_file_unsup = vm["training-set-unsup"].as<std::string>();
  }
  if (vm.count("training-set-ques")) {
    config->training_file_ques = vm["training-set-ques"].as<std::string>();
  }
  if (vm.count("test-set")) {
    config->test_file = vm["test-set"].as<std::string>();
  }
  if (vm.count("test-out-file")) {
    config->test_output_file = vm["test-out-file"].as<std::string>();
  }

  config->pyp_model = true;
  config->iterations = vm["iterations"].as<int>();
  config->minibatch_size = vm["minibatch-size"].as<int>();
  config->minibatch_size_unsup = vm["minibatch-size-unsup"].as<int>();
  config->randomise = vm["randomise"].as<bool>();
  config->pos_annotated = vm["pos-annotated"].as<bool>();

  config->context_type = vm["context-type"].as<std::string>(); //not currently used
  std::string parser_type_str = vm["parser-type"].as<std::string>();
  if (parser_type_str == "arcstandard") 
    config->parser_type = ParserType::arcstandard; 
  else if (parser_type_str == "arcstandard2") 
    config->parser_type = ParserType::arcstandard2; 
  else if (parser_type_str == "arceager")
    config->parser_type = ParserType::arceager; 
  else if (parser_type_str == "eisner")
    config->parser_type = ParserType::eisner; 
  else
    config->parser_type = ParserType::ngram; 

  config->labelled_parser = vm["labelled-parser"].as<bool>();
  config->predict_pos = vm["predict-pos"].as<bool>();
  config->lexicalised = vm["lexicalised"].as<bool>();
  config->char_lexicalised = vm["char-lexicalised"].as<bool>();
  config->semi_supervised = vm["semi-supervised"].as<bool>();
  config->direction_deterministic = vm["direction-det"].as<bool>();
  config->sum_over_beam = vm["sum-over-beam"].as<bool>();
  config->resample = vm["particle-resample"].as<bool>();
  config->root_first = vm["root-first"].as<bool>();
  config->bootstrap = vm["bootstrap"].as<bool>();
  config->bootstrap_iter = vm["bootstrap-iter"].as<int>();
  config->num_particles = vm["num-particles"].as<int>();
  config->max_beam_increment = vm["max-beam-increment"].as<int>();
  config->generate_samples = vm["generate-samples"].as<int>();
  config->complete_parse = vm["complete-parse"].as<bool>();

  config->beam_sizes = {static_cast<unsigned>(vm["max-beam-size"].as<int>())};
  //for (int i = 2; i <= vm["max-beam-size"].as<int>(); i *= 2)
  //  config->beam_sizes.push_back(i);
 
  std::cerr << "################################" << std::endl;
  std::cerr << "# Config Summary" << std::endl;
    std::cerr << "# parser type = " << parser_type_str << std::endl;
  std::cerr << "# context type = " << config->context_type << std::endl;
  std::cerr << "# labelled parser = " << config->labelled_parser << std::endl;
  std::cerr << "# lexicalised parser = " << config->lexicalised << std::endl;
  std::cerr << "# root first = " << config->root_first << std::endl;
  std::cerr << "# bootstrap = " << config->bootstrap << std::endl;
  std::cerr << "# bootstrap iter = " << config->bootstrap_iter << std::endl;
  std::cerr << "# direction deterministic = " << config->direction_deterministic << std::endl;
  std::cerr << "# complete parse = " << config->complete_parse << std::endl;
  std::cerr << "# sum over beam = " << config->sum_over_beam << std::endl;
  std::cerr << "# max beam size = " << config->beam_sizes.back() << std::endl;

  std::cerr << "# supervised training data = " << config->training_file << std::endl;
  std::cerr << "# unsupervised training data = " << config->training_file_unsup << std::endl;
  std::cerr << "# question training data = " << config->training_file_ques << std::endl;
  std::cerr << "# semi-supervised = " << config->semi_supervised << std::endl;
  std::cerr << "# character LM = " << config->char_lexicalised << std::endl;
  std::cerr << "# minibatch size = " << config->minibatch_size << std::endl;
  std::cerr << "# particle resample = " << config->resample << std::endl;
  std::cerr << "# number of particles = " << config->num_particles << std::endl;
  std::cerr << "# iterations = " << config->iterations << std::endl;
  std::cerr << "# randomise = " << config->randomise << std::endl;



  std::cerr << "################################" << std::endl;

  if (config->parser_type == ParserType::ngram) {
    PypModel model(config); 
    model.learn();
  } else {
   if (config->char_lexicalised) {
      if (config->parser_type == ParserType::arcstandard)
        train_dp<ArcStandardLabelledParseModel<ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>, ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>(config);
      //else if (config->parser_type == ParserType::arcstandard)
      //  train_dp<ArcStandardParseModel<ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>, ParsedChLexPypWeights<wordLMOrderAS, charLMOrder, tagLMOrderAS, actionLMOrderAS>>(config);
      else if (config->parser_type == ParserType::arceager)
        train_dp<ArcEagerLabelledParseModel<ParsedChLexPypWeights<wordLMOrderAE, charLMOrder, tagLMOrderAE, actionLMOrderAE>>, ParsedChLexPypWeights<wordLMOrderAE, charLMOrder, tagLMOrderAE, actionLMOrderAE>>(config);
      //else
      //  train_dp<EisnerParseModel<ParsedChLexPypWeights<wordLMOrderE, charLMOrder, tagLMOrderE, 1>>, ParsedChLexPypWeights<wordLMOrderE, charLMOrder, tagLMOrderE, 1>>(config);
    } 
   else if (config->lexicalised) {
      if (config->parser_type == ParserType::arcstandard || config->parser_type == ParserType::arcstandard2) // && config->labelled_parser)
        train_dp<ArcStandardLabelledParseModel<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>, ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>(config);
      //else if (config->parser_type == ParserType::arcstandard)
      //  train_dp<ArcStandardParseModel<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>, ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>(config);
      else if (config->parser_type == ParserType::arceager) //&& config->labelled_parser)
        train_dp<ArcEagerLabelledParseModel<ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>, ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>(config);
      //else if (config->parser_type == ParserType::arceager) 
      //  train_dp<ArcEagerParseModel<ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>, ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>(config);
      //else
      //  train_dp<EisnerParseModel<ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>, ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>(config);
    } else {
      if (config->parser_type == ParserType::arcstandard || config->parser_type == ParserType::arcstandard2) // && config->labelled_parser)
        train_dp<ArcStandardLabelledParseModel<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>, ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>(config);
      //else if (config->parser_type == ParserType::arcstandard)
      //  train_dp<ArcStandardParseModel<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>, ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>(config);
      else if (config->parser_type == ParserType::arceager) //&& config->labelled_parser)
        train_dp<ArcEagerLabelledParseModel<ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>, ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>(config);
      //else if (config->parser_type == ParserType::arceager)
      //  train_dp<ArcEagerParseModel<ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>, ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>(config);
      //else
      //  train_dp<EisnerParseModel<ParsedPypWeights<tagLMOrderE, 1>>, ParsedPypWeights<tagLMOrderE, 1>>(config);
    }
  }

  return 0;
}

