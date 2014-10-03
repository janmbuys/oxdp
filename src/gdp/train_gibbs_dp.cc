#include <boost/program_options.hpp>

#include "corpus/utils.h"
#include "corpus/dict.h"
#include "corpus/model_config.h"

#include "gdp/pyp_model.h"
#include "gdp/pyp_dp_model.h"

using namespace boost::program_options;
using namespace oxlm;

template<class ParseModel, class ParsedWeights>
void train(const boost::shared_ptr<ModelConfig>& config) {
  PypDpModel<ParseModel, ParsedWeights> model(config);

  //learn
  if (config->semi_supervised)
    model.learn_semi_supervised();
  else
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
    ("training-set,i", 
        value<std::string>()->default_value("english-wsj-stanford-unk/english_wsj_train.conll"),
        "corpus of parsed sentences for training, conll format")
    ("training-set-unsup,i", value<std::string>(),
        "corpus of unparsed sentences for semi-supervised training, conll format")
    ("test-set", value<std::string>()->default_value("english-wsj-stanford-unk/english_wsj_dev.conll"),
        "corpus of test sentences to be evaluated at each iteration")
    ("iterations", value<int>()->default_value(1),
        "number of passes through the data")
    ("minibatch-size", value<int>()->default_value(1),
        "number of sentences per minibatch")
    ("randomise", value<bool>()->default_value(true),
        "Visit the training tokens in random order.")
    ("parser-type", value<std::string>()->default_value("arceager"),
        "Parsing strategy.")
    ("lexicalised", value<bool>()->default_value(true),
        "Predict words in addition to POS tags.")
    ("semi-supervised", value<bool>()->default_value(false),
        "Use additional, unlabelled training data.")
    ("max-beam-size", value<int>()->default_value(8),
        "Maximum beam size for decoding (in powers of 2).")
    ("model-out,o", value<std::string>(),  //not used now
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
    std::cout << cmdline_options << "\n";
    return 1;
  }

  notify(vm);
  
  boost::shared_ptr<ModelConfig> config = boost::make_shared<ModelConfig>();
 
  config->training_file = vm["training-set"].as<std::string>();
  if (vm.count("training-set-unsup")) {
    config->training_file_unsup = vm["training-set-unsup"].as<std::string>();
  }
  if (vm.count("test-set")) {
    config->test_file = vm["test-set"].as<std::string>();
  }

  config->iterations = vm["iterations"].as<int>();
  config->minibatch_size = vm["minibatch-size"].as<int>();
  config->randomise = vm["randomise"].as<bool>();

  std::string parser_type_str = vm["parser-type"].as<std::string>();
  if (parser_type_str == "arcstandard")
    config->parser_type = ParserType::arcstandard; 
  else if (parser_type_str == "arceager")
    config->parser_type = ParserType::arceager; 
  else if (parser_type_str == "eisner")
    config->parser_type = ParserType::eisner; 
  else
    config->parser_type = ParserType::ngram; 

  config->lexicalised = vm["lexicalised"].as<bool>();
  config->semi_supervised = vm["semi-supervised"].as<bool>();

  //otherwise override manually
  config->beam_sizes = {1};
  for (int i = 2; i <= vm["max-beam-size"].as<int>(); i *= 2)
    config->beam_sizes.push_back(i);
  
  if (config->parser_type == ParserType::ngram) {
    PypModel model(config); 
    model.learn();
  } else {
    if (config->lexicalised) {
      if (config->parser_type == ParserType::arcstandard)
        train<ArcStandardParseModel<ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>, ParsedLexPypWeights<wordLMOrderAS, tagLMOrderAS, actionLMOrderAS>>(config);
      else if (config->parser_type == ParserType::arceager)
        train<ArcEagerParseModel<ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>, ParsedLexPypWeights<wordLMOrderAE, tagLMOrderAE, actionLMOrderAE>>(config);
      else
        train<EisnerParseModel<ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>, ParsedLexPypWeights<wordLMOrderE, tagLMOrderE, 1>>(config);
    } else {
      if (config->parser_type == ParserType::arcstandard)
        train<ArcStandardParseModel<ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>, ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>>(config);
      else if (config->parser_type == ParserType::arceager)
        train<ArcEagerParseModel<ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>, ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>>(config);
      else
        train<EisnerParseModel<ParsedPypWeights<tagLMOrderE, 1>>, ParsedPypWeights<tagLMOrderE, 1>>(config);
    }
  }

  return 0;
}

