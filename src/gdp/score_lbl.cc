#include <boost/program_options.hpp>

#include "corpus/dict.h"
#include "corpus/corpus.h"

#include "lbl/context_processor.h"
#include "lbl/utils.h"

#include "gdp/lbl_model.h"

using namespace boost::program_options;
using namespace oxlm;
using namespace std;

template<class Model>
void score(const string& model_file, const string& data_file) {
  boost::shared_ptr<Model> model = boost::make_shared<Model>();
  model->load(model_file);

  boost::shared_ptr<ModelConfig> config = model->getConfig();
  boost::shared_ptr<Dict> dict = model->getDict();
  //NGramModel<Model> ngram_model(config->ngram_order, dict->sos(), dict->eos());

  boost::shared_ptr<SentenceCorpus> test_corpus = boost::make_shared<SentenceCorpus>();
  test_corpus->readFile(data_file, dict, true);

  Real total = 0;
  model->evaluate(test_corpus, total);

  cout << total << endl;
}

int main(int argc, char** argv) {
  options_description desc("Command line options");
  desc.add_options()
      ("help,h", "Print help message.")
      ("model,m", value<string>()->required(), "File containing the model")
      ("type,t", value<int>()->required(), "Model type")
      ("data,d", value<string>()->required(), "File containing the test corpus");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }

  notify(vm);

  string model_file = vm["model"].as<string>();
  string data_file = vm["data"].as<string>();
  ModelType model_type = static_cast<ModelType>(vm["type"].as<int>());

  switch (model_type) {
    case NLM:
      score<LblLM>(model_file, data_file);
      return 0;
    case FACTORED_NLM:
      score<FactoredLblLM>(model_file, data_file);
      return 0;
    default:
      cout << "Unknown model type" << endl;
      return 1;
  }

  return 0;
}
