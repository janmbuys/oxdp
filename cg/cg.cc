// STL
#include <vector>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <random>

// Boost
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/lexical_cast.hpp>

// Eigen
#include <Eigen/Core>

// Local
#include "cg/cnlm.h"
#include "corpus/corpus.h"

static const char *REVISION = "$Rev: 247 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;


typedef vector<WordId> Context;
//bool contextLT(const Context& lhs, const Context& rhs) 
//{ return lhs < rhs; }

struct Hypothesis;
typedef std::shared_ptr<Hypothesis> HypothesisPtr;
struct Hypothesis {
  Hypothesis(const Context& c, Real s, HypothesisPtr p) : context(c), score(s), prev(p) {}
  Context context;
  Real score;
  HypothesisPtr prev;
};

bool hypothesisLT(const Hypothesis& lhs, const Hypothesis& rhs) 
{ return lhs.score < rhs.score; }
bool hypothesisPtrLT(const HypothesisPtr& lhs, const HypothesisPtr& rhs) 
{ return lhs->score < rhs->score; }

typedef map<Context, HypothesisPtr> Hypotheses;
typedef vector<HypothesisPtr> Beam;


void beam_search(const ConditionalNLM& model, const Sentence& source, int beam_width, double word_penalty, const Sentence& target) {
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");
  
  Beam old_beam, new_beam, completed;
  Hypotheses seen_contexts;
  VectorReal prediction_vector, source_vector, class_probs, word_probs;

  // initialise the beam
  HypothesisPtr init_h(new Hypothesis(Context(model.config.ngram_order-1, start_id), 0, 0));
  old_beam.push_back(init_h);

  Real best = numeric_limits<Real>::max();
  //for (int t_i=0; t_i < int(source.size()*2); ++t_i) {
  int max_len = (target.empty() ? source.size()*2 : target.size());
  for (int t_i=0; t_i < max_len && !old_beam.empty(); ++t_i) {
    seen_contexts.clear();
    model.source_representation(source, t_i, source_vector);

    for (int b=0; b < int(old_beam.size()) && b < beam_width; ++b) {
      HypothesisPtr h = old_beam.at(b);
      model.hidden_layer(h->context, source_vector, prediction_vector);

      // calculate the distribution over classes
      model.class_log_probs(h->context, source_vector, prediction_vector, class_probs, false);

      for (int c=0; c < class_probs.rows(); ++c) {
        Real clp = -class_probs(c);

        // calculate the distribution over words generated by this class
        model.word_log_probs(c, h->context, source_vector, prediction_vector, word_probs, false);

        Context context(h->context.begin()+1, h->context.end());
        context.push_back(0);

        for (WordId w=0; w < word_probs.rows(); ++w) {
          Real wlp = -word_probs(w);
          WordId w_id = model.map_class_to_word_index(c, w);

          if (w_id == end_id && !target.empty() && t_i != int(target.size()-1))
            continue;

          context.back() = w_id;

          HypothesisPtr new_h(new Hypothesis(context, h->score + clp + wlp - word_penalty, h));

          if (best < new_h->score) continue;
          else if (w_id == end_id) best = new_h->score;

          // recombine this hypothesis if it has been seen before
          auto hyp_ptr = seen_contexts.insert(make_pair(context, new_h));
          if (new_h->score < hyp_ptr.first->second->score) {
            *hyp_ptr.first->second = *new_h;
          }
          else if (hyp_ptr.second) {
            if (w_id == end_id) completed.push_back(new_h); 
            else                new_beam.push_back(new_h);
          }
        }
      }
    }

    old_beam.swap(new_beam);
    new_beam.clear();
    sort(old_beam.begin(), old_beam.end(), hypothesisPtrLT);
/*
    cout << "  " << t_i << ":";
    for (int i=0; i < 10 && i < int(old_beam.size()); ++i)
      cout << " " << model.label_set().Convert(old_beam.at(i)->context.back()) << "=" << old_beam.at(i)->score;
    cout << endl;
*/
  }

  // extract the Viterbi string
  assert(!completed.empty());
  sort(completed.begin(), completed.end(), hypothesisPtrLT);
  HypothesisPtr next_h = completed.front()->prev;
  Sentence result;
  while (next_h->prev) {
    result.push_back(next_h->context.back());
    next_h = next_h->prev;
  }
  reverse(result.begin(), result.end());
  for (auto s : result) 
    cout << model.label_set().Convert(s) << " ";
  //cout << " = " << completed.front()->score << endl;
  cout << endl;
}


void sample_search(const ConditionalNLM& model, const Sentence& source, 
                   int num_samples, int k_best, int source_counter, double word_penalty) {
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Real> distribution(0.0,1.0);

  VectorReal prediction_vector, source_vector, class_probs, word_probs;
  set< pair<Real,Sentence> > samples;

  for (int sample_count=0; sample_count < num_samples; ++sample_count) {
    Sentence sample(model.config.ngram_order-1, start_id);
    Real sample_log_prob=0;
    while (sample.back() != end_id) {
      vector<WordId> context(sample.end()-model.config.ngram_order+1, sample.end());
      Real point = distribution(gen);
      Real accumulator=0.0;

      model.source_representation(source, sample.size()-model.config.ngram_order+1, source_vector);
      model.hidden_layer(context, source_vector, prediction_vector);

      // sample a class
      model.class_log_probs(context, source_vector, prediction_vector, class_probs, false);

      //        class_probs.array() *= 0.7;
      //        class_probs.array() -= log(class_probs.array().exp().sum());

      int c=0;
      Real clp=0;
      for (; c < class_probs.rows(); ++c) {
        clp = class_probs(c);
        if ((accumulator += exp(clp)) >= point || c == class_probs.rows()-1) break;
      }

      // sample a word
      point = distribution(gen);
      accumulator=0;
      model.word_log_probs(c, context, source_vector, prediction_vector, word_probs, false);

      //        word_probs.array() *= 0.7; 
      //        word_probs.array() -= log(word_probs.array().exp().sum());

      for (WordId w=0; w < word_probs.rows(); ++w) {
        Real wlp = word_probs(w);
        if ((accumulator += exp(wlp)) >= point || w == word_probs.rows()-1) {
          sample.push_back(model.map_class_to_word_index(c, w));
          sample_log_prob += clp+wlp + word_penalty;
          //            sample_log_prob += clp+wlp;
          break;
        }
      }
    }
    //      sample_log_prob -= (vm["word-penalty"].as<double>() * floor(fabs(Real(s.size()) - (model.length_ratio*(sample.size() - model.config.ngram_order)))));

    //      samples[sample.size()-model.config.ngram_order].insert(
    //              make_pair(-sample_log_prob, Sentence(sample.begin()+model.config.ngram_order-1, sample.end()-1)));
    samples.insert(make_pair(-sample_log_prob, Sentence(sample.begin()+model.config.ngram_order-1, sample.end()-1)));
  }
  /*
     for (auto s : samples) {
     int c=0;
     for (auto p : s.second) {
     cout << source_counter << " ||| " << s.first << " ||| ";
     for (auto w : p.second)
     cout << model.label_set().Convert(w) << " ";
     cout << "||| " << p.first << endl;
     if (++c > 5) break;
     }
     }
     */
  if (k_best == 1) {
    for (auto w : samples.begin()->second)
      cout << model.label_set().Convert(w) << " ";
    cout << endl;
  }
  else {
    int c=0;
    for (auto s : samples) {
      cout << source_counter << " ||| ";
      for (auto w : s.second)
        cout << model.label_set().Convert(w) << " ";
      cout << "||| " << s.first << endl;
      if (++c >= k_best) break;
    }
  }
}


int main(int argc, char **argv) {
  cerr << "Conditional generation from neural translation models: Copyright 2013 Phil Blunsom, " 
       << REVISION << '\n' << endl;

  ///////////////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm; 

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", value<string>(), 
        "config file specifying additional command line options")
    ;
  options_description generic("Allowed options");
  generic.add_options()
    ("source,s", value<string>(), 
        "corpus of sentences, one per line")
    ("target,t", value<string>(), 
        "reference translations of the source sentences, one per line")
    ("samples", value<int>(),
        "number of samples from the nlm")
    ("beam", value<int>()->default_value(10), 
        "number of hypotheses to extend at each generation step")
    ("k-best", value<int>()->default_value(1), 
        "output k-best samples.")
    ("model-in,m", value<string>(), 
        "model to generate from")
    ("threads", value<int>()->default_value(1), 
        "number of worker threads.")
    ("word-penalty", value<double>()->default_value(0), 
        "word penalty added to the sample log prob for each word generated.")
    ;
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm); 
  if (vm.count("config") > 0) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, cmdline_options), vm); 
  }
  notify(vm);
  ///////////////////////////////////////////////////////////////////////////////////////
  
  if (vm.count("help") || !vm.count("source") || !vm.count("model-in")) { 
    cerr << cmdline_options << "\n"; 
    return 1; 
  }

  omp_set_num_threads(vm["threads"].as<int>());

  ConditionalNLM model;
  std::ifstream f(vm["model-in"].as<string>().c_str());
  boost::archive::text_iarchive ar(f);
  ar >> model;
  cerr << model.length_ratio << endl;

  WordId end_id = model.label_set().Lookup("</s>");

  //////////////////////////////////////////////
  // process the input sentences
  string line, token;
  ifstream source_in(vm["source"].as<string>().c_str());
  ifstream* target_in=NULL;
  if (vm.count("target"))
    target_in = new ifstream(vm["target"].as<string>().c_str());

  int source_counter=0;
  while (getline(source_in, line)) {
    // read the sentence
    stringstream line_stream(line);
    Sentence s;
    while (line_stream >> token) 
      s.push_back(model.source_label_set().Convert(token));
    if (model.config.source_eos) s.push_back(end_id);

    Sentence t;
    if (target_in) {
      assert(getline(*target_in, line));
      stringstream target_line_stream(line);
      while (target_line_stream >> token) 
        t.push_back(model.label_set().Convert(token));
      t.push_back(end_id);
    }

    if (vm.count("samples"))  
      sample_search(model, s, 
                    vm["samples"].as<int>(), vm["k-best"].as<int>(), source_counter, 
                    vm["word-penalty"].as<double>());
    else
      beam_search(model, s, vm["beam"].as<int>(), vm["word-penalty"].as<double>(), t);

    source_counter++;
  }
  source_in.close();
  if (target_in) delete target_in;
  //////////////////////////////////////////////

  return 0;
}

