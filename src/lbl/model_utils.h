#pragma once

#include <map>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include "corpus/dict.h"
#include "lbl/utils.h"

// Helper functions for reading data, evaluating models, etc.

namespace oxlm {

struct UnigramDistribution {
  map<double, string> prob_to_token;
  map<string, double> token_to_prob;

  void read(const string& filename) {
    ifstream file(filename.c_str());
    cerr << "Reading unigram distribution from "
      << filename.c_str() << "." << endl;

    double sum=0;
    string key, value;
    while (file >> value >> key) {
      double v = boost::lexical_cast<double>(value);
      sum += v;
      prob_to_token.insert(make_pair(sum, key));
      token_to_prob.insert(make_pair(key, v));
    }
  }

  double prob(const string& s) const {
   map<string, double>::const_iterator it
     = token_to_prob.find(s);
   return it != token_to_prob.end() ? it->second : 0.0;
 }

  bool empty() const { return prob_to_token.empty(); }
};
    
vector<int> scatterMinibatch(const vector<int>& minibatch);

void loadClassesFromFile(
    const string& class_file, const string& training_file,
    vector<int>& classes, boost::shared_ptr<Dict>& dict, VectorReal& class_bias);

void frequencyBinning(
    const string& training_file, int num_classes,
    vector<int>& classes, boost::shared_ptr<Dict>& dict, VectorReal& class_bias);

/* int convert(
    const string& file, Dict& dict,
    bool immutable_dict, bool convert_unknowns);

boost::shared_ptr<Corpus> readCorpus(
    const string& file, Dict& dict,
    bool immutable_dict = true, bool convert_unknowns = false);

Real perplexity(Real log_likelihood, size_t corpus_size); */

} // namespace oxlm
