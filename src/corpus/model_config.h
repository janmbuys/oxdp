#ifndef _CORPUS_MODEL_CONFIG_H_
#define _CORPUS_MODEL_CONFIG_H_

#include <string>
#include <vector>
#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>

namespace oxlm {

enum class ParserType {ngram, eisner, arcstandard, arceager, arcstandard2};
enum class Activation {linear, sigmoid, tanh, rectifier};

struct ModelConfig {
  ModelConfig();

  std::string training_file;
  std::string training_file_unsup;
  std::string training_file_ques;
  std::string test_file;
  std::string test_output_file;
  int         iterations;
  int         minibatch_size;
  int         minibatch_size_unsup;
  int         ngram_order; 
  std::string model_input_file;
  std::string model_output_file;
  float       l2_lbl;
  int         representation_size;   
  int         threads;
  float       step_size;
  bool        factored;
  int         classes;
  std::string class_file;
  bool        randomise;
  bool        diagonal_contexts;
  int         vocab_size;
  int         noise_samples;
  ParserType  parser_type;
  Activation  activation;
  bool        pyp_model;
  std::string context_type;
  bool        labelled_parser;
  bool        lexicalised;
  bool        char_lexicalised;
  bool        semi_supervised;
  bool        direction_deterministic;
  bool        sum_over_beam;
  bool        resample;
  bool        root_first;
  bool        bootstrap;
  int         max_beam_increment;
  int         num_particles;
  int         generate_samples;
  int         num_tags;
  int         num_labels;
  std::vector<unsigned> beam_sizes;

  bool operator==(const ModelConfig& other) const;

  int numActions() const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & training_file;
    ar & training_file_unsup;
    ar & training_file_ques;
    ar & test_file;
    ar & test_output_file;
    ar & iterations;
    ar & minibatch_size;
    ar & minibatch_size_unsup;
    ar & ngram_order;
    ar & model_input_file;
    ar & model_output_file;
    ar & l2_lbl;
    ar & representation_size;
    ar & step_size;
    ar & factored;
    ar & classes;
    ar & class_file;
    ar & randomise;
    ar & diagonal_contexts;
    ar & vocab_size;
    ar & noise_samples;
    ar & parser_type;
    ar & activation;
    ar & pyp_model;
    ar & labelled_parser;
    ar & lexicalised;
    ar & char_lexicalised;
    ar & semi_supervised;
    ar & direction_deterministic;
    ar & sum_over_beam;
    ar & resample;
    ar & root_first;
    ar & bootstrap;
    ar & max_beam_increment;
    ar & num_particles;
    ar & generate_samples;
    ar & num_tags;
    ar & num_labels;
  }
};

}

#endif
