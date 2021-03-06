#ifndef _CORPUS_MODEL_CONFIG_H_
#define _CORPUS_MODEL_CONFIG_H_

#include <string>
#include <vector>
#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>

namespace oxlm {

enum class ParserType { ngram, arcstandard, arcstandard2 };
enum class Activation { linear, sigmoid, tanh, rectifier };

// Represents a set of global model parameters.
struct ModelConfig {
  ModelConfig();

  std::string training_file;
  std::string training_file_unsup;
  std::string training_file_ques;
  std::string test_file;
  std::string test_file2;
  std::string test_file_unsup;
  std::string test_output_file;
  std::string test_output_file2;
  int iterations;
  int iterations_unsup;
  int minibatch_size;
  int minibatch_size_unsup;
  int ngram_order;
  std::string model_input_file;
  std::string model_output_file;
  float l2_lbl;
  int representation_size;
  int threads;
  float step_size;
  bool factored;
  int classes;
  std::string class_file;
  std::string lower_class_file;
  bool randomise;
  bool diagonal_contexts;
  int vocab_size;
  int noise_samples;
  ParserType parser_type;
  Activation activation;
  bool pyp_model;
  std::string context_type;
  bool labelled_parser;
  bool tag_pos;
  bool predict_pos;
  bool lexicalised;
  bool label_features;
  bool morph_features;
  bool distance_features;
  bool semi_supervised;
  bool restricted_semi_supervised;
  bool direction_deterministic;
  bool sum_over_beam;
  bool resample;
  bool sample_decoding;
  bool root_first;
  bool complete_parse;
  bool bootstrap;
  int bootstrap_iter;
  int max_beam_increment;
  int num_particles;
  int generate_samples;
  int num_tags;
  int num_features;
  int num_labels;
  int num_train_sentences;
  int action_feature_index;
  int label_feature_index;
  int distance_feature_index;
  size_t distance_range;
  int stack_ctx_size;
  int action_ctx_size;
  int child_ctx_level;
  int out_ctx_size;
  int in_window_size;
  std::vector<unsigned> beam_sizes;
  std::vector<std::vector<int>> word_to_features;
  std::vector<int> tag_to_feature;

  bool operator==(const ModelConfig& other) const;

  int numActions() const;
  void addWordFeatures(const std::vector<int>& features);
  std::vector<int> getWordFeatures(int id) const;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& training_file;
    ar& training_file_unsup;
    ar& training_file_ques;
    ar& test_file;
    ar& test_file2;
    ar& test_file_unsup;
    ar& test_output_file;
    ar& test_output_file2;
    ar& minibatch_size;
    ar& minibatch_size_unsup;
    ar& ngram_order;
    ar& model_input_file;
    ar& model_output_file;
    ar& l2_lbl;
    ar& representation_size;
    ar& threads;
    ar& step_size;
    ar& factored;
    ar& classes;
    ar& class_file;
    ar& lower_class_file;
    ar& randomise;
    ar& diagonal_contexts;
    ar& vocab_size;
    ar& noise_samples;
    ar& parser_type;
    ar& activation;
    ar& pyp_model;
    ar& labelled_parser;
    ar& tag_pos;
    ar& predict_pos;
    ar& lexicalised;
    ar& label_features;
    ar& morph_features;
    ar& distance_features;
    ar& semi_supervised;
    ar& restricted_semi_supervised;
    ar& direction_deterministic;
    ar& sum_over_beam;
    ar& resample;
    ar& sample_decoding;
    ar& root_first;
    ar& complete_parse;
    ar& bootstrap;
    ar& bootstrap_iter;
    ar& max_beam_increment;
    ar& num_particles;
    ar& generate_samples;
    ar& num_tags;
    ar& num_features;
    ar& num_labels;
    ar& num_train_sentences;
    ar& action_feature_index;
    ar& label_feature_index;
    ar& distance_feature_index;
    ar& distance_range;
    ar& stack_ctx_size;
    ar& action_ctx_size;
    ar& child_ctx_level;
    ar& out_ctx_size;
    ar& in_window_size;
    ar& beam_sizes;
    ar& word_to_features;
    ar& tag_to_feature;
  }
};

}  // namespace oxlm

#endif
