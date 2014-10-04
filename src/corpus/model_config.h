#ifndef _CORPUS_MODEL_CONFIG_H_
#define _CORPUS_MODEL_CONFIG_H_

#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>

namespace oxlm {

enum class ParserType {ngram, eisner, arcstandard, arceager};

struct ModelConfig {
  ModelConfig();

  std::string      training_file;
  std::string      training_file_unsup;
  std::string      test_file;
  int         iterations;
  int         minibatch_size;
  int         minibatch_size_unsup;
  int         instances;
  int         ngram_order;
  int         feature_context_size;
  std::string      model_input_file;
  std::string      model_output_file;
  float       l2_lbl;
  float       l2_maxent;
  int         word_representation_size;
  int         threads;
  float       step_size;
  bool        factored;
  int         classes;
  std::string      class_file;
  bool        randomise;
  bool        reclass;
  bool        diagonal_contexts;
  bool        uniform;
  bool        pseudo_likelihood_cne;
  bool        mixture;
  bool        lbfgs;
  int         lbfgs_vectors;
  int         test_tokens;
  float       gnorm_threshold;
  float       eta;
  float       multinomial_step_size;
  bool        random_weights;
  int         hash_space;
  bool        count_collisions;
  bool        filter_contexts;
  float       filter_error_rate;
  int         max_ngrams;
  int         min_ngram_freq;
  int         vocab_size;
  int         noise_samples;
  bool        sigmoid;
  ParserType  parser_type;
  bool        lexicalised;
  bool        semi_supervised;
  int         num_tags;
  int         num_actions;
  std::vector<unsigned> beam_sizes;

  bool operator==(const ModelConfig& other) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & training_file;
    ar & training_file_unsup;
    ar & test_file;
    ar & iterations;
    ar & minibatch_size;
    ar & minibatch_size_unsup;
    ar & instances;
    ar & ngram_order;
    ar & feature_context_size;
    ar & model_input_file;
    ar & model_output_file;
    ar & l2_lbl;
    ar & l2_maxent;
    ar & word_representation_size;
    ar & step_size;
    ar & factored;
    ar & classes;
    ar & class_file;
    ar & randomise;
    ar & diagonal_contexts;
    ar & hash_space;
    ar & filter_contexts;
    ar & filter_error_rate;
    ar & vocab_size;
    ar & noise_samples;
    ar & sigmoid;
    ar & parser_type;
    ar & lexicalised;
    ar & semi_supervised;
    ar & num_tags;
    ar & beam_sizes;
  }
};

}

#endif
