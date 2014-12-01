#include "corpus/model_config.h"

#include "utils/constants.h"

namespace oxlm {

ModelConfig::ModelConfig()
    : iterations(0), minibatch_size(0), minibatch_size_unsup(0), instances(0), 
      ngram_order(0), feature_context_size(0), l2_lbl(0), l2_maxent(0),
      word_representation_size(0), threads(1), step_size(0), factored(true),
      classes(0), randomise(false), reclass(false), diagonal_contexts(false),
      uniform(false), pseudo_likelihood_cne(false), mixture(false),
      lbfgs(false), lbfgs_vectors(0), test_tokens(0), gnorm_threshold(0),
      eta(0), multinomial_step_size(0), random_weights(false), hash_space(0),
      count_collisions(false), filter_contexts(false), filter_error_rate(0),
      max_ngrams(0), min_ngram_freq(0), vocab_size(0), noise_samples(0), sigmoid(false), 
      parser_type(ParserType::arcstandard), labelled_parser(false), lexicalised(false),
      char_lexicalised(false), semi_supervised(false), direction_deterministic(false), 
      sum_over_beam(false), resample(false), num_particles(1), num_tags(1), num_actions(1), 
      beam_sizes(1, 1) {}

bool ModelConfig::operator==(const ModelConfig& other) const {
  if (fabs(l2_lbl - other.l2_lbl) > EPS ||
      fabs(l2_maxent - other.l2_maxent) > EPS) {
      std::cout << "Warning: Using different regularizers!" << std::endl;
  }

  return (training_file == other.training_file
      && ngram_order == other.ngram_order
      && feature_context_size == other.feature_context_size
      && word_representation_size == other.word_representation_size
      && factored == other.factored
      && classes == other.classes
      && class_file == other.class_file
      && diagonal_contexts == other.diagonal_contexts
      && hash_space == other.hash_space
      && filter_contexts == other.filter_contexts
      && fabs(filter_error_rate - other.filter_error_rate) < EPS
      && sigmoid == other.sigmoid
      && parser_type == other.parser_type
      && labelled_parser == other.labelled_parser
      && lexicalised == other.lexicalised
      && char_lexicalised == other.char_lexicalised
      && semi_supervised == other.semi_supervised
      && direction_deterministic == other.direction_deterministic
      && sum_over_beam == other.sum_over_beam
      && resample == other.resample
      && num_particles == other.num_particles);
}

} //namespace oxlm

