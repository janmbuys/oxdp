#include "corpus/model_config.h"

#include "utils/constants.h"

namespace oxlm {

ModelConfig::ModelConfig()
    : iterations(0), minibatch_size(0), minibatch_size_unsup(0), ngram_order(0), 
      l2_lbl(0), representation_size(0), threads(1), step_size(0), 
      factored(true), classes(0), randomise(false), diagonal_contexts(false),
      vocab_size(0), noise_samples(0), parser_type(ParserType::arcstandard), 
      activation(Activation::linear), pyp_model(false), labelled_parser(false), 
      lexicalised(false), char_lexicalised(false), semi_supervised(false), 
      direction_deterministic(false), sum_over_beam(false), resample(false), 
      root_first(true), max_beam_increment(1), num_particles(1), num_tags(1), num_labels(1), 
      beam_sizes(1, 1) {}

int ModelConfig::numActions() const {
  if (parser_type == ParserType::arcstandard)
    return 2*num_labels + 1;
  else if (parser_type == ParserType::arceager)
    return 2*num_labels + 2;
  else
    return 1;
}

bool ModelConfig::operator==(const ModelConfig& other) const {
  if (fabs(l2_lbl - other.l2_lbl) > EPS) {
      std::cout << "Warning: Using different regularizers!" << std::endl;
  }

  return (training_file == other.training_file
      && training_file_unsup == other.training_file_unsup
      && training_file_ques == other.training_file_ques
      && ngram_order == other.ngram_order
      && representation_size == other.representation_size
      && factored == other.factored
      && classes == other.classes
      && class_file == other.class_file
      && diagonal_contexts == other.diagonal_contexts
      && vocab_size == other.vocab_size
      && noise_samples == other.noise_samples
      && parser_type == other.parser_type
      && activation == other.activation
      && pyp_model == other.pyp_model
      && labelled_parser == other.labelled_parser
      && lexicalised == other.lexicalised
      && char_lexicalised == other.char_lexicalised
      && semi_supervised == other.semi_supervised
      && root_first == other.root_first
      && num_tags == other.num_tags
      && num_labels == other.num_labels);
}

} //namespace oxlm

