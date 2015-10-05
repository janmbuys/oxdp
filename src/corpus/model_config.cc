#include "corpus/model_config.h"

#include "utils/constants.h"

namespace oxlm {

ModelConfig::ModelConfig()
    : iterations(0), iterations_sv(0), iterations_unsup(0), iterations_test(0), minibatch_size(0), 
      minibatch_size_unsup(0), ngram_order(0), 
      l2_lbl(0), l2_lbl_sv(0), representation_size(0), threads(1), step_size(0), whole_feature_dropout(0), feature_dropout(0), rms_prop(false),
      factored(true), classes(0), randomise(false), diagonal_contexts(false),
      vocab_size(0), noise_samples(0), parser_type(ParserType::arcstandard), 
      activation(Activation::linear), pyp_model(false), labelled_parser(false), 
      discriminative(false), tag_pos(false), predict_pos(false), lexicalised(false), sentence_vector(false),
      //compositional(false), output_compositional(false), pos_annotated(false), 
      label_features(false), morph_features(false), 
      distance_features(false), 
      semi_supervised(false), direction_deterministic(false), sum_over_beam(false), resample(false), 
      root_first(true), complete_parse(true), bootstrap(false), bootstrap_iter(0), max_beam_increment(1), 
      num_particles(1), generate_samples(0), num_tags(1), num_features(1), num_labels(1), 
      num_train_sentences(0), distance_range(0), stack_ctx_size(0), child_ctx_level(0),
      out_ctx_size(0), in_window_size(0), beam_sizes(1, 1) {}

int ModelConfig::numActions() const {
  if (parser_type == ParserType::arcstandard)
    return 2*num_labels + 1;
  else if (parser_type == ParserType::arcstandard2)
    return 4*num_labels + 1;
  else
    return 1;
}

void ModelConfig::addWordFeatures(const std::vector<int>& features) {
  word_to_features.push_back(std::vector<int>());
  for (auto feat: features)
    word_to_features.back().push_back(feat); 
}

std::vector<int> ModelConfig::getWordFeatures(int id) const {
  if (id < 0)
    return std::vector<int>();
  return word_to_features[id];
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
      && lower_class_file == other.lower_class_file
      && diagonal_contexts == other.diagonal_contexts
      //&& vocab_size == other.vocab_size
      && noise_samples == other.noise_samples
      && parser_type == other.parser_type
      && activation == other.activation
      && pyp_model == other.pyp_model
      && context_type == other.context_type
      && labelled_parser == other.labelled_parser
      && discriminative == other.discriminative
      && predict_pos == other.predict_pos
      && lexicalised == other.lexicalised
      && sentence_vector == other.sentence_vector
     // && compositional == other.compositional
    //  && output_compositional == other.output_compositional
    //  && pos_annotated == other.pos_annotated
      && label_features == other.label_features
      && morph_features == other.morph_features
      && distance_features == other.distance_features
      && semi_supervised == other.semi_supervised
      && root_first == other.root_first
      && complete_parse == other.complete_parse
      && distance_range == other.distance_range
      && stack_ctx_size == other.stack_ctx_size
      && child_ctx_level == other.child_ctx_level
      && out_ctx_size == other.out_ctx_size
      && in_window_size == other.in_window_size);
}

} //namespace oxlm

