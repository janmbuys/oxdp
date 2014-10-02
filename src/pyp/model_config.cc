#include "model_config.h"

namespace oxlm {

ModelConfig::ModelConfig(): 
    iterations(1), 
    minibatch_size(1), 
    randomise(false), 
    parser_type(ParserType::arcstandard),
    lexicalised(false),
    semi_supervised(false),
    vocab_size(1),
    num_tags(1),
    num_actions(1),
    beam_sizes(1, 1)
    {}

bool ModelConfig::operator==(const ModelConfig& other) const {
  return (training_file == other.training_file
      && training_file_unsup == other.training_file_unsup
      && parser_type == other.parser_type
      && lexicalised == other.lexicalised
      && semi_supervised == other.semi_supervised);
}

}

