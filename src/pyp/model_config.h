#ifndef _GDP_MODEL_CONFIG_H_
#define _GDP_MODEL_CONFIG_H_

#include <string>
#include <vector>

namespace oxlm {

enum class ParserType {ngram, eisner, arcstandard, arceager};

struct ModelConfig {
  ModelConfig();
  
  std::string training_file;
  std::string test_file;
  int iterations;
  int minibatch_size;
  bool randomise;
  ParserType parser_type;
  bool lexicalised;
  int vocab_size; 
  int num_tags;
  int num_actions;
  std::vector<unsigned> beam_sizes;

  bool operator==(const ModelConfig& other) const;
};

}

#endif
