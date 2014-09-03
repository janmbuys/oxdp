#ifndef _GDP_MODEL_CONFIG_H_
#define _GDP_MODEL_CONFIG_H_

#include <string>

namespace oxlm {

enum class ParserType {eisner, arcstandard, arceager};

struct ModelConfig {
  ModelConfig();
  
  std::string training_file;
  std::string test_file;
  int iterations;
  int minibatch_size;
  bool randomise;
  ParserType parser_type;
  bool lexicalised;

  bool operator==(const ModelConfig& other) const;
};

}

#endif
