#include "transition_parser.h"

namespace oxlm {

TransitionParser::TransitionParser(const boost::shared_ptr<ModelConfig>& config):
  Parser(),
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{1},
  config_(config)
  {
    if (!config->root_first)
      buffer_next_ = 1;
  }

TransitionParser::TransitionParser(const TaggedSentence& parse, const boost::shared_ptr<ModelConfig>& config):  
  Parser(parse),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{1},
  config_(config)
  {
    if (!config->root_first)
      buffer_next_ = 1;
  }

TransitionParser::TransitionParser(const TaggedSentence& parse, int num_particles, const boost::shared_ptr<ModelConfig>& config):  
  Parser(parse),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{num_particles},
  config_(config)
  {
    if (!config->root_first)
      buffer_next_ = 1;
  }

}
