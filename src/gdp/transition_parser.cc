#include "transition_parser.h"

namespace oxlm {

//TODO change to put root on initial stack

TransitionParser::TransitionParser():
  Parse(),
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{1}
  {
  }

TransitionParser::TransitionParser(Words tags):
  Parse(tags),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{1}
  {
  }

TransitionParser::TransitionParser(Words sent, Words tags):  
  Parse(sent, tags),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{1}
  {
  }

TransitionParser::TransitionParser(Words sent, Words tags, int num_particles):  
  Parse(sent, tags),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{num_particles}
  {
  }

}
