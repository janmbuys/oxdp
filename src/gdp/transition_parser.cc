#include "transition_parser.h"

namespace oxlm {

//TODO change to put root on initial stack

TransitionParser::TransitionParser():
  Parser(),
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{1}
  {
  }

TransitionParser::TransitionParser(Words tags):
  Parser(tags),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{1}
  {
  }

TransitionParser::TransitionParser(Words sent, Words tags):  
  Parser(sent, tags),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{1}
  {
  }

//copy constructor
/* TransitionParser::TransitionParser(const TransitionParser& parse):  
  Parse(parse),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{1}
  {
  }  */

//what I want to do
//implicitly defines the copy constructor
TransitionParser::TransitionParser(const ParsedSentence& parse):  
  Parser(parse),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{1}
  {
  }

TransitionParser::TransitionParser(Words sent, Words tags, int num_particles):  
  Parser(sent, tags),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{num_particles}
  {
  }

}
