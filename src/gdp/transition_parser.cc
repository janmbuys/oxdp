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

TransitionParser::TransitionParser(const ParsedSentence& parse, int num_particles):  
  Parser(parse),  
  stack_(),
  buffer_next_{0},
  actions_(),
  importance_weight_{0},
  beam_weight_{0},
  num_particles_{num_particles}
  {
  }

void TransitionParser::extractExamples(const boost::shared_ptr<ParseDataSet>& examples) const {
  TransitionParser parser(*this); //will this work?
 
  for (kAction& a: actions()) {
    //TODO update: if in set of shift actions
    if (a == kAction::sh) {
      //tag prediction
      examples->add_tag_example(DataPoint(parser.next_tag(), parser.tagContext(a)));  
       
      //word prediction
      //if (!(word_examples == nullptr))  //do we want to do this?
      examples->add_word_example(DataPoint(parser.next_word(), parser.wordContext()));  
    }  

    //action prediction
    examples->add_action_example(DataPoint(static_cast<WordId>(a), parser.actionContext()));
    //std::cout << static_cast<WordId>(a) << std::endl;
    parser.executeAction(a);
  }
}


}
