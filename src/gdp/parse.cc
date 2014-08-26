#include "parse.h"

namespace oxlm {

Parse::Parse()
  ParsedSentence(), 
  left_children_(),
  right_children_(),
  weight_{0}
  {
    //used for generation
    push_arc();
  }

Parse::Parse(Words tags):
  ParsedSentence(tags), 
  left_children_(tags.size(), Indices()),  
  right_children_(tags.size(), Indices())  
  weight_{0}
  {
  }

Parse::Parse(Words sent, Words tags):
  ParsedSentence(sent, tags), 
  left_children_(tags.size(), Indices()),  
  right_children_(tags.size(), Indices())  
  weight_{0}
  {
  }

ParsedSentence::ParsedSentence(Words sent, Words tags, Indices arcs):
  Parse(sent, tags)  //make sure that this is fine
  {
    for (int i = 0; i < size(); ++i) 
      set_arc(i, arcs.at(i));
  }



}


