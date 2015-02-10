#include "parser.h"

namespace oxlm {

//constructor used for generation
Parser::Parser():
  ParsedSentence(), 
  left_children_(),
  right_children_(),
  weight_{0}
{
  push_arc();
}

Parser::Parser(Words tags):
  ParsedSentence(tags), 
  left_children_(tags.size(), Indices()),  
  right_children_(tags.size(), Indices()),  
  weight_{0}
{
}

Parser::Parser(Words sent, Words tags):
  ParsedSentence(sent, tags), 
  left_children_(tags.size(), Indices()),  
  right_children_(tags.size(), Indices()), 
  weight_{0}
{
}

Parser::Parser(Words sent, Words tags, Indices arcs):
  ParsedSentence(sent, tags), 
  left_children_(tags.size(), Indices()),  
  right_children_(tags.size(), Indices()),
  weight_{0}
{
  for (int i = 0; i < size(); ++i) 
    set_arc(i, arcs.at(i));
}

Parser::Parser(Words sent, Words tags, Indices arcs, Words labels):
  ParsedSentence(sent, tags), 
  left_children_(tags.size(), Indices()),  
  right_children_(tags.size(), Indices()),
  weight_{0}
{
  for (int i = 0; i < size(); ++i) {
    set_arc(i, arcs.at(i));
    set_label(i, labels.at(i));
  }
}


Parser::Parser(const TaggedSentence& parse):
  ParsedSentence(parse),
  left_children_(parse.size(), Indices()),  
  right_children_(parse.size(), Indices()), 
  weight_{0}
{
} 

}

