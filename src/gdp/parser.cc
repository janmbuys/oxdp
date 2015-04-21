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

Parser::Parser(const Words& sent):
  ParsedSentence(sent, WordsList(sent.size(), Words())),
  left_children_(),
  right_children_(),
  weight_{0}
{
} 

Parser::Parser(const TaggedSentence& parse):
  ParsedSentence(parse),
  left_children_(parse.size(), Indices()),  
  right_children_(parse.size(), Indices()), 
  weight_{0}
{
} 

}

