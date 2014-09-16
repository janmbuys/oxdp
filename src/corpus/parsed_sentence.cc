#include "corpus/parsed_sentence.h"

namespace oxlm {

ParsedSentence::ParsedSentence():
  TaggedSentence(), 
  arcs_()
  {
  }

ParsedSentence::ParsedSentence(Words tags):
  TaggedSentence(tags), 
  arcs_(tags.size(), -1)
  {
  }

ParsedSentence::ParsedSentence(Words sent, Words tags):
  TaggedSentence(sent, tags), 
  arcs_(tags.size(), -1)
  {
  }

ParsedSentence::ParsedSentence(Words sent, Words tags, Indices arcs):
  TaggedSentence(sent, tags), 
  arcs_(arcs)
  {
  }

ParsedSentence::ParsedSentence(const TaggedSentence& parse):
  TaggedSentence(parse),
  arcs_(parse.size(), -1)
  {
  }

}

