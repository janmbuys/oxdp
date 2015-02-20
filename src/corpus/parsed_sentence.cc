#include "corpus/parsed_sentence.h"

namespace oxlm {

ParsedSentence::ParsedSentence():
  TaggedSentence(), 
  arcs_(),
  labels_()
  {
  }

ParsedSentence::ParsedSentence(WordsList tags):
  TaggedSentence(tags), 
  arcs_(tags.size(), -1),
  labels_(tags.size(), 0)
  {
  }

ParsedSentence::ParsedSentence(Words sent, WordsList tags):
  TaggedSentence(sent, tags), 
  arcs_(tags.size(), -1),
  labels_(tags.size(), 0)
  {
  }

ParsedSentence::ParsedSentence(Words sent, WordsList tags, Indices arcs):
  TaggedSentence(sent, tags), 
  arcs_(arcs),
  labels_(tags.size(), 0)
  {
  }

ParsedSentence::ParsedSentence(Words sent, WordsList tags, Indices arcs, Words labels):
  TaggedSentence(sent, tags), 
  arcs_(arcs),
  labels_(labels) 
  {
  }

ParsedSentence::ParsedSentence(const TaggedSentence& parse):
  TaggedSentence(parse),
  arcs_(parse.size(), -1),
  labels_(parse.size(), 0)
  {
  }

}

