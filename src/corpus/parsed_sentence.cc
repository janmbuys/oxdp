#include "corpus/parsed_sentence.h"

namespace oxlm {

ParsedSentence::ParsedSentence():
  TaggedSentence(), 
  arcs_(),
  labels_(),
  id_(0)
  {
  }

ParsedSentence::ParsedSentence(WordsList features):
  TaggedSentence(features), 
  arcs_(features.size(), -1),
  labels_(features.size(), 0),
  id_(0)
  {
  }

ParsedSentence::ParsedSentence(Words sent, Words tags, WordsList features):
  TaggedSentence(sent, tags, features), 
  arcs_(features.size(), -1),
  labels_(features.size(), 0),
  id_(0)
  {
  }

ParsedSentence::ParsedSentence(Words sent, Words tags, WordsList features, Indices arcs, int id):
  TaggedSentence(sent, tags, features), 
  arcs_(arcs),
  labels_(features.size(), 0),
  id_(id)
  {
  }

ParsedSentence::ParsedSentence(Words sent, Words tags, WordsList features, Indices arcs, Words labels, int id):
  TaggedSentence(sent, tags, features), 
  arcs_(arcs),
  labels_(labels),
  id_(id) 
  {
  }

ParsedSentence::ParsedSentence(const TaggedSentence& parse):
  TaggedSentence(parse),
  arcs_(parse.size(), -1),
  labels_(parse.size(), 0)
  {
  }

}

