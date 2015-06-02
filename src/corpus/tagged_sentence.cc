#include "corpus/tagged_sentence.h"

namespace oxlm {

TaggedSentence::TaggedSentence():
  Sentence(), 
  tags_(),
  features_()
  {
  }

TaggedSentence::TaggedSentence(WordsList features):
  Sentence(), 
  tags_(),
  features_(features)
  {
  }

TaggedSentence::TaggedSentence(Words sent, Words tags, WordsList features):
  Sentence(sent), 
  tags_(tags),
  features_(features)
  {
  }

TaggedSentence::TaggedSentence(Words sent, Words tags, WordsList features, int id):
  Sentence(sent, id), 
  tags_(tags),
  features_(features)
  {
  }


}

