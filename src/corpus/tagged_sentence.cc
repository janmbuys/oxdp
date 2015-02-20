#include "corpus/tagged_sentence.h"

namespace oxlm {

TaggedSentence::TaggedSentence():
  Sentence(), 
  features_()
  {
  }

TaggedSentence::TaggedSentence(WordsList features):
  Sentence(), 
  features_(features)
  {
  }

TaggedSentence::TaggedSentence(Words sent, WordsList features):
  Sentence(sent), 
  features_(features)
  {
  }

}

