#include "corpus/tagged_sentence.h"

namespace oxlm {

TaggedSentence::TaggedSentence():
  Sentence(), 
  tags_()
  {
  }

TaggedSentence::TaggedSentence(Words tags):
  Sentence(), 
  tags_(tags)
  {
  }

TaggedSentence::TaggedSentence(Words sent, Words tags):
  Sentence(sent), 
  tags_(tags)
  {
  }


}

