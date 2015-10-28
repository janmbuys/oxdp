#include "corpus/sentence.h"

namespace oxlm {

Sentence::Sentence() : sentence_(), id_(0) {}

Sentence::Sentence(Words sent) : sentence_(sent), id_(0) {}

Sentence::Sentence(Words sent, int id) : sentence_(sent), id_(id) {}

}  // namespace oxlm

