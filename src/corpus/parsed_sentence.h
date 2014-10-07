#ifndef _CORPUS_PARSESENT_H_
#define _CORPUS_PARSESENT_H_

#include "corpus/dict.h"
#include "corpus/tagged_sentence.h"

namespace oxlm {

class ParsedSentence: public TaggedSentence {
  public:

  ParsedSentence();
  
  ParsedSentence(Words tags);

  ParsedSentence(Words sent, Words tags);

  ParsedSentence(Words sent, Words tags, Indices arcs);

  ParsedSentence(const TaggedSentence& parse);

  void print_arcs() const {
    for (auto ind: arcs_)
      std::cout << ind << " ";
    std::cout << std::endl;   
  }

  virtual void set_arc(WordIndex i, WordIndex j) {
    if ((j >=0) && (j < size()))
      arcs_[i] = j;
  }

  virtual void push_arc() {
    arcs_.push_back(-1);
  }

  WordIndex arc_at(WordIndex i) const {
    return arcs_.at(i);
  }

  bool has_arc(WordIndex i, WordIndex j) const {
    return (arcs_[i] == j);
  }

  static bool eq_arcs(const boost::shared_ptr<ParsedSentence>& p1, 
                          const boost::shared_ptr<ParsedSentence>& p2) {
    if (p1->size() != p2->size())
      return false;
    for (int i = 0; i < p1->size(); ++i) 
      if (p1->arc_at(i) != p2->arc_at(i))
        return false;  
    return true;
  }

  bool is_projective_dependency() const {
    for (int i = 0; i < (size() - 1); ++i)
      for (int j = i + 1; (j < size()); ++j)
        if ((arcs_[i]<i &&
              (arcs_[j]<i && arcs_[j]>arcs_[i])) ||
            ((arcs_[i]>i && arcs_[i]>j) &&
              (arcs_[j]<i || arcs_[j]>arcs_[i])) ||
            ((arcs_[i]>i && arcs_[i]<j) &&
              (arcs_[j]>i && arcs_[j]<arcs_[i])))
          return false;
    return true;
  }

  private:
  Words arcs_;

};


}

#endif
