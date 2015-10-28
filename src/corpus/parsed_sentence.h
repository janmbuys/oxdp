#ifndef _CORPUS_PARSESENT_H_
#define _CORPUS_PARSESENT_H_

#include "corpus/dict.h"
#include "corpus/tagged_sentence.h"

namespace oxlm {

// Represents a parsed sentence.  
class ParsedSentence : public TaggedSentence {
 public:
  ParsedSentence();

  ParsedSentence(WordsList features);

  ParsedSentence(Words sent, Words tags, WordsList features);

  ParsedSentence(Words sent, Words tags, WordsList features, Indices arcs,
                 int id);

  ParsedSentence(Words sent, Words tags, WordsList features, Indices arcs,
                 Words labels, int id);

  ParsedSentence(const TaggedSentence& parse);

  void print_arcs() const {
    for (auto ind : arcs_) { 
      std::cout << ind << " ";
    }
    std::cout << std::endl;
  }

  std::string arcs_string() const {
    std::string arcs = "";
    for (auto ind : arcs_) {
      arcs += ind + " ";
    }
    return arcs;
  }

  void print_labels() const {
    for (auto lab : labels_) {
      std::cout << lab << " ";
    }
    std::cout << std::endl;
  }

  void print_labels(const boost::shared_ptr<Dict>& dict) const {
    for (auto lab : labels_) {
      std::cout << dict->lookupLabel(lab) << " ";
    }
    std::cout << std::endl;
  }

  virtual void set_arc(WordIndex i, WordIndex j) {
    if ((j >= 0) && (j < size())) arcs_[i] = j;
  }

  void set_label(WordIndex i, WordId l) { labels_[i] = l; }

  virtual void push_arc() {
    arcs_.push_back(-1);
    labels_.push_back(-1);
  }

  WordIndex arc_at(WordIndex i) const {
    if (i >= 0) {
      return arcs_.at(i);
    } else {
      return -1;
    }
  }

  WordId label_at(WordIndex i) const {
    if (i >= 0) {
      return labels_.at(i);
    } else {
      return 0;
    }
  }

  bool has_arc(WordIndex i, WordIndex j) const {
    if (i >= 0 && j >= 0) {
      return (arcs_[i] == j);
    } else {
      return false;
    }
  }

  static bool eq_arcs(const boost::shared_ptr<ParsedSentence>& p1,
                      const boost::shared_ptr<ParsedSentence>& p2) {
    if (p1->size() != p2->size()) return false;
    for (int i = 1; i < p1->size(); ++i) {
      if (p1->arc_at(i) != p2->arc_at(i)) return false;
    }
    return true;
  }

  static bool eq_lab_arcs(const boost::shared_ptr<ParsedSentence>& p1,
                          const boost::shared_ptr<ParsedSentence>& p2) {
    if (p1->size() != p2->size()) return false;
    for (int i = 1; i < p1->size(); ++i) {
      if (p1->arc_at(i) != p2->arc_at(i)) return false;
      if (p1->label_at(i) != p2->label_at(i)) return false;
    }
    return true;
  }

  bool projective_dependency() const {
    for (int i = 0; i < (size() - 1); ++i) {
      for (int j = i + 1; (j < size()); ++j) {
        if ((arcs_[i] < i && (arcs_[j] < i && arcs_[j] > arcs_[i])) ||
            ((arcs_[i] > i && arcs_[i] > j) &&
             (arcs_[j] < i || arcs_[j] > arcs_[i])) ||
            ((arcs_[i] > i && arcs_[i] < j) &&
             (arcs_[j] > i && arcs_[j] < arcs_[i]))) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  Indices arcs_;
  Words labels_;
};

}  // namespace oxlm

#endif
