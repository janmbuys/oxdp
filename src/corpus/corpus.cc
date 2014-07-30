#include "corpus.h" 

namespace oxlm {

void Dict::convertWhitespaceDelimitedLine(const std::string& line, std::vector<WordId>* out, bool frozen) {
  size_t cur = 0;
  size_t last = 0;
  int state = 0;
  out->clear();
  while(cur < line.size()) {
    if (is_ws(line[cur++])) {
      if (state == 0) 
        continue;
      out->push_back(convert(line.substr(last, cur - last - 1), frozen));
      state = 0;
    } else {
      if (state == 1) 
        continue;
      last = cur - 1;
      state = 1;
    }
  }

  if (state == 1)
    out->push_back(convert(line.substr(last, cur - last), frozen));
}

void Dict::convertWhitespaceDelimitedConllLine(const std::string& line, std::vector<WordId>* sent_out, std::vector<WordId>* tag_out, std::vector<WordIndex>* dep_out, bool frozen) {
  size_t cur = 0;
  size_t last = 0;
  int state = 0;
  int col_num = 0;
   
  while(cur < line.size()) {
    if (is_ws(line[cur++])) {
      if (state == 0) 
        continue;
      if (col_num == 1) //1 - word
        sent_out->push_back(convert(line.substr(last, cur - last - 1), frozen));
      else if (col_num == 4) //4 - postag (3 - course postag)
        tag_out->push_back(convertTag(line.substr(last, cur - last - 1), frozen));
      else if (col_num == 6)
        dep_out->push_back(static_cast<WordIndex>(stoi(line.substr(last, cur - last - 1))));
      ++col_num;
      state = 0;
    } else {
      if (state == 1) 
        continue;
      last = cur - 1;
      state = 1;
    }
  }

  if ((state == 1) && (col_num == 1)) //use relavent if we need last column
    sent_out->push_back(convert(line.substr(last, cur - last), frozen));
}

//leave as is for compatability
void Dict::readFromFile(const std::string& filename, std::vector<std::vector<WordId> >* src, std::set<WordId>* src_vocab, bool frozen) {
  src->clear();
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  int lc = 0;
  while(getline(in, line)) {
    ++lc;
    src->push_back(std::vector<WordId>());
    convertWhitespaceDelimitedLine(line, &src->back(), frozen);
    for (WordId i = 0; i < static_cast<WordId>(src->back().size()); ++i) 
      src_vocab->insert(src->back()[i]);
  }
}

void Dict::readFromConllFile(const std::string& filename, std::vector<std::vector<WordId> >* sents, std::vector<std::vector<WordId> >* tags, std::vector<std::vector<WordIndex> >* deps, bool frozen) {
  sents->clear();
  tags->clear();
  deps->clear();
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  int lc = 0;
  int state = 1; //have to add new vector

  while(getline(in, line)) {
    ++lc;

    if (line=="") { 
      state = 1;
    } else {
      if (state==1) {
        //add start of sentence symbol
        sents->push_back(std::vector<WordId>());
        sents->back().push_back(0); 
        tags->push_back(std::vector<WordId>());
        tags->back().push_back(0);
        deps->push_back(std::vector<WordIndex>());
        deps->back().push_back(-1);
        state = 0;
      }

      convertWhitespaceDelimitedConllLine(line, &sents->back(), &tags->back(), &deps->back(), frozen);        
    }
  } 
}

WordId Dict::convert(const Word& word, bool frozen) {
  
  if (words_.size()==0 && !frozen) {
    words_.push_back(word);
    d_[word] = 0;
    return 0;
  } else if (word == sos_) {
    return 0;
  }

  Word lword(word);
  //convert to lower case 
  //std::transform(lword.begin(), lword.end(), lword.begin(), tolower);

  auto i = d_.find(lword);
  if (i == d_.end()) {
    if (frozen) 
      return bad0_id_;
    //if already a singleton, add to main dictionary, else add as singleton
    //may disable for some cases - TODO add boolean
    auto i = single_d_.find(lword);
    if (i == single_d_.end()) {
      single_d_[lword] = -1;
      return bad0_id_;
    } 

    words_.push_back(lword);
    d_[lword] = words_.size()-1;
    return words_.size()-1;
  } else {
    return i->second;
  }
}

WordId Dict::convertTag(const Word& tag, bool frozen) {
  auto i = tag_d_.find(tag);
  if (i == tag_d_.end()) {
    if (frozen) 
      return bad0_id_;
    tags_.push_back(tag);
    tag_d_[tag] = tags_.size()-1;
    return tags_.size()-1;
  } else {
    return i->second;
  }
}

Word Dict::lookup(WordId id) const {
  if (!valid(id)) 
    return b0_;
  return words_[id];
}

Word Dict::lookupTag(WordId id) const {
  if (!valid(id)) 
    return b0_;
  return tags_[id];
}

}
