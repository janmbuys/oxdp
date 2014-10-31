#include "corpus/parsed_corpus.h"
#include "corpus/dict.h"

namespace oxlm {

ParsedCorpus::ParsedCorpus():
  sentences_()
{
}

void ParsedCorpus::convertWhitespaceDelimitedConllLine(const std::string& line, 
      const boost::shared_ptr<Dict>& dict, Words* sent_out, Words* tags_out, Indices* arcs_out, Words* labels_out, bool frozen) {
  size_t cur = 0;
  size_t last = 0;
  int state = 0;
  int col_num = 0;
   
  while(cur < line.size()) {
    if (Dict::is_ws(line[cur++])) {
      if (state == 0) 
        continue;
      if (col_num == 1) //1 - word
        sent_out->push_back(dict->convert(line.substr(last, cur - last - 1), frozen));
      else if (col_num == 4) //4 - postag (3 - course postag)
        tags_out->push_back(dict->convertTag(line.substr(last, cur - last - 1), frozen));
      else if (col_num == 6) //arc head
        arcs_out->push_back(static_cast<WordIndex>(stoi(line.substr(last, cur - last - 1))));
      else if (col_num == 7) //label 
        labels_out->push_back(dict->convertLabel(line.substr(last, cur - last - 1), frozen));
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
    sent_out->push_back(dict->convert(line.substr(last, cur - last), frozen));
}

void ParsedCorpus::readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, 
                                bool frozen) {
  Words sent;
  Words tags;
  Indices arcs;
  Words labels;
 
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  int state = 1; //have to add new vector

  while(getline(in, line)) {
    //token level

    if (line=="") { 
      //end of sentence
        
      //add end of sentence symbol if defined
      if (dict->eos() != -1) {
        sent.push_back(dict->eos()); 
        tags.push_back(dict->eos()); 
        arcs.push_back(-1);
        labels.push_back(-1);
      }

      //could make storing the label optional
      sentences_.push_back(ParsedSentence(sent, tags, arcs, labels)); 
      //add arcs seperately
      //for (unsigned i = 1; i < sent.size(); ++i)
      //  sentences_.back().set_arc(i, arcs.at(i));
      //sentences_.back().print_arcs();

      state = 1;
    } else {
      if (state==1) {
        //start of sentence
        sent.clear();
        tags.clear();
        arcs.clear();
        labels.clear();

        //add start of sentence symbol
        sent.push_back(dict->sos()); 
        tags.push_back(dict->sos());
        arcs.push_back(-1);
        labels.push_back(-1);
        state = 0;
      }

      convertWhitespaceDelimitedConllLine(line, dict, &sent, &tags, &arcs, &labels, frozen); 
      //std::cout << labels[1] << " ";
    }
  }

  vocab_size_ = dict->size();
  //std::cout << dict->size() << " dict labels: " << dict->label_size() << std::endl;
  num_labels_ = dict->label_size();
}

size_t ParsedCorpus::size() const {
  return sentences_.size();
}

size_t ParsedCorpus::numTokens() const {
  size_t total = 0;
  for (auto sent: sentences_)
    total += sent.size() - 1;

  return total;
}

std::vector<int> ParsedCorpus::unigramCounts() const {
  std::vector<int> counts(vocab_size_, 0);
  for (auto sent: sentences_) {
    for (size_t j = 0; j < sent.size(); ++j)
      counts[sent.word_at(j)] += 1;
  }

  return counts;
}

std::vector<int> ParsedCorpus::actionCounts() const {
  //this is labelled, for arc-standard, but can't really compute it directly for arc-eager
    
  std::vector<int> counts(num_labels_*2+1, 0);
  for (auto sent: sentences_) {
    for (size_t j = 1; j < sent.size(); ++j) {
      WordId lab = sent.label_at(j);
      if (sent.arc_at(j) < j) {
        //left arc
        ++counts[lab+1];
      } else {
        //right arc 
        ++counts[lab+num_labels_ + 1];
      }
        
    }
    counts[0] += sent.size() - 1;
  }

  return counts;
}

}

