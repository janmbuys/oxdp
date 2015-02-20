#include "corpus/parsed_corpus.h"
#include "corpus/dict.h"

namespace oxlm {

ParsedCorpus::ParsedCorpus(const boost::shared_ptr<ModelConfig>& config):
  sentences_(),
  config_(config)
{
}

void ParsedCorpus::convertWhitespaceDelimitedConllLine(const std::string& line, 
      const boost::shared_ptr<Dict>& dict, Words* sent_out, WordsList* tags_out, Indices* arcs_out, Words* labels_out, bool frozen) {
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
      else if (col_num == 4) //4 - postag (3 - coarse postag)
        tags_out->push_back(Words(1, dict->convertTag(line.substr(last, cur - last - 1), frozen)));
      else if (col_num == 6) //arc head index
        arcs_out->push_back(static_cast<WordIndex>(stoi(line.substr(last, cur - last - 1))));
      else if (col_num == 7) { //label 
        if (config_->labelled_parser)
          labels_out->push_back(dict->convertLabel(line.substr(last, cur - last - 1), frozen));
        else
          labels_out->push_back(dict->convertLabel("<null>", frozen));
      }
       ++col_num;
      state = 0;
    } else {
      if (state == 1) 
        continue;
      last = cur - 1;
      state = 1;
    }
  }
  
  //in case we need to process last column (n):
  /*if ((state == 1) && (col_num == n)) 
    sent_out->push_back(dict->convert(line.substr(last, cur - last), frozen)); */
}

void ParsedCorpus::readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, 
                                bool frozen) {
  Words sent;
  WordsList features;
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
        features.push_back(Words(1, dict->eos())); 
        arcs.push_back(-1);
        labels.push_back(-1);
      }

      sentences_.push_back(ParsedSentence(sent, features, arcs, labels)); 
      state = 1;
    } else {
      if (state==1) {
        //start of sentence
        sent.clear();
        features.clear();
        arcs.clear();
        labels.clear();

        //add start of sentence symbol if defined
        if (dict->sos() != -1) {
          sent.push_back(dict->sos()); 
          features.push_back(Words(1, dict->sos()));
          arcs.push_back(-1);
          labels.push_back(-1);
        }
        state = 0;
      }

      convertWhitespaceDelimitedConllLine(line, dict, &sent, &features, &arcs, &labels, frozen); 
    }
  }

  //update vocab sizes
  config_->vocab_size = dict->size();
  config_->num_tags = dict->tag_size();
  config_->num_labels = dict->label_size();
}

size_t ParsedCorpus::size() const {
  return sentences_.size();
}

size_t ParsedCorpus::numTokens() const {
  size_t total = 0;
  for (auto sent: sentences_)
    total += sent.size();

  return total;
}

std::vector<int> ParsedCorpus::unigramCounts() const {
  std::vector<int> counts(config_->vocab_size, 0);
  for (auto sent: sentences_) {
    for (size_t j = 0; j < sent.size(); ++j)
      counts[sent.word_at(j)] += 1;
  }

  return counts;
}

std::vector<int> ParsedCorpus::actionCounts() const {
  std::vector<int> counts(config_->numActions(), 0);

  for (auto sent: sentences_) {
    int la_count = 0;
    int ra_count = 0;

    for (size_t j = 1; j < sent.size(); ++j) {
      //count left-arcs and right-arcs
      WordId lab = sent.label_at(j);
      if (sent.arc_at(j) < j) {
        ++la_count;
        ++counts[lab + 1];
      } else {
        ++ra_count;
        ++counts[lab + 1 + config_->num_labels];
      }
    }
 
    if (config_->parser_type == ParserType::arceager) {
      counts[0] += sent.size() - 1 - ra_count; //shift
      counts[counts.size() - 1] += sent.size() - 1 - la_count; //reduce
    } else if (config_->parser_type == ParserType::arcstandard) {
      counts[0] += sent.size() - 1;
    }
  }

  return counts;
}

}

