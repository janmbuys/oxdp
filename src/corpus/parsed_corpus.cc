#include "corpus/parsed_corpus.h"
#include "corpus/dict.h"

namespace oxlm {

ParsedCorpus::ParsedCorpus(const boost::shared_ptr<ModelConfig>& config):
  sentences_(),
  config_(config)
{
}

void ParsedCorpus::convertWhitespaceDelimitedConllLine(const std::string& line, 
      const boost::shared_ptr<Dict>& dict, Words* sent_out, Words* tags_out, WordsList* features_out, Indices* arcs_out, Words* labels_out, bool frozen) {
  size_t cur = 0;
  size_t last = 0;
  int state = 0;
  int col_num = 0;
  
  //std::string word_str; 
  //std::string feature_str;
  //std::string tag_str; 
  WordId word_id = -1;
  Words features;
  if (config_->pyp_model || config_->predict_pos)
    features.push_back(-1); //placeholder for tag

  while(cur < line.size()) {
    if (Dict::is_ws(line[cur++])) {
      if (state == 0) 
        continue;
      if (col_num == 1) { //1 - annotated word
        if (config_->lexicalised || config_->pyp_model) {
          std::string word_str = line.substr(last, cur - last - 1);
          word_id = dict->convert(word_str, frozen);
          sent_out->push_back(word_id);
        }
      } else if (col_num == 2) { //2 - unannotated word 
        std::string word_str = line.substr(last, cur - last - 1);
        if (!config_->lexicalised && !config_->pyp_model) {
          word_id = dict->convert(word_str, frozen);
          sent_out->push_back(word_id);
        }
        if (!config_->pyp_model && config_->lexicalised) {
          //split stem as feature if present
          std::stringstream feature_stream(word_str);
          std::string feat;
          while (std::getline(feature_stream, feat, '|')) {
            if (!feat.empty()) 
              features.push_back(dict->convertFeature(feat, frozen));
          }
        } 
      } else if (col_num == 3) { //3 - coarse postag
        if (config_->morph_features)
          features.push_back(dict->convertFeature(line.substr(last, cur - last - 1), frozen));
      } else if (col_num == 4) { //4 - postag 
          std::string tag_str = line.substr(last, cur - last - 1);
        tags_out->push_back(dict->convertTag(tag_str, frozen));
        if (config_->pyp_model || config_->predict_pos)
          features[0] = dict->convertFeature(tag_str, frozen);
      } else if (col_num == 5) { //5 morphological features (| seperated)
          std::string feature_str = line.substr(last, cur - last -1); 
        if (config_->morph_features) {
          std::stringstream feature_stream(feature_str);
          std::string feat;
          while (std::getline(feature_stream, feat, '|')) {
            if (!feat.empty()) 
              features.push_back(dict->convertFeature(feat, frozen));
          }
        }
      } else if (col_num == 6) { //arc head index
        arcs_out->push_back(static_cast<WordIndex>(stoi(line.substr(last, cur - last - 1))));
      } else if (col_num == 7) { //label 
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

  if (word_id >= 0) {
    features_out->push_back(features);
  }
}

void ParsedCorpus::readFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, 
                                bool frozen) {
  Words sent;
  Words tags;
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
        tags.push_back(dict->eos()); 
        if (!config_->pyp_model && config_->lexicalised && config_->predict_pos) 
          features.push_back(Words(2, dict->eos()));
        else
          features.push_back(Words(1, dict->eos()));
        arcs.push_back(-1);
        labels.push_back(-1);
      }

      //word word-to-feature mapping
      for (unsigned i = 0; i < sent.size(); ++i) {
        dict->setWordFeatures(sent[i], features[i]);
      }


      int index = sentences_.size() + 1;
      if (frozen)
        index = 0;

      sentences_.push_back(ParsedSentence(sent, tags, features, arcs, labels, index)); 
      state = 1;
    } else {
      if (state==1) {
        //start of sentence
        sent.clear();
        tags.clear();
        features.clear();
        arcs.clear();
        labels.clear();

        //add start of sentence symbol if defined
        if (dict->sos() != -1) {
          sent.push_back(dict->sos()); 
          tags.push_back(dict->sos()); 
          if (!config_->pyp_model && config_->lexicalised && config_->predict_pos) 
            features.push_back(Words(2, dict->sos()));
          else
            features.push_back(Words(1, dict->sos()));
          arcs.push_back(-1);
          labels.push_back(-1);
        }
        state = 0;
      }

      convertWhitespaceDelimitedConllLine(line, dict, &sent, &tags, &features, &arcs, &labels, frozen); 
    }
  }

  //update vocab sizes
  config_->vocab_size = dict->size();
  config_->num_tags = dict->tag_size();
  config_->num_features = dict->feature_size();
  config_->num_labels = dict->label_size();
}

Words ParsedCorpus::convertWhitespaceDelimitedTxtLine(const std::string& line, 
        const boost::shared_ptr<Dict>& dict, bool frozen) {
  Words out;

  size_t cur = 0;
  size_t last = 0;
  int state = 0;
     
  //add start of sentence symbol if defined
  if (dict->sos() != -1)
    out.push_back(dict->sos()); 

  while (cur < line.size()) {
    if (Dict::is_ws(line[cur++])) {
      if (state == 0) 
        continue;
      out.push_back(dict->convert(line.substr(last, cur - last - 1), frozen));
      state = 0;
    } else {
      if (state == 1) 
        continue;
      last = cur - 1;
      state = 1;
    }
  }

  if (state == 1)
    out.push_back(dict->convert(line.substr(last, cur - last), frozen));

  //add end of sentence symbol if defined 
  if (dict->eos() != -1) 
    out.push_back(dict->eos()); 

  return out;
}

void ParsedCorpus::readTxtFile(const std::string& filename, const boost::shared_ptr<Dict>& dict, bool frozen) {
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;

  WordsList features;
  Words tags;
  Indices arcs;
  Words labels;

  while(getline(in, line)) {
    //start of sentence
    features.clear();
    tags.clear();
    arcs.clear();
    labels.clear();

    Words sent = convertWhitespaceDelimitedTxtLine(line, dict, frozen);

    for (unsigned i = 0; i < sent.size(); ++i) {
      features.push_back(Words(1, 0));
      tags.push_back(1);
      arcs.push_back(-1);
      labels.push_back(-1);
    }
 
    int index = sentences_.size() + 1;
    if (frozen)
      index = 0;

    sentences_.push_back(ParsedSentence(sent, tags, features, arcs, labels, index)); 
  }

  config_->vocab_size = dict->size();
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

std::vector<int> ParsedCorpus::tagCounts() const {
  std::vector<int> counts(config_->num_tags, 0);
  for (auto sent: sentences_) {
    for (size_t j = 0; j < sent.size(); ++j)
      counts[sent.tag_at(j)] += 1;
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
    } else if (config_->parser_type == ParserType::arcstandard || config_->parser_type == ParserType::arcstandard2) {
      counts[0] += sent.size() - 1;
    }
  }

  return counts;
}

}

