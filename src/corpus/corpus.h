#ifndef _PYPDICT_H_
#define _PYPDICT_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>

#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/functional/hash.hpp>

namespace oxlm {

typedef std::string Word;
typedef int WordId;
typedef int WordIndex;
typedef std::vector<WordId> Words;

class Dict {
// typedef std::unordered_map<std::string, WordId, std::hash<std::string> > Map;
// typedef std::map<std::string, WordId> Map;
public:
    Dict() : b0_("<bad0>"), 
             sos_("<s>"), 
             eos_("</s>"), 
             bad0_id_(-1) 
    {
        words_.reserve(1000);
        Convert(sos_);
        Convert(eos_);
    }

    Dict(Word sos, Word eos) : b0_("<bad0>"), 
                               sos_(sos), 
                               eos_(eos), 
                               bad0_id_(-1)
    {
        words_.reserve(1000);
        Convert(sos_);
        ConvertPOS(sos_);
        if (eos!="")
            Convert(eos_);
        //should probably move to the outside sometime
        //std::vector<Word> action_words = {"sh", "la", "ra", "re", "la2", "ra2"};
        //if (has_actions)
        //    for (auto a: action_words)
        //        ConvertAction(a);
    }
    
    inline WordId min() const {
        return 0;
    }
    inline WordId max() const {
        return words_.size()-1;
    }
    inline size_t size() const {
        return words_.size();
    }
    inline size_t pos_size() const {
        return pos_.size();
    }

    static bool is_ws(char x) {
        return (x == ' ' || x == '\t');
    }

    inline void ConvertWhitespaceDelimitedLine(const std::string& line, std::vector<WordId>* out) {
        size_t cur = 0;
        size_t last = 0;
        int state = 0;
        out->clear();
        while(cur < line.size()) {
            if (is_ws(line[cur++])) {
                if (state == 0) continue;
                out->push_back(Convert(line.substr(last, cur - last - 1)));
                state = 0;
            } else {
                if (state == 1) continue;
                last = cur - 1;
                state = 1;
            }
        }
        if (state == 1)
            out->push_back(Convert(line.substr(last, cur - last)));
    }

    inline void ConvertWhitespaceDelimitedConllLine(const std::string& line, std::vector<WordId>* sent_out, std::vector<WordId>* pos_out, std::vector<WordIndex>* dep_out, bool frozen) {
        size_t cur = 0;
        size_t last = 0;
        int state = 0;
        int col_num = 0;
        
        while(cur < line.size()) {
            //std::cerr << "cur " << cur << std::endl;
            if (is_ws(line[cur++])) {
                if (state == 0) continue;
                if (col_num == 1)
                  sent_out->push_back(Convert(line.substr(last, cur - last - 1), frozen));
                else if (col_num == 4)
                  pos_out->push_back(ConvertPOS(line.substr(last, cur - last - 1), frozen));
                else if (col_num == 6)
                  dep_out->push_back(static_cast<WordIndex>(stoi(line.substr(last, cur - last - 1))));
                ++col_num;
                state = 0;
            } else {
                if (state == 1) continue;
                last = cur - 1;
                state = 1;
            }
        }

        if ((state == 1) && (col_num == 1)) //use only if we need last column
            sent_out->push_back(Convert(line.substr(last, cur - last), frozen));
    }

    inline void ConvertWhitespaceDelimitedActionLine(const std::string& line, std::vector<WordId>* out) {
        size_t cur = 0;
        size_t last = 0;
        int state = 0;
        out->clear();
        //for first word
        while(is_ws(line[cur++])) continue;
        last = cur - 1;
        state = 1;
        while(!is_ws(line[cur++])) continue;
        out->push_back(ConvertAction(line.substr(last, cur - last - 1)));
        state = 0;

        //remaining words
        while(cur < line.size()) {
            if (is_ws(line[cur++])) {
                if (state == 0) continue;
                out->push_back(Convert(line.substr(last, cur - last - 1)));
                state = 0;
            } else {
                if (state == 1) continue;
                last = cur - 1;
                state = 1;
            }
        }
        if (state == 1)
            out->push_back(Convert(line.substr(last, cur - last)));
    }

    inline WordId Lookup(const Word& word) const {
        auto i = d_.find(word);
        if (i == d_.end()) return bad0_id_;
        return i->second;
    }

    inline WordId Convert(const Word& word, bool frozen = false) {
        if (word == "ROOT") {
          words_.push_back(word);
          d_[word] = 0;
          return 0;
        }

        //convert to lower case 
        Word lword(word);
        std::transform(lword.begin(), lword.end(), lword.begin(), tolower);

        auto i = d_.find(lword);
        if (i == d_.end()) {
            if (frozen)
                return bad0_id_;
            //if already a singleton, add to main dictionary, else add as singleton
            auto i = sd_.find(lword);
            if (i == sd_.end()) {
                sd_[lword] = -1;
                return bad0_id_;
            } 

            words_.push_back(lword);
            d_[lword] = words_.size()-1;
            return words_.size()-1;
        } else {
            return i->second;
        }
    }

    inline WordId ConvertPOS(const Word& tag, bool frozen = false) {
        auto i = pd_.find(tag);
        if (i == pd_.end()) {
            if (frozen)
                return bad0_id_;
            pos_.push_back(tag);
            pd_[tag] = pos_.size()-1;
            return pos_.size()-1;
        } else {
            return i->second;
        }
    }

    inline WordId ConvertAction(const Word& word, bool frozen = false) {
        auto i = ad_.find(word);
        if (i == ad_.end()) {
            if (frozen)
                return bad0_id_;
            a_words_.push_back(word);
            ad_[word] = a_words_.size()-1;
            return a_words_.size()-1;
        } else {
            return i->second;
        }
    }

    inline const std::vector<Word> getVocab() const {
        return words_;
    }

    inline bool valid(const WordId id) const {
        return id >= 0;
    }

    inline const Word& Convert(const WordId id) const {
        if (!valid(id)) return b0_;
        return words_[id];
    }

    inline const Word& ConvertPOS(const WordId id) const {
        if (!valid(id)) return b0_;
        return pos_[id];
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & b0_;
        ar & sos_;
        ar & eos_;
        ar & bad0_id_;
        ar & words_;
        ar & d_;
    }
private:
    Word b0_, sos_, eos_;
    WordId bad0_id_;
    std::vector<Word> words_;
    std::map<std::string, WordId> d_;
    std::map<std::string, WordId> sd_;
    std::vector<Word> pos_;
    std::map<std::string, WordId> pd_;
    std::vector<Word> a_words_;
    std::map<std::string, WordId> ad_;
};

inline void ReadFromFile(const std::string& filename,
                         Dict* d,
                         std::vector<std::vector<WordId> >* src,
                         std::set<WordId>* src_vocab) {
    src->clear();
    std::cerr << "Reading from " << filename << std::endl;
    std::ifstream in(filename);
    assert(in);
    std::string line;
    int lc = 0;
    while(getline(in, line)) {
        ++lc;
        src->push_back(std::vector<WordId>());
        d->ConvertWhitespaceDelimitedLine(line, &src->back());
        for (WordId i = 0; i < static_cast<WordId>(src->back().size()); ++i) 
            src_vocab->insert(src->back()[i]);
    }
}

inline void ReadFromConllFile(const std::string& filename,
                         Dict* d,
                         std::vector<std::vector<WordId> >* sents,
                         std::vector<std::vector<WordId> >* ptags,
                         std::vector<std::vector<WordIndex> >* deps,
                         bool frozen) {
    sents->clear();
    ptags->clear();
    deps->clear();
    std::cerr << "Reading from " << filename << std::endl;
    std::ifstream in(filename);
    assert(in);
    std::string line;
    int lc = 0;
    int state = 1; //have to add new vector

    while(getline(in, line)) {
        ++lc;

        //std::cerr << "line " << lc << std::endl;
        if (line=="") { 
            //add to vocab
            //for (WordId i = 0; i < static_cast<WordId>(sents->back().size()); ++i) 
            //    sent_vocab->insert(sents->back()[i]);
            state = 1;
            //std::cerr << "new line " << lc << std::endl;
        } else {
            if (state==1) {
                sents->push_back(std::vector<WordId>());
                sents->back().push_back(0); //add ROOT
                ptags->push_back(std::vector<WordId>());
                ptags->back().push_back(0);
                deps->push_back(std::vector<WordIndex>());
                deps->back().push_back(-1);
                state = 0;
            }

            d->ConvertWhitespaceDelimitedConllLine(line, &sents->back(), &ptags->back(), &deps->back(), frozen);        
            //sent_vocab->insert(sents->back().back()); //add word to vocab
        }
    } 

    //std::cerr << "done reading conll" << std::endl;
}

inline void ConvertWhitespaceDelimitedDependencyLine(const std::string& line, std::vector<WordIndex>* out) {
    size_t cur = 0;
    size_t last = 0;
    int state = 0;
    out->clear();
    while(cur < line.size()) {
        if (line[cur++]==' ') {
            if (state == 0) continue;
            out->push_back(static_cast<WordIndex>(stoi(line.substr(last, cur - last - 1))));
            state = 0;
        } else {
            if (state == 1) continue;
            last = cur - 1;
            state = 1;
        }
    }
    if (state == 1)
        out->push_back(static_cast<WordIndex>(stoi(line.substr(last, cur - last))));
}

inline void ReadFromDependencyFile(const std::string& filename,
                         std::vector<std::vector<WordIndex>>* src) {
    src->clear();
    std::cerr << "Reading from " << filename << std::endl;
    std::ifstream in(filename);
    assert(in);
    std::string line;
    int lc = 0;
    while(getline(in, line)) {
        ++lc;
        src->push_back(std::vector<WordId>());
        ConvertWhitespaceDelimitedDependencyLine(line, &src->back());
    }
}

inline void ReadFromActionFile(const std::string& filename,
                         Dict* d,
                         std::vector<std::vector<WordId> >* src,
                         std::set<WordId>* src_vocab) {
    src->clear();
    std::cerr << "Reading from " << filename << std::endl;
    std::ifstream in(filename);
    assert(in);
    std::string line;
    int lc = 0;
    while(getline(in, line)) {
        ++lc;
        src->push_back(std::vector<WordId>());
        // treat first word (action) differently
        d->ConvertWhitespaceDelimitedActionLine(line, &src->back());
        //TODO need to make sure that first word is skipped
        for (WordId i = 1; i < static_cast<WordId>(src->back().size()); ++i) {
            //std::cout << src->back()[i] << " ";
            src_vocab->insert(src->back()[i]);
        }
        //std::cout << std::endl;
    }
}

template <typename Container>
struct container_hash {
    std::size_t operator()(Container const& c) const {
        return boost::hash_range(c.begin(), c.end());
    }
};

}


namespace std {
template<typename S, typename T> struct hash<pair<S, T>> {
    inline size_t operator()(const pair<S, T> & v) const {
        size_t seed = 0;
        boost::hash_combine(seed, v.first);
        boost::hash_combine(seed, v.second);
        return seed;
    }
};
}


#endif // PYPDICT_H_
