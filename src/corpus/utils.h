#ifndef _CORPUS_UTILS_H_
#define _CORPUS_UTILS_H_

#include <string>
#include <vector>
#include <iostream>

namespace oxlm {

typedef std::string Word;
typedef int WordId;
typedef int WordIndex;
typedef std::vector<WordId> Words;
typedef std::vector<WordIndex> Indices; // WxList;
typedef std::vector<Words> WordsList; //Sentences
typedef std::vector<Indices> IndicesList;


}

#endif
