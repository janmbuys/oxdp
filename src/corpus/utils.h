#ifndef _CORPUS_UTILS_H_
#define _CORPUS_UTILS_H_

#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
    
namespace oxlm {

typedef std::string Word;
typedef int WordId;
typedef int WordIndex;
typedef std::vector<WordId> Words;
typedef std::vector<WordIndex> Indices; 
typedef std::vector<Words> WordsList; 
typedef std::vector<Indices> IndicesList;

typedef double Real;

typedef std::chrono::high_resolution_clock Clock;
typedef Clock::time_point Time;

//time functions

inline Time get_time() {
  return Clock::now();
}

inline float get_duration(const Time& start_time, const Time& stop_time) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() / 1000.0;
}

inline float perplexity(double log_likelihood, size_t corpus_size) {
  return std::exp(log_likelihood / corpus_size);
}

}

#endif
