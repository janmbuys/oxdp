#ifndef _GDP_PARSE_MODEL_H_
#define _GDP_PARSE_MODEL_H_

#include <cstdlib>
#include <memory>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "utils/random.h"
#include "corpus/parsed_weights_interface.h"

/*namespace std {

template<typename T, typename... Args>
inline unique_ptr<T> make_unique(Args&&... args) {
  return unique_ptr<T>{new T{args...}};
} 

} */

namespace oxlm {

class ParseModel {
  public:
  ParseModel(unsigned size);

  protected:
  MT19937 eng;
  unsigned beam_size;  

};

//see if this works here
/*
template<class Parser>
bool cmp_particle_weights(const std::unique_ptr<Parser>& p1, 
                          const std::unique_ptr<Parser>& p2) {
  //null should be the biggest
  if (p1 == nullptr)
    return false;
  else if (p2 == nullptr)
    return true;
  else
    return (p1->particle_weight() < p2->particle_weight());
}

template<class Parser>
bool cmp_reduce_particle_weights(const std::unique_ptr<Parser>& p1, 
                                 const std::unique_ptr<Parser>& p2) {
  //null should be the biggest
  if (p1 == nullptr)
    return false;
  else if (p2 == nullptr)
    return true;
  //then those that cannot reduce
  else if (!p1->reduce_valid())
    return false;
  else if (!p2->reduce_valid())
    return true;
  else
    return (p1->particle_weight() < p2->particle_weight());
}

template<class Parser>
bool cmp_weighted_importance_weights(const std::unique_ptr<Parser>& p1, 
                                     const std::unique_ptr<Parser>& p2) {
  //null or no particles should be the biggest
  if ((p1 == nullptr) || (p1->num_particles() == 0))
    return false;
  else if ((p2 == nullptr) || (p2->num_particles() == 0))
    return true;
  else
    return (p1->weighted_importance_weight() < p2->weighted_importance_weight());
}
 */
}

#endif
