#ifndef _GDP_PARSE_MODEL_H_
#define _GDP_PARSE_MODEL_H_

#include <cstdlib>
#include <memory>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "utils/random.h"
#include "corpus/parsed_weights_interface.h"

namespace oxlm {

class ParseModel {
  public:
  ParseModel(unsigned size);

  protected:
  MT19937 eng;
  unsigned beam_size;  

};

}

#endif
