#ifndef _PYP_PARWEIGHTS_I_H_ 
#define _PYP_PARWEIGHTS_I_H_ 

#include "corpus/parsed_weights_interface.h"
#include "pyp/pyp_weights_interface.h"

namespace oxlm {

class PypParsedWeightsInterface: public ParsedWeightsInterface, public PypWeightsInterface {

  public:
  virtual double wordLikelihood() const = 0;
    
  virtual double tagLikelihood() const = 0;
  
  virtual double actionLikelihood() const = 0;

  virtual ~PypParsedWeightsInterface() {}
};

}
#endif
