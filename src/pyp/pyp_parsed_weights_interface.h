#ifndef _PYP_PARWEIGHTS_I_H_ 
#define _PYP_PARWEIGHTS_I_H_ 

#include "corpus/parsed_weights_interface.h"
#include "pyp/pyp_weights_interface.h"

namespace oxlm {

class PypParsedWeightsInterface: public ParsedWeightsInterface, public PypWeightsInterface {

  public:
  virtual double wordLikelihood() const;
    
  virtual double tagLikelihood() const;
  
  virtual double actionLikelihood() const;

  virtual ~PypParsedWeightsInterface() {}
};

}
#endif
