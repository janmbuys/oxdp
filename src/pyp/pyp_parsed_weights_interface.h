#ifndef _PYP_PARWEIGHTS_I_H_ 
#define _PYP_PARWEIGHTS_I_H_ 

#include "corpus/parsed_weights_interface.h"
#include "corpus/parse_data_set.h"
#include "pyp/pyp_weights_interface.h"

namespace oxlm {

class PypParsedWeightsInterface: public ParsedWeightsInterface, public PypWeightsInterface {

  public:
  virtual Real wordLikelihood() const = 0;
    
  virtual Real tagLikelihood() const = 0;
  
  virtual Real actionLikelihood() const = 0;

  virtual void updateInsert(const ParseDataSet& examples, MT19937& eng) = 0;

  virtual void updateRemove(const ParseDataSet& examples, MT19937& eng) = 0;

  virtual ~PypParsedWeightsInterface() {}
};

}
#endif
