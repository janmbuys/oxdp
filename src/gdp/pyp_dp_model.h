#ifndef _GDP_PYP_MODEL_H_
#define _GDP_PYP_MODEL_H_

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "pyp/pyp_parsed_weights_interface.h"
#include "gdp/model_config.h"
#include "corpus/dict.h"
#include "corpus/corpus.h"

namespace oxlm {

//identity more abstractions later
//TODO add context extractor
class PypModel {
  public:
  PypModel();

  PypModel(const boost::shared_ptr<ModelConfig>& config);

  void learn();

  void evaluate() const;

  void evaluate(const boost::shared_ptr<Corpus>& corpus) const;

  private:
  boost::shared_ptr<ModelConfig> config_;
  boost::shared_ptr<Dict> dict_;
  boost::shared_ptr<PypParsedWeightsInterface> weights_; //see if this works
  size_t num_actions_;
};

}

#endif

