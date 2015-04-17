#include "pyp/parsed_pyp_weights.h"

namespace oxlm {

template<unsigned tOrder, unsigned aOrder>
ParsedPypWeights<tOrder, aOrder>::ParsedPypWeights(boost::shared_ptr<Dict> dict, 
        boost::shared_ptr<Dict> ch_dict, size_t num_actions):
  PypWeights<tOrder>(dict->tag_size()),    
  shre_lm_(2, 1, 1, 1, 1),
  action_lm_(num_actions, 1, 1, 1, 1), //-1
  num_actions_(num_actions) {}

template<unsigned tOrder, unsigned aOrder>
Real ParsedPypWeights<tOrder, aOrder>::predictWord(WordId word, Context context) const {
  return 0;
}

template<unsigned tOrder, unsigned aOrder>
Reals ParsedPypWeights<tOrder, aOrder>::predictWord(Context context) const {
  return Reals(numWords(), 0);
}

template<unsigned tOrder, unsigned aOrder>
Reals ParsedPypWeights<tOrder, aOrder>::predictWordOverTags(WordId word, Context context) const {
  return Reals(numTags(), 0);
}

template<unsigned tOrder, unsigned aOrder>
Real ParsedPypWeights<tOrder, aOrder>::predictTag(WordId tag, Context context) const {
  return PypWeights<tOrder>::predict(tag, context);
}

template<unsigned tOrder, unsigned aOrder>
Reals ParsedPypWeights<tOrder, aOrder>::predictTag(Context context) const {
  Reals weights(numTags(), 0);
  for (int i = 0; i < numTags(); ++i)
    weights[i] = predictTag(i, context);
  return weights;
}

template<unsigned tOrder, unsigned aOrder>
Real ParsedPypWeights<tOrder, aOrder>::predictAction(WordId action, Context context) const {
  /*if (action == 0) {
    return -std::log(shre_lm_.prob(0, context.words));
  } else {
    return -std::log(shre_lm_.prob(1, context.words)) 
        - std::log(action_lm_.prob(action - 1, context.words));
  } */
  return -std::log(action_lm_.prob(action, context.words));
}


template<unsigned tOrder, unsigned aOrder>
Reals ParsedPypWeights<tOrder, aOrder>::predictAction(Context context) const {
  Reals weights(num_actions_, 0);
  for (int i = 0; i < num_actions_; ++i)
    weights[i] = predictAction(i, context);
  return weights;
}

template<unsigned tOrder, unsigned aOrder>
Real ParsedPypWeights<tOrder, aOrder>::wordLikelihood() const {
  return 0;
}

template<unsigned tOrder, unsigned aOrder>
Real ParsedPypWeights<tOrder, aOrder>::tagLikelihood() const {
  return PypWeights<tOrder>::likelihood();
}

template<unsigned tOrder, unsigned aOrder>
Real ParsedPypWeights<tOrder, aOrder>::actionLikelihood() const {
  //return -shre_lm_.log_likelihood() - action_lm_.log_likelihood();
  return -action_lm_.log_likelihood();
}

template<unsigned tOrder, unsigned aOrder>
Real ParsedPypWeights<tOrder, aOrder>::likelihood() const {
  return tagLikelihood() + actionLikelihood();
}

template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::resampleHyperparameters(MT19937& eng) {
  PypWeights<tOrder>::resampleHyperparameters(eng);
  //shre_lm_.resample_hyperparameters(eng);
  action_lm_.resample_hyperparameters(eng);
  std::cerr << "  [Action LLH=" << actionLikelihood() << "]\n\n";    
}

//update PYP model to insert new training examples 
template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::updateInsert(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) {
  PypWeights<tOrder>::updateInsert(examples->tag_examples(), eng);
  for (unsigned i = 0; i < examples->action_example_size(); ++i) {
   /*if (examples->action_at(i) == 0) 
      shre_lm_.increment(0, examples->action_context_at(i).features[0], eng);
    else { 
      shre_lm_.increment(1, examples->action_context_at(i).features[0], eng);
      action_lm_.increment(examples->action_at(i) - 1, examples->action_context_at(i).words, eng);
    } */ 
    action_lm_.increment(examples->action_at(i), examples->action_context_at(i).words, eng);
  }
}

//update PYP model to remove old training examples
template<unsigned tOrder, unsigned aOrder>
void ParsedPypWeights<tOrder, aOrder>::updateRemove(const boost::shared_ptr<ParseDataSet>& examples, MT19937& eng) {
  PypWeights<tOrder>::updateRemove(examples->tag_examples(), eng);
  for (unsigned i = 0; i < examples->action_example_size(); ++i) {
    /*if (examples->action_at(i) == 0) 
      shre_lm_.decrement(0, examples->action_context_at(i).features[0], eng);
    else { 
      shre_lm_.decrement(1, examples->action_context_at(i).features[0], eng);
      action_lm_.decrement(examples->action_at(i) - 1, examples->action_context_at(i).words, eng);
    } */
    action_lm_.decrement(examples->action_at(i), examples->action_context_at(i).words, eng);
  }
}

template<unsigned tOrder, unsigned aOrder>
int ParsedPypWeights<tOrder, aOrder>::numWords() const {
  return 1;
}

template<unsigned tOrder, unsigned aOrder>
int ParsedPypWeights<tOrder, aOrder>::numTags() const {
  return PypWeights<tOrder>::vocabSize();
}

template<unsigned tOrder, unsigned aOrder>
int ParsedPypWeights<tOrder, aOrder>::numActions() const {
  return num_actions_;
}

template class ParsedPypWeights<tagLMOrderAS, actionLMOrderAS>;

#if ((tagLMOrderAS != tagLMOrderAE) || (actionLMOrderAS != actionLMOrderAE))
template class ParsedPypWeights<tagLMOrderAE, actionLMOrderAE>;
#endif

template class ParsedPypWeights<tagLMOrderE, 1>;

}

