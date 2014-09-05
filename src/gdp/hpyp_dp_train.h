#ifndef _GDP_HPYP_DP_TRAIN_H_
#define _GDP_HPYP_DP_TRAIN_H_

#include <iostream>
#include <unordered_map>
#include <cstdlib>
#include <chrono>

#include "hpyp_dp_parse.h"
#include "transition_parser.h"
#include "eisner_parser.h"
#include "hpyplm/hpyplm.h"
#include "corpus/dict.h"
#include "utils/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

namespace oxlm {

//for Eisner parsing
template<unsigned kShiftOrder, unsigned kTagOrder>
void trainSupervisedEisnerParser(const boost::shared_ptr<Sentences>& sents, const boost::shared_ptr<Sentences>& tags, const boost::shared_ptr<IndicesList>& gold_deps, int num_iterations, bool with_words, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kTagOrder>* tag_lm) {

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_sh(sents->size(), WordsList());
  std::vector<WordsList> examples_list_tag(sents->size(), WordsList());
  
  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {
     //std::cout << "Training iter " << iter << std::endl;

    for (unsigned j = 0; j < sents->size(); ++j) {
      //if (tags[j].size() <= 11) {
      //remove old sample from model
      if (iter > 0) {
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        if (with_words)
          updatePYPModel(false, eng, examples_list_sh[j], shift_lm);
      }

      if (iter == 0) {
        EisnerParser sample_parse(sents->at(j), tags->at(j), gold_deps->at(j));
        extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_tag[j]);
        //for (auto ex: examples_list_tag[j])
        //  std::cout << dict.lookupTag(ex[0]) << " " << dict.lookupTag(ex[1]) << std::endl;
      } 

      //update with new sample
      if (with_words)
        updatePYPModel(true, eng, examples_list_sh[j], shift_lm);
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
    //}
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words)
        std::cerr << " [Shift LLH=" << shift_lm->log_likelihood() << "]\n"; 
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words) {
        shift_lm->resample_hyperparameters(eng);      
        std::cerr << "  [Shift LLH=" << shift_lm->log_likelihood() << "]\n\n";    
      }
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
    }
  }
}

//for three-way decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
void trainSupervisedParser(const boost::shared_ptr<Sentences>& sents, const boost::shared_ptr<Sentences>& tags, const boost::shared_ptr<IndicesList>& gold_deps, int num_iterations, bool with_words, bool arceager, bool static_oracle, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kReduceOrder>* reduce_lm, PYPLM<kTagOrder>* tag_lm) {

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_sh(sents->size(), WordsList());
  std::vector<WordsList> examples_list_tag(sents->size(), WordsList());
  std::vector<WordsList> examples_list_re(sents->size(), WordsList());
  
  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {
     //std::cout << "Training iter " << iter << std::endl;

    for (unsigned j = 0; j < sents->size(); ++j) {
      //if (tags[j].size() <= 11) {
      //remove old sample from model
      if (iter > 0) {
        if (with_words)
          updatePYPModel(false, eng, examples_list_sh[j], shift_lm);
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
      }

      //get a parse (derivation) given the current parameters
      ArcList gold_dep(gold_deps->at(j).size());
      gold_dep.set_arcs(gold_deps->at(j));  //problematic

      if (iter == 0) {
        if (arceager) {
          //std::cout << "gold " << j << std::endl;
          ArcEagerParser sample_parse = staticEagerGoldParseSentence(sents->at(j), tags->at(j), gold_dep);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
          //gold_dep.print_arcs();
          //sample_parse.print_arcs();
        } else { 
          ArcStandardParser sample_parse = staticGoldParseSentence(sents->at(j), tags->at(j), gold_dep);
          //std::cout << sample_parse.actions_str() << "\n";
          //sample_parse.print_arcs();
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
        }
      } else if (!static_oracle) {
        unsigned num_particles = 100;
        bool resample = false;
        
        if (arceager) {
          ArcEagerParser sample_parse = particleEagerGoldParseSentence(sents->at(j), tags->at(j), gold_dep, num_particles, resample, with_words, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
        } else {  
          ArcStandardParser sample_parse = particleDynamicGoldParseSentence(sents->at(j), tags->at(j), gold_dep, num_particles, resample, with_words, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
          //technically need a MH acceptance test
        }
      } 

      //update with new sample
      if (with_words)
        updatePYPModel(true, eng, examples_list_sh[j], shift_lm);
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
      updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
    //}
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words)
        std::cerr << " [Shift LLH=" << shift_lm->log_likelihood() << "]\n"; 
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
      std::cerr << " [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n";
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words) {
        shift_lm->resample_hyperparameters(eng);      
        std::cerr << "  [Shift LLH=" << shift_lm->log_likelihood() << "]\n\n";    
      }
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
      reduce_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n\n";
    }
  
  }
}

//for three-way decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kTagOrder>
void trainSemisupervisedParser(const boost::shared_ptr<Sentences>& sents, const boost::shared_ptr<Sentences>& tags, const boost::shared_ptr<IndicesList>& gold_deps, int num_iterations, bool with_words, bool arceager, bool static_oracle, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kReduceOrder>* reduce_lm, PYPLM<kTagOrder>* tag_lm) {

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_sh(sents->size(), WordsList());
  std::vector<WordsList> examples_list_tag(sents->size(), WordsList());
  std::vector<WordsList> examples_list_re(sents->size(), WordsList());
  
  unsigned split_n = 4000;
  unsigned num_particles = 100;
  bool resample = false;
  unsigned max_sent_length = 15;

  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {
     //std::cout << "Training iter " << iter << std::endl;

    for (unsigned j = 0; j < split_n; ++j) {
      //if (tags[j].size() <= (max_sent_length+1)) {
      //remove old sample from model
      if (iter > 0) {
        if (with_words)
          updatePYPModel(false, eng, examples_list_sh[j], shift_lm);
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
      }

      //get a parse (derivation) given the current parameters
      ArcList gold_dep(sents->at(j).size());
      gold_dep.set_arcs(gold_deps->at(j)); 

      if (iter == 0) {
        if (arceager) {
          ArcEagerParser sample_parse = staticEagerGoldParseSentence(sents->at(j), tags->at(j), gold_dep);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
        } else { 
          ArcStandardParser sample_parse = staticGoldParseSentence(sents->at(j), tags->at(j), gold_dep);
          //sample_parse.print_arcs();
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
        }
        //std::cout << "\n" << sample_parse.actions_str() << "\n";
      } else if (!static_oracle) {
        
        if (arceager) {
          ArcEagerParser sample_parse = particleEagerGoldParseSentence(sents->at(j), tags->at(j), gold_dep, num_particles, resample, with_words, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
        } else {  
          ArcStandardParser sample_parse = particleDynamicGoldParseSentence(sents->at(j), tags->at(j), gold_dep, num_particles, resample, with_words, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, &examples_list_sh[j], &examples_list_re[j], &examples_list_tag[j]);
          //technically need a MH acceptance test
        }
      } 

      //update with new sample
      if (with_words)
        updatePYPModel(true, eng, examples_list_sh[j], shift_lm);
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
      updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
    //}
    }

    //Unsupervised part
    num_particles = 1000;

    for (unsigned j = split_n; j < tags->size(); ++j) {
      if (tags->at(j).size() <= (max_sent_length+1)) {
        
        //remove old sample from model (if there is any)
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
        
        //only for evaluation
        ArcList gold_dep(tags->at(j).size());
        gold_dep.set_arcs(gold_deps->at(j)); 
    
        //sample new parse (derivation) 
        //technically need a MH acceptance test
        if (arceager) {
          ArcEagerParser sample_parse = particleEagerParseSentence(Words(tags->at(j).size(), '_'), tags->at(j), gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, nullptr, &examples_list_re[j], &examples_list_tag[j]);
        } else {
          ArcStandardParser sample_parse = particleParseSentence(Words(tags->at(j).size(), '_'), tags->at(j), gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          extractParseTrainExamples(sample_parse, nullptr, &examples_list_re[j], &examples_list_tag[j]);
        }

        //update with new sample
        updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
      }
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words)
        std::cerr << " [Shift LLH=" << shift_lm->log_likelihood() << "]\n"; 
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
      std::cerr << " [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n";
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      if (with_words) {
        shift_lm->resample_hyperparameters(eng);      
        std::cerr << "  [Shift LLH=" << shift_lm->log_likelihood() << "]\n\n";    
      }
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
      reduce_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n\n";
    }
  
  }
}



//for ternary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder,  unsigned kTagOrder>
void trainUnsupervisedParser(const boost::shared_ptr<Sentences>& tags, const boost::shared_ptr<IndicesList>& gold_deps, int num_iterations, bool init, bool arceager, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kReduceOrder>* reduce_lm, PYPLM<kTagOrder>* tag_lm) {
  unsigned num_particles = 1000;
  bool resample = false;
  unsigned max_sent_length = 15;

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_tag(tags->size(), WordsList());
  std::vector<WordsList> examples_list_re(tags->size(), WordsList());
 
  std::string log_file = "arceager.scores.out";
  std::ofstream outl;
  outl.open(log_file); 

  if (init) {
    //initialize tag_lm with a sequential trigram model
    std::vector<WordId> ctx(kTagOrder - 1, 0);
    for (unsigned j = 0; j < tags->size(); ++j) {
      ctx.resize(kTagOrder - 1);
      for (unsigned i = 0; i < tags->at(j).size(); ++i) {
        WordId tag = tags->at(j)[i];
        Words tag_tuple(1, tag);
        tag_tuple.insert(tag_tuple.end(), ctx.begin() + (ctx.size() - (kTagOrder - 1)), ctx.end());
        examples_list_tag[j].push_back(tag_tuple);
        ctx.push_back(tag);
      }
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm); 
    }
  } 

  //TODO try different initializations
  
  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {

    //for (unsigned j = 0; j < tags.size(); ++j) {
    for (unsigned j = 4000; j < tags->size(); ++j) {
      if (tags->at(j).size() <= (max_sent_length+1)) {
      //if (tags[j].size() <= std::min(iter/2 + 3, 11)) {

      //if (((iter < 20) && (tags[j].size() <= 11)) || 
      //    ((iter >= 20) && (iter < 40) && (tags[j].size()) <= 21)) {
        
        //remove old sample from model (if there is any)
        updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
        
        //if (iter > 0) {
        //} else if (init) {
        //  updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        //}

        //only for evaluation
        ArcList gold_dep(tags->at(j).size());
        gold_dep.set_arcs(gold_deps->at(j)); 
    
        //sample new parse (derivation) 
        if (arceager) {
          ArcEagerParser sample_parse;
          if ((iter==0) && init)
            sample_parse = particleInitEagerParseSentence(Words(tags->at(j).size(), '_'), tags->at(j), gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *tag_lm);
          else
            sample_parse = particleEagerParseSentence(Words(tags->at(j).size(), '_'), tags->at(j), gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          //technically need a MH acceptance test
         
          extractParseTrainExamples(sample_parse, nullptr, &examples_list_re[j], &examples_list_tag[j]);
        } else {
          ArcStandardParser sample_parse;
          if ((iter==0) && init)
            sample_parse = particleInitParseSentence(Words(tags->at(j).size(), '_'), tags->at(j), gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *tag_lm);
          else
            sample_parse = particleParseSentence(Words(tags->at(j).size(), '_'), tags->at(j), gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
          //technically need a MH acceptance test
         
          extractParseTrainExamples(sample_parse, nullptr, &examples_list_re[j], &examples_list_tag[j]);
 
        }

        //update with new sample
        updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
      }
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
      std::cerr << " [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n";
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
      reduce_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n\n";
      
      //evaluate
      std::string test_file;
      double acc;
  
      test_file = "english-wsj-conll07-nowords-nopunc10/english_wsj_dev.conll";
      acc = evaluate(test_file, arceager, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
      outl << acc << " ";

      test_file = "english-wsj-conll07-nowords-nopunc20/english_wsj_dev.conll";
      acc = evaluate(test_file, arceager, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
      outl << acc << " ";

      test_file = "english-wsj-conll07-nowords-nopunc/english_wsj_dev.conll";
      acc = evaluate(test_file, arceager, false, dict, eng, *shift_lm, *reduce_lm, *tag_lm);
      outl << acc << std::endl;
    }
  }

  outl.close();
}

//for binary decisions
template<unsigned kShiftOrder, unsigned kReduceOrder, unsigned kArcOrder, unsigned kTagOrder>
void trainUnsupervisedParser(const boost::shared_ptr<Sentences>& tags, const boost::shared_ptr<IndicesList>& gold_deps, int num_iterations, bool init, Dict& dict, MT19937& eng, PYPLM<kShiftOrder>* shift_lm, PYPLM<kReduceOrder>* reduce_lm, PYPLM<kArcOrder>* arc_lm, PYPLM<kTagOrder>* tag_lm) {
  unsigned num_particles = 1000;
  bool resample = false;

  //keep an example list for each sentence
  std::vector<WordsList> examples_list_tag(tags->size(), WordsList());
  std::vector<WordsList> examples_list_re(tags->size(), WordsList());
  std::vector<WordsList> examples_list_arc(tags->size(), WordsList());
 
  if (init) {
    //initialize tag_lm with a sequential trigram model
    std::vector<WordId> ctx(kTagOrder - 1, 0);
    for (unsigned j = 0; j < tags->size(); ++j) {
      ctx.resize(kTagOrder - 1);
      for (unsigned i = 0; i < tags->at(j).size(); ++i) {
        WordId tag = tags->at(j)[i];
        Words tag_tuple(1, tag);
        tag_tuple.insert(tag_tuple.end(), ctx.begin() + (ctx.size() - (kTagOrder - 1)), ctx.end());
        examples_list_tag[j].push_back(tag_tuple);
        ctx.push_back(tag);
      }
      updatePYPModel(true, eng, examples_list_tag[j], tag_lm); 
    }
  } 
  
  //repeat for number of samples: for each sentence, get training examples and update model
  for (int iter = 0; iter < num_iterations; ++iter) {

    for (unsigned j = 0; j < tags->size(); ++j) {
      if (((iter < 20) && (tags->at(j).size() <= 11)) || 
          ((iter >= 20) && (iter < 40) && (tags->at(j).size()) <= 21)) {

        //remove old sample from model
        if (iter > 0) {
          updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
          updatePYPModel(false, eng, examples_list_re[j], reduce_lm);
          updatePYPModel(false, eng, examples_list_arc[j], arc_lm);
        } else if (init) {
          updatePYPModel(false, eng, examples_list_tag[j], tag_lm);
        }

        //only for evaluation
        ArcList gold_dep(tags->at(j).size());
        gold_dep.set_arcs(gold_deps->at(j)); 
    
        //sample new parse (derivation) 
        ArcStandardParser sample_parse = particleParseSentence(Words(tags->at(j).size(), '_'), tags->at(j), gold_dep, num_particles, resample, false, false, dict, eng, *shift_lm, *reduce_lm, *arc_lm, *tag_lm);
        //technically need a MH acceptance test
      
        extractParseTrainExamples(sample_parse, nullptr, &examples_list_re[j], &examples_list_arc[j], &examples_list_tag[j]);

        //update with new sample
        updatePYPModel(true, eng, examples_list_tag[j], tag_lm);
        updatePYPModel(true, eng, examples_list_re[j], reduce_lm);
        updatePYPModel(true, eng, examples_list_arc[j], arc_lm);
      }
    }

    std::cerr << ".";
    if ((iter == 0)) {
      std::cerr << (iter + 1) << " iterations\n";
      std::cerr << " [Tag LLH=" << tag_lm->log_likelihood() << "]\n";    
      std::cerr << " [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n";
      std::cerr << " [Arc LLH=" << arc_lm->log_likelihood() << "]\n";
    } else if (iter % 10 == 9) { //if (i % 30 == 29) 
      std::cerr << (iter + 1) << " iterations\n";
      tag_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Tag LLH=" << tag_lm->log_likelihood() << "]\n\n";    
      reduce_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Reduce LLH=" << reduce_lm->log_likelihood() << "]\n\n";
      arc_lm->resample_hyperparameters(eng);      
      std::cerr << "  [Arc LLH=" << arc_lm->log_likelihood() << "]\n\n";
    }
  }
}

}
#endif
