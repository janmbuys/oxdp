#include "gdp/arc_standard_parse_model.h"

namespace oxlm {

ArcStandardParseModel::ArcStandardParseModel(unsigned beam_size):
  ParseModel(beam_size) {}

//TODO make sure we are always dealing with neg log prob weights
ArcStandardParser ArcStandardParseModel::beamParseSentence(const ParsedSentence& sent, 
                                                            const ParsedWeightsInterface& weights) {
  std::vector<AsParserList> beam_chart; 
  beam_chart.push_back(AsParserList());
  beam_chart[0].push_back(boost::make_shared<ArcStandardParser>(sent)); 

  //shift ROOT symbol (probability 1)
  beam_chart[0][0]->shift(); 

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 1; k <= sent.size(); ++k) {
    //there are k beam lists. perform reduces down to list 1

    for (unsigned i = k - 1; i > 0; --i) { 
      //prune if size exceeds beam_size
      if (beam_chart[i].size() > beam_size) {
        std::sort(beam_chart[i].begin(), beam_chart[i].end(), TransitionParser::cmp_particle_weights); 
        //remove items with worst scores
        for (unsigned j = beam_chart[i].size(); j > beam_size; --j)
          beam_chart[i].pop_back();
      }

      //for every item in the list, add valid reduce actions to list i - 1 
      for (unsigned j = 0; (j < beam_chart[i].size()); ++j) {
        double reduceleftarcp = weights.predictAction(static_cast<WordId>(kAction::la), beam_chart[i][j]->actionContext());
        double reducerightarcp = weights.predictAction(static_cast<WordId>(kAction::ra), beam_chart[i][j]->actionContext());
        //std::cout << "(la: " << reduceleftarcp << ", ra: " << reducerightarcp << ")" << " ";
        double reducep = neg_log_sum_exp(reduceleftarcp, reducerightarcp);
       
        //TODO have option to make la/ra choice deterministic
        beam_chart[i-1].push_back(boost::make_shared<ArcStandardParser>(*beam_chart[i][j]));
        if (i > 1) { //left arc only invalid when stack size is 2 **
          beam_chart[i-1].push_back(boost::make_shared<ArcStandardParser>(*beam_chart[i][j]));

          beam_chart[i-1].back()->leftArc();
          beam_chart[i-1].back()->add_particle_weight(reduceleftarcp);
          beam_chart[i-1].rbegin()[1]->rightArc();
          beam_chart[i-1].rbegin()[1]->add_particle_weight(reducerightarcp); 

          if (k == sent.size()) {  
            beam_chart[i-1].back()->add_importance_weight(reducep); 
            beam_chart[i-1].rbegin()[1]->add_importance_weight(reducep); 
          } 
        } else {
          beam_chart[i-1].back()->rightArc();
          beam_chart[i-1].back()->add_particle_weight(reducerightarcp); 
          
          if (k == sent.size()) 
            beam_chart[i-1].back()->add_importance_weight(reducerightarcp/reducep); 
        }
      }
    }

    if ((beam_chart[0].size() > beam_size) || (k == sent.size())) {
        std::sort(beam_chart[0].begin(), beam_chart[0].end(), TransitionParser::cmp_particle_weights); 
        //remove items with worst scores
        for (unsigned j = beam_chart[0].size(); j > beam_size; --j)
          beam_chart[0].pop_back();
    }

    //perform shifts
    if (k < sent.size()) {
      for (unsigned i = 0; (i < k); ++i) { 
        for (unsigned j = 0; j < beam_chart[i].size(); ++j) {
          double shiftp = weights.predictAction(static_cast<WordId>(kAction::sh), 
                                                      beam_chart[i][j]->actionContext());
          double tagp = weights.predictTag(beam_chart[i][j]->next_tag(), 
                                           beam_chart[i][j]->tagContext());
          double wordp = weights.predictWord(beam_chart[i][j]->next_word(), 
                                             beam_chart[i][j]->wordContext());

          beam_chart[i][j]->shift();
          beam_chart[i][j]->add_particle_weight(shiftp); 
          beam_chart[i][j]->add_importance_weight(tagp); 
          beam_chart[i][j]->add_importance_weight(wordp); 
          beam_chart[i][j]->add_particle_weight(tagp); 
          beam_chart[i][j]->add_particle_weight(wordp); 
        }
      }
      //insert new beam_chart[0] to increment indexes
      beam_chart.insert(beam_chart.begin(), AsParserList());
    } 
    //std::cout << std::endl; 
  }
 
  //TODO sum over identical parses in final beam 
  unsigned n = 0; //index to final beam
  for (unsigned i = 0; (i < beam_chart[n].size()); ++i) 
    beam_chart[n][0]->add_beam_weight(beam_chart[n][i]->particle_weight());

  //print parses
  for (unsigned i = 0; (i < 5) && (i < beam_chart[n].size()); ++i) {
    std::cout << beam_chart[n][i]->particle_weight() << " ";
    beam_chart[n][i]->print_arcs();
    beam_chart[n][i]->print_actions();

    //can't do this now, but add if needed later
    //float dir_acc = (beam_chart[n][i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
    //std::cout << "  Dir Accuracy: " << dir_acc;
  }  

  if (beam_chart[n].size()==0) {
    std::cout << "no parse found" << std::endl;
    return ArcStandardParser(sent);  
  } else
    return ArcStandardParser(*beam_chart[n][0]); 
}  

ArcStandardParser ArcStandardParseModel::staticGoldParseSentence(const ParsedSentence& sent, 
                                    const ParsedWeightsInterface& weights) {
  ArcStandardParser parser(sent);
  
  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && (a != kAction::re)) {
    a = parser.oracleNext(sent);  
    if (a != kAction::re) {
      //update particle weight
      double actionp = weights.predictAction(static_cast<WordId>(a), parser.actionContext());
      parser.add_particle_weight(actionp);

      if (a == kAction::sh) {
        double tagp = weights.predictTag(parser.next_tag(), parser.tagContext());
        double wordp = weights.predictWord(parser.next_word(), parser.wordContext());
        parser.add_particle_weight(tagp);
        parser.add_particle_weight(wordp);
      }

      parser.executeAction(a);
    } 
  }

  return parser;
}
    
ArcStandardParser ArcStandardParseModel::staticGoldParseSentence(const ParsedSentence& sent) {
  ArcStandardParser parser(sent);
  
  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && (a != kAction::re)) {
    a = parser.oracleNext(sent);  
    if (a != kAction::re) 
      parser.executeAction(a);
  }

  return parser;
}

}

