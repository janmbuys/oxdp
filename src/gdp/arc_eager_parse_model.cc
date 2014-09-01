#include "gdp/arc_eager_parse_model.h"

namespace oxlm {

ArcEagerParseModel::ArcEagerParseModel(unsigned size):
  ParseModel(size) {}

ArcEagerParser ArcEagerParseModel::beamParseSentence(const ParsedSentence& sent, 
                                                            const ParsedWeightsInterface& weights) {
  //index in beam_chart is depth-of-stack - 1
  std::vector<AeParserList> beam_chart; 
  beam_chart.push_back(AeParserList());
  beam_chart[0].push_back(boost::make_shared<ArcEagerParser>(sent)); 

  //shift ROOT symbol (probability 1)
  beam_chart[0][0]->shift(); 

  //add reduce actions, then shift word k (expect for last iteration) 
  for (unsigned k = 1; k < sent.size(); ++k) {
    //there are k beam lists. perform reduces down to list 1
    for (unsigned i = k - 1; i > 0; --i) { 
      //prune if size exceeds beam_size
      if (beam_chart[i].size() > beam_size) {
        std::sort(beam_chart[i].begin(), beam_chart[i].end(), TransitionParser::cmp_particle_weights); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_chart[i].size(); j > beam_size; --j)
          beam_chart[i].pop_back();
      }

      //std::cout << "reduce list size: " << beam_chart[i].size() << std::endl;
      //consider reduce and left arc actions
      //for every item in the list, add valid reduce actions to list i - 1 
      for (unsigned j = 0; (j < beam_chart[i].size()); ++j) {
        double leftarcreducep = weights.predictAction(static_cast<WordId>(kAction::la), 
                                                       beam_chart[i][j]->actionContext());
        double reducep = weights.predictAction(static_cast<WordId>(kAction::re), 
                                                       beam_chart[i][j]->actionContext());
        //std::cout << "(la: " << leftarcreducep << ", re: " << reducep << ") ";
        //double reducetotalp = leftarcreducep + reducep;
       
        //actually also need importance weight if either is invalid
        //left arc invalid also after last shift
        if (beam_chart[i][j]->left_arc_valid()) { 
          beam_chart[i-1].push_back(boost::make_shared<ArcEagerParser>(*beam_chart[i][j]));
          beam_chart[i-1].back()->leftArc();
          beam_chart[i-1].back()->add_particle_weight(leftarcreducep);
        } 
        
        if (beam_chart[i][j]->reduce_valid()) {          
          beam_chart[i-1].push_back(boost::make_shared<ArcEagerParser>(*beam_chart[i][j]));
          beam_chart[i-1].back()->reduce();
          beam_chart[i-1].back()->add_particle_weight(reducep); 
          
        }
      }
      //std::cout << std::endl;
    }

    if (beam_chart[0].size() > beam_size) {
        std::sort(beam_chart[0].begin(), beam_chart[0].end(), TransitionParser::cmp_particle_weights); //handle pointers
        //remove items with worst scores
        for (unsigned j = beam_chart[0].size(); j > beam_size; --j)
          beam_chart[0].pop_back();
    }

    //perform shifts: shift or right arc
    for (unsigned i = 0; (i < k); ++i) { 
      unsigned list_size = beam_chart[i].size();
      for (unsigned j = 0; j < list_size; ++j) {
          
        double shiftp = weights.predictAction(static_cast<WordId>(kAction::sh), 
                                              beam_chart[i][j]->actionContext());
        double rightarcshiftp = weights.predictAction(static_cast<WordId>(kAction::ra), 
                                              beam_chart[i][j]->actionContext());
        //double shifttotalp = shiftp + rightarcshiftp;
        //std::cout << "(sh: " << shiftp << ", ra: " << rightarcshiftp << ") ";

        //ra not valid for stop symbol
        if (k < (sent.size() - 1)) {
          double tagp = weights.predictTag(beam_chart[i][j]->next_tag(), 
                                           beam_chart[i][j]->tagContext(kAction::ra));
          double wordp = weights.predictWord(beam_chart[i][j]->next_word(), 
                                             beam_chart[i][j]->wordContext());
         
          beam_chart[i].push_back(boost::make_shared<ArcEagerParser>(*beam_chart[i][j]));
          beam_chart[i].back()->rightArc();
          beam_chart[i].back()->add_particle_weight(rightarcshiftp);
          beam_chart[i].back()->add_importance_weight(tagp); 
          beam_chart[i].back()->add_importance_weight(wordp); 
          beam_chart[i].back()->add_particle_weight(tagp); 
          beam_chart[i].back()->add_particle_weight(wordp); 
        }

        //shift is valid
        double tagp = weights.predictTag(beam_chart[i][j]->next_tag(), 
                                           beam_chart[i][j]->tagContext(kAction::sh));
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
    beam_chart.insert(beam_chart.begin(), AeParserList());
       
    //std::cout << std::endl;
  }
 
  //completion: reduce after last shift
  //std::cout << "completion" << std::endl;
  for (unsigned i = beam_chart.size() - 1; i > 0; --i) {  //sent.size()
    //prune if size exceeds beam_size
    if (beam_chart[i].size() > beam_size) {
      std::sort(beam_chart[i].begin(), beam_chart[i].end(), ArcEagerParser::cmp_reduce_particle_weights); //handle pointers
      //remove items with worst scores, and those that cannot reduce
      for (unsigned j = beam_chart[i].size() - 1; ((j >= beam_size) || ((j > 0) && !beam_chart[i][j]->reduce_valid())); --j)
        beam_chart[i].pop_back();
    }

    //std::cout << i <<  " reduce list size: " << beam_chart[i].size() << std::endl;
    //consider reduce and left arc actions
    //for every item in the list, add valid reduce actions to list i - 1 
    for (unsigned j = 0; (j < beam_chart[i].size()); ++j) {
      double reducep = weights.predictAction(static_cast<WordId>(kAction::re), 
                                             beam_chart[i][j]->actionContext());
                
      if (beam_chart[i][j]->reduce_valid()) {  
        beam_chart[i-1].push_back(boost::make_shared<ArcEagerParser>(*beam_chart[i][j]));
        beam_chart[i-1].back()->reduce();
        //in some models, don't add the weights
        beam_chart[i-1].back()->add_particle_weight(reducep); 
        beam_chart[i-1].back()->add_importance_weight(reducep); 
        //std::cout << j << " re valid ";
      }
    }
    //std::cout << std::endl;
  }

  std::sort(beam_chart[0].begin(), beam_chart[0].end(), TransitionParser::cmp_particle_weights); //handle pointers

  //print parses
  unsigned const n = 0; 
  //std::cout << "Beam size: " << beam_chart[n].size() << std::endl;
  for (unsigned i = 0; (i < beam_chart[n].size()); ++i) 
    beam_chart[n][0]->add_particle_weight(beam_chart[n][i]->particle_weight());

  for (unsigned i = 0; (i < 5) && (i < beam_chart[n].size()); ++i) {
    beam_chart[n][i]->print_arcs();
    beam_chart[n][i]->print_actions();

    //float dir_acc = (beam_chart[n][i]->directed_accuracy_count(gold_dep) + 0.0)/(sent.size()-1);
    //std::cout << "  Dir Accuracy: " << dir_acc;
    //std::cout << "  Sample weight: " << (beam_chart[n][i]->particle_weight()) << std::endl;
  }  

  if (beam_chart[n].size()==0) {
    //std::cout << "no parse found" << std::endl;
    return ArcEagerParser(sent);  
  } else {
    beam_chart[n][0]->print_arcs();
    beam_chart[n][0]->print_actions();
    return ArcEagerParser(*beam_chart[n][0]); 
  }

}

ArcEagerParser ArcEagerParseModel::staticGoldParseSentence(const ParsedSentence& sent,
                                        const ParsedWeightsInterface& weights) {
  ArcEagerParser parser(sent);

  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && !(parser.buffer_empty() && (a == kAction::sh))) {
    a = parser.oracleNext(sent);
    if (!(parser.buffer_empty() && (a == kAction::sh))) {
      //update particle weight
      double actionp = weights.predictAction(static_cast<WordId>(a), parser.actionContext());
      parser.add_particle_weight(actionp);

      if (a == kAction::sh || a == kAction::ra) {
        double tagp = weights.predictTag(parser.next_tag(), parser.tagContext(a));
        double wordp = weights.predictWord(parser.next_word(), parser.wordContext());
        parser.add_particle_weight(tagp);
        parser.add_particle_weight(wordp);
      }
      
      parser.executeAction(a); 
    } 
  }

  return parser;
}

ArcEagerParser ArcEagerParseModel::staticGoldParseSentence(const ParsedSentence& sent) {
  ArcEagerParser parser(sent);

  kAction a = kAction::sh;
  while (!parser.inTerminalConfiguration() && !(parser.buffer_empty() && (a == kAction::sh))) {
    a = parser.oracleNext(sent);
    if (!(parser.buffer_empty() && (a == kAction::sh))) 
      parser.executeAction(a); 
  }

  return parser;
}

}

