#ifndef _GDP_TR_PARSER_I_H
#define _GDP_TR_PARSER_I_H

namespace oxlm {

class TransitionParserInterface {
  public:

  virtual bool shift() = 0;

  virtual bool leftArc() = 0;

  virtual bool rightArc() = 0;

  virtual kAction oracleNext(const ParsedSentence& gold_arcs) const = 0;

  virtual bool executeAction(kAction a) = 0;
  
  virtual bool inTerminalConfiguration() const = 0;

  virtual Words wordContext() const = 0;

  virtual Words tagContext() const = 0;
  
  virtual Words actionContext() const = 0;

  virtual void extractExamples(const boost::shared_ptr<DataSet>& word_examples,
                       const boost::shared_ptr<DataSet>& tag_examples,
                       const boost::shared_ptr<DataSet>& action_examples) const = 0;

  virtual ~TransitionParserInterface() {}
};

}

#endif
