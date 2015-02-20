#############################################
# Libraries
#############################################

add_libraries(corpus
  dict.cc
  sentence.cc
  context.cc
  tagged_sentence.cc
  parsed_sentence.cc
  corpus.cc
  sentence_corpus.cc
  parsed_corpus.cc
  data_point.cc
  data_set.cc
  parse_data_set.cc
  model_config.cc
)

#target_link_libs(corpus)

