#############################################
# Libraries
#############################################

add_libraries(gdp
  parser.cc
  eisner_parser.cc
  transition_parser.cc  
  arc_eager_labelled_parser.cc
  arc_standard_labelled_parser.cc
  accuracy_counts.cc
  eisner_parse_model.cc
  arc_standard_labelled_parse_model.cc
  arc_eager_labelled_parse_model.cc
  ngram_model.cc
  pyp_model.cc
  pyp_dp_model.cc
  lbl_model.cc
  lbl_dp_model.cc
)

  #arc_eager_parser.cc
  #arc_standard_parser.cc
  #arc_standard_parse_model.cc
  #arc_eager_parse_model.cc
  
target_link_libs(gdp utils corpus pyp lbl) 

#############################################
# Executables
#############################################

set(EXECUTABLES
  extract_word_vectors
  evaluate_lbl
  predict_lbl
  score_lbl
  train_sgd
  train_gibbs
)

foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} utils corpus pyp gdp lbl)
endforeach(f)

