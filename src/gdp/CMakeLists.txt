#############################################
# Libraries
#############################################

add_libraries(gdp
  parser.cc
  transition_parser.cc  
  arc_standard_labelled_parser.cc
  accuracy_counts.cc
  arc_standard_labelled_parse_model.cc
  ngram_model.cc
  pyp_model.cc
  pyp_dp_model.cc
  lbl_model.cc
  lbl_dp_model.cc
)
 
target_link_libs(gdp utils corpus pyp lbl) 

#############################################
# Executables
#############################################

set(EXECUTABLES
  extract_word_vectors
  train_sgd
  train_gibbs
)

foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} utils corpus pyp gdp lbl)
endforeach(f)

