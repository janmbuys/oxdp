#############################################
# Libraries
#############################################

add_libraries(gdp
  parser.cc
  eisner_parser.cc
  transition_parser.cc  
  arc_eager_parser.cc
  arc_standard_parser.cc
  accuracy_counts.cc
  eisner_parse_model.cc
  arc_standard_parse_model.cc
  arc_eager_parse_model.cc
  model_config.cc
  pyp_dp_model.cc
)

target_link_libs(gdp corpus pyp)

#############################################
# Executables
#############################################

set(EXECUTABLES
  train_pyp_dp
)
foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} corpus pyp gdp)
endforeach(f)

