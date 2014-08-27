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
)

target_link_libs(gdp corpus)

#############################################
# Executables
#############################################

set(EXECUTABLES
  train_hpyp_dp
)
foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} gdp)
endforeach(f)

