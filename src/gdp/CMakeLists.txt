#############################################
# Libraries
#############################################

add_libraries(gdp
  accuracy_counts.cc
  arc_eager_parser.cc
  arc_standard_parser.cc
  eisner_parser.cc
  parse.cc
  transition_parser.cc  
)

#target_link_libs(gdp)

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

