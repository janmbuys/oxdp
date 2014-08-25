#############################################
# Libraries
#############################################

add_libraries(gdp
  transition_parser.cc  
)

#target_link_libs(gdp)

#############################################
# Executables
#############################################

set(EXECUTABLES
  hpyp_dp
)
foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} gdp)
endforeach(f)

