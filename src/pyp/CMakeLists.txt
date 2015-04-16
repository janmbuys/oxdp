#############################################
# Libraries
#############################################

add_libraries(pyp
  pyp_weights.cc
  parsed_pyp_weights.cc
  parsed_lex_pyp_weights.cc
  parsed_calex_pyp_weights.cc
  parsed_chlex_pyp_weights.cc
)

target_link_libs(pyp corpus)

#############################################
# Executables
#############################################

set(EXECUTABLES
  pyplm_test
)
foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} corpus pyp)
endforeach(f)
