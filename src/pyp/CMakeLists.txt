#############################################
# Libraries
#############################################

add_libraries(pyp
  pyp_weights.cc
  parsed_pyp_weights.cc
)

target_link_libs(pyp corpus)

