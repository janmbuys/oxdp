#############################################
# Libraries
#############################################

add_libraries(pyp
  model_config.cc
  pyp_weights.cc
  parsed_pyp_weights.cc
  parsed_lex_pyp_weights.cc
  pyp_model.cc
)

target_link_libs(pyp corpus)

