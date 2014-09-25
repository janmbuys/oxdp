#############################################
# Libraries
#############################################

add_libraries(lbl
  class_distribution.cc
  config.cc
  context_cache.cc
  context_processor.cc
  factored_metadata.cc
  factored_weights.cc
  metadata.cc
  minibatch_words.cc
  model.cc
  model_utils.cc
  weights.cc
  word_distributions.cc
  word_to_class_index.cc
)

target_link_libs(lbl corpus murmurhash rt ${Boost_LIBRARIES})

#############################################
# Executables
#############################################

set(EXECUTABLES
  extract_word_vectors
  evaluate
  predict
  train_sgd
  train_factored_sgd
)
foreach(f ${EXECUTABLES})
 add_executable(${f} ${f}.cc)
 target_link_libraries(${f} lbl corpus murmurhash utils)
endforeach(f)

add_subdirectory(tests)
