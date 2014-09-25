set(TESTS
    class_distribution_test
    context_processor_test
    factored_metadata_test
    factored_weights_test
    metadata_test
    model_utils_test
    operators_test
    train_factored_sgd_test
    train_sgd_test
    utils_test
    weights_test
    word_distributions_test
    word_to_class_index_test)

set(LIBRARIES murmurhash corpus lbl)

foreach(test ${TESTS})
    lbl_add_test(${test})
    target_link_libraries(${test} ${LIBRARIES})
endforeach(test)
