#pragma once

#include <map>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include "corpus/dict.h"
#include "lbl/utils.h"

// Helper functions for reading data, evaluating models, etc.

namespace oxlm {
   
vector<int> scatterMinibatch(const vector<int>& minibatch);

void loadClassesFromFile(
    const string& class_file, const string& training_file,
    vector<int>& classes, boost::shared_ptr<Dict>& dict, VectorReal& class_bias);

void frequencyBinning(
    const string& training_file, int num_classes,
    vector<int>& classes, boost::shared_ptr<Dict>& dict, VectorReal& class_bias);

} // namespace oxlm
