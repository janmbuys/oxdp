#pragma once

#include <chrono>
#include <unordered_map>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "third_party/eigen/Eigen/Dense"
#include "third_party/eigen/Eigen/Sparse"

#include "corpus/utils.h"
#include "corpus/model_config.h"
#include "corpus/corpus.h"
#include "lbl/operators.h"

using namespace std;
using namespace chrono;

namespace oxlm {

typedef vector<vector<int>>                        GlobalFeatureIndexes;
typedef boost::shared_ptr<GlobalFeatureIndexes>    GlobalFeatureIndexesPtr;
typedef unordered_map<int, vector<int>>            MinibatchFeatureIndexes;
typedef boost::shared_ptr<MinibatchFeatureIndexes> MinibatchFeatureIndexesPtr;

typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;
typedef Eigen::Map<VectorReal>                              VectorRealMap;
typedef Eigen::Array<Real, Eigen::Dynamic, 1>               ArrayReal;
typedef Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>  Array2DReal;
typedef Eigen::SparseVector<Real>                           SparseVectorReal;

// Helper operations on vectors.

inline VectorReal softMax(const VectorReal& v) {
  Real max = v.maxCoeff();
  return (v.array() - (log((v.array() - max).exp().sum()) + max)).exp();
}

inline VectorReal logSoftMax(const VectorReal& v) {
  Real max = v.maxCoeff();
  Real log_z = log((v.array() - max).exp().sum()) + max;
  return v.array() - log_z;
}

inline VectorReal logSoftMax(const VectorReal& v, Real& log_z) {
  Real max = v.maxCoeff();
  log_z = log((v.array() - max).exp().sum()) + max;
  return v.array() - log_z;
}

template<class Matrix>
inline Matrix sigmoid(const Matrix& v) {
  return (1.0 + (-v).array().exp()).inverse();
}

//input is sigmoid(x)
inline Array2DReal sigmoidDerivative(const MatrixReal& v) {
  return v.array() * (1 - v.array());
}

template<class Matrix>
inline Matrix rectifier(const Matrix& v) {
  return v.unaryExpr(CwiseRectifierOp<Real>());
}

inline Array2DReal rectifierDerivative(const MatrixReal& v) {
  return v.unaryExpr(CwiseRectifierDerivativeOp<Real>());
}

template<class Matrix>
inline Matrix tanh(const Matrix& v) {
  Matrix w = (-2.0*v).array().exp();
  return (1.0 - w.array())*(1.0 + w.array()).inverse();
}

inline Array2DReal tanhDerivative(const MatrixReal& v) {
  return 1 - v.array()*v.array(); 
}

template<class Matrix>
inline Matrix applyActivation(
    Activation activation, const Matrix& v) {
  switch (activation) {
    case Activation::linear:
      return v;
    case Activation::sigmoid:
      return sigmoid(v);
    case Activation::rectifier:
      return rectifier(v);
    case Activation::tanh:
      return tanh(v);
    default:
      return v;
  }
}

// Note: v here is the hidden layer after the activation has been applied.
// Be careful how you define future activations.
inline Array2DReal activationDerivative(
    Activation activation, const MatrixReal& v) {
  switch (activation) {
    case Activation::linear:
      return Array2DReal::Ones(v.rows(), v.cols());
    case Activation::sigmoid:
      return sigmoidDerivative(v);
    case Activation::rectifier:
      return rectifierDerivative(v);
    case Activation::tanh:
      return tanhDerivative(v);
    default:
      return v;
  }
}

class NotImplementedException : public exception {
  virtual const char* what() const throw() {
    return "This method was not implemented";
  }
};

} // namespace oxlm
