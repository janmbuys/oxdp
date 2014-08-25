#include "lbl/global_factored_maxent_weights.h"

#include <iomanip>

#include <boost/make_shared.hpp>

#include "lbl/bloom_filter.h"
#include "lbl/bloom_filter_populator.h"
#include "lbl/class_context_extractor.h"
#include "lbl/class_context_hasher.h"
#include "lbl/collision_global_feature_store.h"
#include "lbl/feature_approximate_filter.h"
#include "lbl/feature_context_mapper.h"
#include "lbl/feature_exact_filter.h"
#include "lbl/feature_filter.h"
#include "lbl/feature_matcher.h"
#include "lbl/feature_no_op_filter.h"
#include "lbl/sparse_global_feature_store.h"
#include "lbl/unconstrained_feature_store.h"
#include "lbl/word_context_extractor.h"
#include "lbl/word_context_hasher.h"

namespace oxlm {

GlobalFactoredMaxentWeights::GlobalFactoredMaxentWeights() {}

GlobalFactoredMaxentWeights::GlobalFactoredMaxentWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata)
    : FactoredWeights(config, metadata), metadata(metadata) {
  initialize();
}

GlobalFactoredMaxentWeights::GlobalFactoredMaxentWeights(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<FactoredMaxentMetadata>& metadata,
    const boost::shared_ptr<Corpus>& training_corpus)
    : FactoredWeights(config, metadata, training_corpus), metadata(metadata) {
  initialize();
}

void GlobalFactoredMaxentWeights::initialize() {
  int num_classes = index->getNumClasses();
  V.resize(num_classes);
  boost::shared_ptr<FeatureContextMapper> mapper = metadata->getMapper();
  boost::shared_ptr<BloomFilterPopulator> populator = metadata->getPopulator();
  boost::shared_ptr<FeatureMatcher> matcher = metadata->getMatcher();

  if (config->hash_space) {
    boost::shared_ptr<GlobalCollisionSpace> space =
        boost::make_shared<GlobalCollisionSpace>(config->hash_space);
    boost::shared_ptr<FeatureContextHasher> hasher =
        boost::make_shared<ClassContextHasher>(config->hash_space);
    GlobalFeatureIndexesPairPtr feature_indexes_pair;
    boost::shared_ptr<BloomFilter<NGram>> bloom_filter;
    boost::shared_ptr<FeatureFilter> filter;
    if (config->filter_contexts) {
      if (config->filter_error_rate > 0) {
        bloom_filter = populator->get();
        filter = boost::make_shared<FeatureApproximateFilter>(
            num_classes, hasher, bloom_filter);
      } else {
        feature_indexes_pair = matcher->getGlobalFeatures();
        filter = boost::make_shared<FeatureExactFilter>(
            feature_indexes_pair->getClassIndexes(),
            boost::make_shared<ClassContextExtractor>(mapper));
      }
    } else {
      filter = boost::make_shared<FeatureNoOpFilter>(num_classes);
    }

    U = boost::make_shared<CollisionGlobalFeatureStore>(
        num_classes, config->hash_space, config->feature_context_size,
        space, hasher, filter);

    for (int i = 0; i < num_classes; ++i) {
      int class_size = index->getClassSize(i);
      hasher = boost::make_shared<WordContextHasher>(i, config->hash_space);
      if (config->filter_contexts) {
        if (config->filter_error_rate) {
          filter = boost::make_shared<FeatureApproximateFilter>(
              class_size, hasher, bloom_filter);
        } else {
          filter = boost::make_shared<FeatureExactFilter>(
              feature_indexes_pair->getWordIndexes(i),
              boost::make_shared<WordContextExtractor>(i, mapper));
        }
      } else {
        filter = boost::make_shared<FeatureNoOpFilter>(class_size);
      }
      V[i] = boost::make_shared<CollisionGlobalFeatureStore>(
          class_size, config->hash_space, config->feature_context_size,
          space, hasher, filter);
    }
  } else if (config->sparse_features) {
    auto feature_indexes_pair = matcher->getGlobalFeatures();
    U = boost::make_shared<SparseGlobalFeatureStore>(
        num_classes,
        feature_indexes_pair->getClassIndexes(),
        boost::make_shared<ClassContextExtractor>(mapper));

    for (int i = 0; i < num_classes; ++i) {
      V[i] = boost::make_shared<SparseGlobalFeatureStore>(
          index->getClassSize(i),
          feature_indexes_pair->getWordIndexes(i),
          boost::make_shared<WordContextExtractor>(i, mapper));
    }
  } else {
    U = boost::make_shared<UnconstrainedFeatureStore>(
        num_classes,
        boost::make_shared<ClassContextExtractor>(mapper));

    for (int i = 0; i < num_classes; ++i) {
      V[i] = boost::make_shared<UnconstrainedFeatureStore>(
          index->getClassSize(i),
          boost::make_shared<WordContextExtractor>(i, mapper));
    }
  }
}

void GlobalFactoredMaxentWeights::getProbabilities(
    const vector<WordId>& words,
    const vector<vector<int>>& contexts,
    const MatrixReal& prediction_vectors,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs) const {
  class_probs = S.transpose() * prediction_vectors + T * MatrixReal::Ones(1, words.size());

  for (size_t i = 0; i < words.size(); ++i) {
    int word_id = words[i];
    int class_id = index->getClass(word_id);

    VectorReal prediction_vector = prediction_vectors.col(i);
    VectorReal class_scores = class_probs.col(i) + U->get(contexts[i]);
    class_probs.col(i) = softMax(class_scores);

    VectorReal word_scores = classR(class_id).transpose() * prediction_vector +
                             classB(class_id) + V[class_id]->get(contexts[i]);
    word_probs.push_back(softMax(word_scores));
  }
}

boost::shared_ptr<MinibatchFactoredMaxentWeights> GlobalFactoredMaxentWeights::getGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    Real& objective) const {
  vector<WordId> words;
  vector<vector<WordId>> contexts;
  vector<MatrixReal> context_vectors;
  MatrixReal prediction_vectors;
  MatrixReal class_probs;
  vector<VectorReal> word_probs;
  objective = getObjective(
      corpus, indices, words, contexts, context_vectors, prediction_vectors,
      class_probs, word_probs);

  MatrixReal weighted_representations = getWeightedRepresentations(
      words, prediction_vectors, class_probs, word_probs);

  return getFullGradient(
      corpus, indices, words, contexts, context_vectors, prediction_vectors,
      weighted_representations, class_probs, word_probs);
}

boost::shared_ptr<MinibatchFactoredMaxentWeights> GlobalFactoredMaxentWeights::getFullGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const vector<WordId>& words,
    const vector<vector<int>>& contexts,
    const vector<MatrixReal>& context_vectors,
    const MatrixReal& prediction_vectors,
    const MatrixReal& weighted_representations,
    MatrixReal& class_probs,
    vector<VectorReal>& word_probs) const {
  boost::shared_ptr<FactoredWeights> base_gradient =
      FactoredWeights::getFullGradient(
          words, contexts, context_vectors, prediction_vectors,
          weighted_representations, class_probs, word_probs);

  boost::shared_ptr<MinibatchFactoredMaxentWeights> gradient =
      boost::make_shared<MinibatchFactoredMaxentWeights>(
          config, metadata, corpus, indices, base_gradient);

  for (size_t i = 0; i < words.size(); ++i) {
    int word_id = words[i];
    int class_id = index->getClass(word_id);

    gradient->U->update(contexts[i], class_probs.col(i));
    gradient->V[class_id]->update(contexts[i], word_probs[i]);
  }

  return gradient;
}

bool GlobalFactoredMaxentWeights::checkGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient,
    Real eps) {
  if (!FactoredWeights::checkGradient(corpus, indices, gradient, eps)) {
    return false;
  }

  if (config->hash_space == 0) {
    // If no hashing is used, check gradients individually.
    if (!checkGradientStore(corpus, indices, U, gradient->U, eps)) {
      return false;
    }

    for (size_t i = 0; i < V.size(); ++i) {
      if (!checkGradientStore(corpus, indices, V[i], gradient->V[i], eps)) {
        return false;
      }
    }
  } else {
    // Class and word features are hashed in the same global collision space, so
    // we can only verify if the sum of gradients is correct.
    boost::shared_ptr<MinibatchFeatureStore> gradient_sum = gradient->U;
    for (size_t i = 0; i < V.size(); ++i) {
      gradient_sum->update(gradient->V[i]);
    }

    if (!checkGradientStore(corpus, indices, U, gradient_sum, eps)) {
      return false;
    }
  }

  return true;
}

bool GlobalFactoredMaxentWeights::checkGradientStore(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    const boost::shared_ptr<GlobalFeatureStore>& store,
    const boost::shared_ptr<MinibatchFeatureStore>& gradient_store,
    Real eps) {
  vector<pair<int, int>> feature_indexes = store->getFeatureIndexes();
  for (const auto& index: feature_indexes) {
    store->updateFeature(index, eps);
    Real objective_plus = getObjective(corpus, indices);
    store->updateFeature(index, -eps);

    store->updateFeature(index, -eps);
    Real objective_minus = getObjective(corpus, indices);
    store->updateFeature(index, eps);

    double est_gradient = (objective_plus - objective_minus) / (2 * eps);
    if (fabs(gradient_store->getFeature(index) - est_gradient) > eps) {
      return false;
    }
  }

  return true;
}

boost::shared_ptr<MinibatchFactoredMaxentWeights> GlobalFactoredMaxentWeights::estimateGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    Real& objective) const {
  throw NotImplementedException();
}

void GlobalFactoredMaxentWeights::updateSquared(
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& global_gradient) {
  FactoredWeights::updateSquared(global_gradient);

  U->updateSquared(global_gradient->U);
  for (size_t i = 0; i < V.size(); ++i) {
    V[i]->updateSquared(global_gradient->V[i]);
  }
}

void GlobalFactoredMaxentWeights::updateAdaGrad(
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& global_gradient,
    const boost::shared_ptr<GlobalFactoredMaxentWeights>& adagrad) {
  FactoredWeights::updateAdaGrad(global_gradient, adagrad);

  U->updateAdaGrad(global_gradient->U, adagrad->U, config->step_size);
  for (size_t i = 0; i < V.size(); ++i) {
    V[i]->updateAdaGrad(global_gradient->V[i], adagrad->V[i], config->step_size);
  }
}

Real GlobalFactoredMaxentWeights::regularizerUpdate(
    const boost::shared_ptr<MinibatchFactoredMaxentWeights>& global_gradient,
    Real minibatch_factor) {
  Real ret = FactoredWeights::regularizerUpdate(
      global_gradient, minibatch_factor);

  Real sigma = minibatch_factor * config->step_size * config->l2_maxent;
  Real factor = 0.5 * minibatch_factor * config->l2_maxent;
  U->l2GradientUpdate(global_gradient->U, sigma);
  ret += U->l2Objective(global_gradient->U, factor);
  for (size_t i = 0; i < V.size(); ++i) {
    V[i]->l2GradientUpdate(global_gradient->V[i], sigma);
    ret += V[i]->l2Objective(global_gradient->V[i], factor);
  }

  return ret;
}

Real GlobalFactoredMaxentWeights::predict(
    int word_id, vector<int> context) const {
  int class_id = index->getClass(word_id);
  int word_class_id = index->getWordIndexInClass(word_id);
  VectorReal prediction_vector = getPredictionVector(context);

  Real class_prob = 0;
  auto ret = normalizerCache.get(context);
  if (ret.second) {
    Real class_score = U->get(context)(class_id);
    class_prob = S.col(class_id).dot(prediction_vector) + T(class_id) + class_score - ret.first;
  } else {
    Real normalizer = 0;
    VectorReal class_probs = logSoftMax(
        S.transpose() * prediction_vector + T + U->get(context), normalizer);
    normalizerCache.set(context, normalizer);
    class_prob = class_probs(class_id);
  }

  context.insert(context.begin(), class_id);
  Real word_prob = 0;
  ret = classNormalizerCache.get(context);
  if (ret.second) {
    Real word_score = V[class_id]->get(context)(word_class_id);
    word_prob = R.col(word_id).dot(prediction_vector) + B(word_id) + word_score - ret.first;
  } else {
    Real normalizer = 0;
    VectorReal word_probs = logSoftMax(
        classR(class_id).transpose() * prediction_vector + classB(class_id) + V[class_id]->get(context),
        normalizer);
    classNormalizerCache.set(context, normalizer);
    word_prob = word_probs(word_class_id);
  }

  return class_prob + word_prob;
}

bool GlobalFactoredMaxentWeights::operator==(
    const GlobalFactoredMaxentWeights& other) const {
  if (V.size() != other.V.size()) {
    return false;
  }

  for (size_t i = 0; i < V.size(); ++i) {
    if (!(V[i]->operator==(other.V[i]))) {
      return false;
    }
  }

  return FactoredWeights::operator==(other)
      && *metadata == *other.metadata
      && U->operator==(other.U);
}

} // namespace oxlm