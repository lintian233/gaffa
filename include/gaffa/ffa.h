#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct FfaTransformShape {
  std::size_t rows = 0;
  std::size_t bins = 0;
};

struct FfaTransformPlan {
  std::size_t bins = 0;
};

template <typename T>
struct FfaTransformResult {
  std::vector<T> data;
  FfaTransformShape shape{};
};

FfaTransformResult<float> ffa_transform_cpu(
    std::span<const float> time_series,
    const FfaTransformPlan& plan);

std::vector<double> ffa_trial_periods(
    FfaTransformShape shape,
    double tsamp);

}  // namespace gaffa
