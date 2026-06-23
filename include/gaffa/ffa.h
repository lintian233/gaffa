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

// Executes one rows x bins FFA transform block without allocating memory.
// input must contain exactly rows * bins values. scratch and output must each
// contain at least rows * bins values.
void ffa_transform_block_cpu(std::span<const float> input,
                             FfaTransformShape shape,
                             std::span<float> scratch,
                             std::span<float> output);

FfaTransformResult<float> ffa_transform_cpu(
    std::span<const float> time_series,
    const FfaTransformPlan& plan);

std::vector<double> ffa_trial_periods(
    FfaTransformShape shape,
    double tsamp);

}  // namespace gaffa
