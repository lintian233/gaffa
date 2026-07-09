#pragma once

#include <cstddef>
#include <functional>
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

struct FfaTransformRowView {
  // Row index within the materialized FFA transform result.
  std::size_t shift = 0;

  // One transform result row. Equivalent to materialized transform[shift, :].
  // Valid only for the callback duration.
  std::span<const float> profile;
};

using FfaTransformRowConsumer =
    std::function<void(const FfaTransformRowView&)>;

// Executes one rows x bins FFA transform block without allocating memory.
// input must contain exactly rows * bins values. scratch and output must each
// contain at least rows * bins values.
void ffa_transform_block_cpu(std::span<const float> input,
                             FfaTransformShape shape,
                             std::span<float> scratch,
                             std::span<float> output);

// Emits final rows from one FFA transform block without materializing the final
// output block. Each emitted row is equivalent to the matching row in
// ffa_transform_block_cpu() output; intermediate recursive rows are not exposed.
// scratch and work must each contain at least rows * bins values. row_buffer
// must contain at least bins values. rows_eval must satisfy 0 < rows_eval <=
// rows.
void for_each_ffa_transform_row_cpu(
    std::span<const float> input,
    FfaTransformShape shape,
    std::size_t rows_eval,
    std::span<float> scratch,
    std::span<float> work,
    std::span<float> row_buffer,
    const FfaTransformRowConsumer& consumer);

FfaTransformResult<float> ffa_transform_cpu(
    std::span<const float> time_series,
    const FfaTransformPlan& plan);

std::vector<double> ffa_trial_periods(
    FfaTransformShape shape,
    double tsamp);

}  // namespace gaffa
