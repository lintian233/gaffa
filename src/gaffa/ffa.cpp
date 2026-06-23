// The core FFA transform in this file is derived from riptide
// (https://github.com/v-morello/riptide), licensed under the MIT License.

#include "gaffa/ffa.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace gaffa {
namespace {

struct ConstBlockView {
  const float* data = nullptr;
  std::size_t rows = 0;
  std::size_t bins = 0;

  const float* row_ptr(std::size_t row) const {
    return data + row * bins;
  }

  std::size_t head_rows() const {
    return rows / 2;
  }

  ConstBlockView head() const {
    return ConstBlockView{data, head_rows(), bins};
  }

  ConstBlockView tail() const {
    const std::size_t head_count = head_rows();
    return ConstBlockView{row_ptr(head_count), rows - head_count, bins};
  }
};

struct BlockView {
  float* data = nullptr;
  std::size_t rows = 0;
  std::size_t bins = 0;

  float* row_ptr(std::size_t row) const {
    return data + row * bins;
  }

  std::size_t head_rows() const {
    return rows / 2;
  }

  BlockView head() const {
    return BlockView{data, head_rows(), bins};
  }

  BlockView tail() const {
    const std::size_t head_count = head_rows();
    return BlockView{row_ptr(head_count), rows - head_count, bins};
  }

  ConstBlockView as_const() const {
    return ConstBlockView{data, rows, bins};
  }
};

void validate_shape(FfaTransformShape shape) {
  if (shape.rows == 0) {
    throw std::invalid_argument("FFA transform rows must be > 0");
  }
  if (shape.bins <= 1) {
    throw std::invalid_argument("FFA transform bins must be > 1");
  }
}

void add(const float* lhs, const float* rhs, std::size_t size, float* output) {
  for (std::size_t index = 0; index < size; ++index) {
    output[index] = lhs[index] + rhs[index];
  }
}

void fused_rollback_add(const float* lhs,
                        const float* rhs,
                        std::size_t size,
                        std::size_t shift,
                        float* output) {
  const std::size_t rolled = shift % size;
  const std::size_t first_count = size - rolled;
  add(lhs, rhs + rolled, first_count, output);
  add(lhs + first_count, rhs, rolled, output + first_count);
}

void merge(ConstBlockView head, ConstBlockView tail, BlockView output) {
  const std::size_t rows = output.rows;
  const std::size_t bins = output.bins;
  const float head_scale =
      static_cast<float>(head.rows - 1) / static_cast<float>(rows - 1);
  const float tail_scale =
      static_cast<float>(tail.rows - 1) / static_cast<float>(rows - 1);

  for (std::size_t shift = 0; shift < rows; ++shift) {
    const auto head_shift =
        static_cast<std::size_t>(head_scale * static_cast<float>(shift) + 0.5F);
    const auto tail_shift =
        static_cast<std::size_t>(tail_scale * static_cast<float>(shift) + 0.5F);
    const std::size_t compensation = shift - (head_shift + tail_shift);

    fused_rollback_add(head.row_ptr(head_shift), tail.row_ptr(tail_shift), bins,
                       head_shift + compensation, output.row_ptr(shift));
  }
}

void transform_recursive(ConstBlockView input, BlockView scratch, BlockView output) {
  const std::size_t rows = input.rows;
  const std::size_t bins = input.bins;

  if (rows == 1) {
    std::memcpy(output.data, input.data, bins * sizeof(float));
    return;
  }

  if (rows == 2) {
    add(input.row_ptr(0), input.row_ptr(1), bins, output.row_ptr(0));
    fused_rollback_add(input.row_ptr(0), input.row_ptr(1), bins, 1,
                       output.row_ptr(1));
    return;
  }

  transform_recursive(input.head(), output.head(), scratch.head());
  transform_recursive(input.tail(), output.tail(), scratch.tail());
  merge(scratch.head().as_const(), scratch.tail().as_const(), output);
}

void validate_transform_buffers(std::span<const float> input,
                                std::span<float> scratch,
                                std::span<float> output,
                                FfaTransformShape shape) {
  validate_shape(shape);

  const std::size_t expected_size = shape.rows * shape.bins;
  if (input.size() != expected_size) {
    throw std::invalid_argument("FFA input size does not match transform shape");
  }
  if (scratch.size() < expected_size) {
    throw std::invalid_argument("FFA scratch buffer is too small");
  }
  if (output.size() < expected_size) {
    throw std::invalid_argument("FFA output buffer is too small");
  }
}

}  // namespace

void ffa_transform_block_cpu(std::span<const float> input,
                             FfaTransformShape shape,
                             std::span<float> scratch,
                             std::span<float> output) {
  validate_transform_buffers(input, scratch, output, shape);

  transform_recursive(ConstBlockView{input.data(), shape.rows, shape.bins},
                      BlockView{scratch.data(), shape.rows, shape.bins},
                      BlockView{output.data(), shape.rows, shape.bins});
}

FfaTransformResult<float> ffa_transform_cpu(
    std::span<const float> time_series,
    const FfaTransformPlan& plan) {
  if (plan.bins <= 1) {
    throw std::invalid_argument("FFA transform bins must be > 1");
  }
  if (plan.bins > time_series.size()) {
    throw std::invalid_argument(
        "FFA transform bins must be <= time series length");
  }

  const std::size_t rows = time_series.size() / plan.bins;
  const std::size_t used_samples = rows * plan.bins;
  const FfaTransformShape shape{
      .rows = rows,
      .bins = plan.bins,
  };

  FfaTransformResult<float> result{
      .data = std::vector<float>(used_samples),
      .shape = shape,
  };
  std::vector<float> scratch(used_samples);

  ffa_transform_block_cpu(time_series.first(used_samples), shape, scratch,
                          result.data);

  return result;
}

std::vector<double> ffa_trial_periods(FfaTransformShape shape, double tsamp) {
  validate_shape(shape);
  if (!(tsamp > 0.0)) {
    throw std::invalid_argument("FFA sampling interval must be > 0");
  }

  std::vector<double> periods(shape.rows);
  if (shape.rows == 1) {
    periods[0] = static_cast<double>(shape.bins) * tsamp;
    return periods;
  }

  const double bins = static_cast<double>(shape.bins);
  const double max_shift = static_cast<double>(shape.rows - 1);
  for (std::size_t shift = 0; shift < shape.rows; ++shift) {
    periods[shift] =
        bins * bins /
        (bins - static_cast<double>(shift) / max_shift) * tsamp;
  }
  return periods;
}

}  // namespace gaffa
