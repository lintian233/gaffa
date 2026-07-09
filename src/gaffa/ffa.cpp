// The core FFA transform in this file is derived from riptide
// (https://github.com/v-morello/riptide), licensed under the MIT License.

#include "gaffa/ffa.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#include <immintrin.h>
#define GAFFA_HAS_X86_AVX2_DISPATCH 1
#endif

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

struct PortableTransformKernel {
  static inline void add(const float* lhs,
                         const float* rhs,
                         std::size_t size,
                         float* output) {
#pragma omp simd
    for (std::size_t index = 0; index < size; ++index) {
      output[index] = lhs[index] + rhs[index];
    }
  }
};

#if defined(GAFFA_HAS_X86_AVX2_DISPATCH)
struct Avx2TransformKernel {
  __attribute__((target("avx2"))) static inline void add(
      const float* lhs,
      const float* rhs,
      std::size_t size,
      float* output) {
    std::size_t index = 0;
    for (; index + 8 <= size; index += 8) {
      const __m256 lhs_values = _mm256_loadu_ps(lhs + index);
      const __m256 rhs_values = _mm256_loadu_ps(rhs + index);
      _mm256_storeu_ps(output + index,
                       _mm256_add_ps(lhs_values, rhs_values));
    }
    for (; index < size; ++index) {
      output[index] = lhs[index] + rhs[index];
    }
  }
};
#endif

template <typename Kernel>
void fused_rollback_add(const float* lhs,
                        const float* rhs,
                        std::size_t size,
                        std::size_t shift,
                        float* output) {
  const std::size_t rolled = shift % size;
  if (rolled == 0) {
    Kernel::add(lhs, rhs, size, output);
    return;
  }

  const std::size_t first_count = size - rolled;
  Kernel::add(lhs, rhs + rolled, first_count, output);
  Kernel::add(lhs + first_count, rhs, rolled, output + first_count);
}

template <typename Kernel>
void merge(ConstBlockView head,
           ConstBlockView tail,
           BlockView output) {
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

    fused_rollback_add<Kernel>(head.row_ptr(head_shift),
                               tail.row_ptr(tail_shift), bins,
                               head_shift + compensation,
                               output.row_ptr(shift));
  }
}

template <typename Kernel>
void merge_row(ConstBlockView head,
               ConstBlockView tail,
               std::size_t output_rows,
               std::size_t shift,
               float* output) {
  const std::size_t bins = head.bins;
  const float head_scale =
      static_cast<float>(head.rows - 1) / static_cast<float>(output_rows - 1);
  const float tail_scale =
      static_cast<float>(tail.rows - 1) / static_cast<float>(output_rows - 1);
  const auto head_shift =
      static_cast<std::size_t>(head_scale * static_cast<float>(shift) + 0.5F);
  const auto tail_shift =
      static_cast<std::size_t>(tail_scale * static_cast<float>(shift) + 0.5F);
  const std::size_t compensation = shift - (head_shift + tail_shift);

  fused_rollback_add<Kernel>(head.row_ptr(head_shift), tail.row_ptr(tail_shift),
                             bins, head_shift + compensation, output);
}

template <typename Kernel>
void transform_recursive(ConstBlockView input,
                         BlockView scratch,
                         BlockView output) {
  const std::size_t rows = input.rows;
  const std::size_t bins = input.bins;

  if (rows == 1) {
    std::memcpy(output.data, input.data, bins * sizeof(float));
    return;
  }

  if (rows == 2) {
    Kernel::add(input.row_ptr(0), input.row_ptr(1), bins, output.row_ptr(0));
    fused_rollback_add<Kernel>(input.row_ptr(0), input.row_ptr(1), bins, 1,
                               output.row_ptr(1));
    return;
  }

  transform_recursive<Kernel>(input.head(), output.head(), scratch.head());
  transform_recursive<Kernel>(input.tail(), output.tail(), scratch.tail());
  merge<Kernel>(scratch.head().as_const(), scratch.tail().as_const(), output);
}

void transform_recursive_portable(ConstBlockView input,
                                  BlockView scratch,
                                  BlockView output) {
  transform_recursive<PortableTransformKernel>(input, scratch, output);
}

template <typename Kernel>
void emit_transform_rows(ConstBlockView input,
                         std::size_t rows_eval,
                         BlockView scratch,
                         BlockView work,
                         std::span<float> row_buffer,
                         const FfaTransformRowConsumer& consumer) {
  const std::size_t rows = input.rows;
  const std::size_t bins = input.bins;

  if (rows == 1) {
    consumer(FfaTransformRowView{
        .shift = 0,
        .profile = std::span<const float>(input.row_ptr(0), bins),
    });
    return;
  }

  if (rows == 2) {
    Kernel::add(input.row_ptr(0), input.row_ptr(1), bins, row_buffer.data());
    consumer(FfaTransformRowView{
        .shift = 0,
        .profile = row_buffer.first(bins),
    });

    if (rows_eval > 1) {
      fused_rollback_add<Kernel>(input.row_ptr(0), input.row_ptr(1), bins, 1,
                                 row_buffer.data());
      consumer(FfaTransformRowView{
          .shift = 1,
          .profile = row_buffer.first(bins),
      });
    }
    return;
  }

  transform_recursive<Kernel>(input.head(), scratch.head(), work.head());
  transform_recursive<Kernel>(input.tail(), scratch.tail(), work.tail());

  const ConstBlockView head = work.head().as_const();
  const ConstBlockView tail = work.tail().as_const();
  for (std::size_t shift = 0; shift < rows_eval; ++shift) {
    merge_row<Kernel>(head, tail, rows, shift, row_buffer.data());
    consumer(FfaTransformRowView{
        .shift = shift,
        .profile = row_buffer.first(bins),
    });
  }
}

void emit_transform_rows_portable(ConstBlockView input,
                                  std::size_t rows_eval,
                                  BlockView scratch,
                                  BlockView work,
                                  std::span<float> row_buffer,
                                  const FfaTransformRowConsumer& consumer) {
  emit_transform_rows<PortableTransformKernel>(input, rows_eval, scratch, work,
                                               row_buffer, consumer);
}

#if defined(GAFFA_HAS_X86_AVX2_DISPATCH)
__attribute__((target("avx2"))) void transform_recursive_avx2(
    ConstBlockView input,
    BlockView scratch,
    BlockView output) {
  // Dispatch once per transform block. FFA rows are short, so per-row function
  // pointer dispatch is measurable in the recursive merge hot path.
  transform_recursive<Avx2TransformKernel>(input, scratch, output);
}

__attribute__((target("avx2"))) void emit_transform_rows_avx2(
    ConstBlockView input,
    std::size_t rows_eval,
    BlockView scratch,
    BlockView work,
    std::span<float> row_buffer,
    const FfaTransformRowConsumer& consumer) {
  emit_transform_rows<Avx2TransformKernel>(input, rows_eval, scratch, work,
                                           row_buffer, consumer);
}

bool cpu_supports_avx2() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
}
#endif

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

void validate_transform_row_buffers(std::span<const float> input,
                                    FfaTransformShape shape,
                                    std::size_t rows_eval,
                                    std::span<float> scratch,
                                    std::span<float> work,
                                    std::span<float> row_buffer,
                                    const FfaTransformRowConsumer& consumer) {
  validate_shape(shape);

  if (rows_eval == 0 || rows_eval > shape.rows) {
    throw std::invalid_argument(
        "FFA transform rows_eval must satisfy 0 < rows_eval <= rows");
  }

  const std::size_t expected_size = shape.rows * shape.bins;
  if (input.size() != expected_size) {
    throw std::invalid_argument("FFA input size does not match transform shape");
  }
  if (scratch.size() < expected_size) {
    throw std::invalid_argument("FFA scratch buffer is too small");
  }
  if (work.size() < expected_size) {
    throw std::invalid_argument("FFA work buffer is too small");
  }
  if (row_buffer.size() < shape.bins) {
    throw std::invalid_argument("FFA row buffer is too small");
  }
  if (!consumer) {
    throw std::invalid_argument("FFA transform row consumer must be callable");
  }
}

}  // namespace

void ffa_transform_block_cpu(std::span<const float> input,
                             FfaTransformShape shape,
                             std::span<float> scratch,
                             std::span<float> output) {
  validate_transform_buffers(input, scratch, output, shape);

  const ConstBlockView input_view{input.data(), shape.rows, shape.bins};
  const BlockView scratch_view{scratch.data(), shape.rows, shape.bins};
  const BlockView output_view{output.data(), shape.rows, shape.bins};

#if defined(GAFFA_HAS_X86_AVX2_DISPATCH)
  if (cpu_supports_avx2()) {
    transform_recursive_avx2(input_view, scratch_view, output_view);
    return;
  }
#endif

  transform_recursive_portable(input_view, scratch_view, output_view);
}

void for_each_ffa_transform_row_cpu(
    std::span<const float> input,
    FfaTransformShape shape,
    std::size_t rows_eval,
    std::span<float> scratch,
    std::span<float> work,
    std::span<float> row_buffer,
    const FfaTransformRowConsumer& consumer) {
  validate_transform_row_buffers(input, shape, rows_eval, scratch, work,
                                 row_buffer, consumer);

  const ConstBlockView input_view{input.data(), shape.rows, shape.bins};
  const BlockView scratch_view{scratch.data(), shape.rows, shape.bins};
  const BlockView work_view{work.data(), shape.rows, shape.bins};

#if defined(GAFFA_HAS_X86_AVX2_DISPATCH)
  if (cpu_supports_avx2()) {
    emit_transform_rows_avx2(input_view, rows_eval, scratch_view, work_view,
                             row_buffer, consumer);
    return;
  }
#endif

  emit_transform_rows_portable(input_view, rows_eval, scratch_view, work_view,
                               row_buffer, consumer);
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
