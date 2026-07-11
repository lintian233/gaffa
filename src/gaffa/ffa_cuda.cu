#include "gaffa/ffa_cuda.h"

#include "gaffa/cuda_memory.h"
#include "gaffa/time_series.h"
#include "gaffa/time_series_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gaffa {
namespace {

enum class FfaCudaBufferRole : std::uint8_t {
  Input,
  Scratch,
  Output,
};

struct FfaCudaCopyOp {
  FfaCudaBufferRole input_role = FfaCudaBufferRole::Input;
  FfaCudaBufferRole output_role = FfaCudaBufferRole::Output;
  std::size_t input_begin_row = 0;
  std::size_t output_begin_row = 0;
  std::size_t rows = 0;
};

struct FfaCudaMergeOp {
  FfaCudaBufferRole head_role = FfaCudaBufferRole::Input;
  FfaCudaBufferRole tail_role = FfaCudaBufferRole::Input;
  FfaCudaBufferRole output_role = FfaCudaBufferRole::Output;
  std::size_t head_begin_row = 0;
  std::size_t tail_begin_row = 0;
  std::size_t output_begin_row = 0;
  std::size_t head_rows = 0;
  std::size_t tail_rows = 0;
  std::size_t output_rows = 0;
};

struct FfaCudaTransformLevel {
  std::vector<FfaCudaCopyOp> copy_ops;
  std::vector<FfaCudaMergeOp> merge_ops;
};

struct FfaCudaDeviceTransformLevel {
  CudaDeviceBuffer<FfaCudaCopyOp> copy_ops;
  std::size_t copy_op_count = 0;
  std::size_t max_copy_elements = 0;
  CudaDeviceBuffer<FfaCudaMergeOp> merge_ops;
  std::size_t merge_op_count = 0;
  std::size_t max_merge_elements = 0;
};

__device__ const float* select_const_buffer(FfaCudaBufferRole role,
                                            const float* input,
                                            const float* scratch,
                                            const float* output) {
  switch (role) {
    case FfaCudaBufferRole::Input:
      return input;
    case FfaCudaBufferRole::Scratch:
      return scratch;
    case FfaCudaBufferRole::Output:
      return output;
  }
  return nullptr;
}

__device__ float* select_mutable_buffer(FfaCudaBufferRole role,
                                        float* scratch,
                                        float* output) {
  switch (role) {
    case FfaCudaBufferRole::Scratch:
      return scratch;
    case FfaCudaBufferRole::Output:
      return output;
    case FfaCudaBufferRole::Input:
      return nullptr;
  }
  return nullptr;
}

__global__ void prepare_copy_float_kernel(const float* input,
                                          std::size_t input_nsamples,
                                          float* output,
                                          std::size_t output_nsamples) {
  const std::size_t output_index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t series = blockIdx.y;
  if (output_index >= output_nsamples) {
    return;
  }

  output[series * output_nsamples + output_index] =
      input[series * input_nsamples + output_index];
}

__global__ void ffa_copy_level_kernel(const float* input,
                                      float* scratch,
                                      float* output,
                                      std::size_t input_stride,
                                      std::size_t scratch_stride,
                                      std::size_t output_stride,
                                      std::size_t bins,
                                      const FfaCudaCopyOp* ops,
                                      std::size_t op_count,
                                      std::size_t max_op_elements) {
  const std::size_t element =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t op_index = blockIdx.y;
  const std::size_t series = blockIdx.z;
  if (op_index >= op_count || element >= max_op_elements) {
    return;
  }

  const FfaCudaCopyOp op = ops[op_index];
  const std::size_t op_elements = op.rows * bins;
  if (element >= op_elements) {
    return;
  }

  const std::size_t row = element / bins;
  const std::size_t bin = element % bins;
  const std::size_t source_stride =
      op.input_role == FfaCudaBufferRole::Input
          ? input_stride
          : (op.input_role == FfaCudaBufferRole::Scratch ? scratch_stride
                                                         : output_stride);
  const std::size_t destination_stride =
      op.output_role == FfaCudaBufferRole::Scratch ? scratch_stride
                                                   : output_stride;
  const float* source_base =
      select_const_buffer(op.input_role, input, scratch, output) +
      series * source_stride;
  float* output_base =
      select_mutable_buffer(op.output_role, scratch, output) +
      series * destination_stride;

  output_base[(op.output_begin_row + row) * bins + bin] =
      source_base[(op.input_begin_row + row) * bins + bin];
}

__global__ void ffa_merge_level_kernel(const float* input,
                                       float* scratch,
                                       float* output,
                                       std::size_t input_stride,
                                       std::size_t scratch_stride,
                                       std::size_t output_stride,
                                       std::size_t bins,
                                       const FfaCudaMergeOp* ops,
                                       std::size_t op_count,
                                       std::size_t max_op_elements) {
  const std::size_t element =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t op_index = blockIdx.y;
  const std::size_t series = blockIdx.z;
  if (op_index >= op_count || element >= max_op_elements) {
    return;
  }

  const FfaCudaMergeOp op = ops[op_index];
  const std::size_t op_elements = op.output_rows * bins;
  if (element >= op_elements) {
    return;
  }

  const std::size_t shift = element / bins;
  const std::size_t bin = element % bins;
  const float head_scale = static_cast<float>(op.head_rows - 1) /
                           static_cast<float>(op.output_rows - 1);
  const float tail_scale = static_cast<float>(op.tail_rows - 1) /
                           static_cast<float>(op.output_rows - 1);
  const auto head_shift = static_cast<std::size_t>(
      head_scale * static_cast<float>(shift) + 0.5F);
  const auto tail_shift = static_cast<std::size_t>(
      tail_scale * static_cast<float>(shift) + 0.5F);
  const std::size_t compensation = shift - (head_shift + tail_shift);
  const std::size_t rolled = (head_shift + compensation) % bins;

  const std::size_t head_stride =
      op.head_role == FfaCudaBufferRole::Input
          ? input_stride
          : (op.head_role == FfaCudaBufferRole::Scratch ? scratch_stride
                                                        : output_stride);
  const std::size_t tail_stride =
      op.tail_role == FfaCudaBufferRole::Input
          ? input_stride
          : (op.tail_role == FfaCudaBufferRole::Scratch ? scratch_stride
                                                        : output_stride);
  const std::size_t destination_stride =
      op.output_role == FfaCudaBufferRole::Scratch ? scratch_stride
                                                   : output_stride;
  const float* head_base =
      select_const_buffer(op.head_role, input, scratch, output) +
      series * head_stride;
  const float* tail_base =
      select_const_buffer(op.tail_role, input, scratch, output) +
      series * tail_stride;
  float* output_base =
      select_mutable_buffer(op.output_role, scratch, output) +
      series * destination_stride;

  output_base[(op.output_begin_row + shift) * bins + bin] =
      head_base[(op.head_begin_row + head_shift) * bins + bin] +
      tail_base[(op.tail_begin_row + tail_shift) * bins +
                ((bin + rolled) % bins)];
}

std::size_t checked_multiply(std::size_t lhs,
                             std::size_t rhs,
                             const char* message);

std::size_t checked_multiply(std::size_t lhs,
                             std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs,
                        std::size_t rhs,
                        const char* message) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::overflow_error(message);
  }
  return lhs + rhs;
}

std::size_t checked_float_bytes(std::size_t count, const char* message) {
  return checked_multiply(count, sizeof(float), message);
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void validate_program_options(const CudaFfaProgramOptions& options) {
  if (options.device_id < 0) {
    throw std::invalid_argument("CUDA FFA program device_id must be >= 0");
  }
}

void validate_execution_options(const CudaFfaExecutionOptions& options) {
  if (options.series_tile_size == 0) {
    throw std::invalid_argument("CUDA FFA series_tile_size must be > 0");
  }
  if (options.threads_per_block == 0) {
    throw std::invalid_argument("CUDA FFA threads_per_block must be > 0");
  }
}

void validate_launch_options(const CudaLaunchOptions& options) {
  if (options.device_id < 0) {
    throw std::invalid_argument("CUDA FFA launch device_id must be >= 0");
  }
  if (options.threads_per_block == 0) {
    throw std::invalid_argument(
        "CUDA FFA launch threads_per_block must be > 0");
  }
}

unsigned int checked_grid_dim(std::size_t value, const char* message) {
  if (value == 0 ||
      value > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::overflow_error(message);
  }
  return static_cast<unsigned int>(value);
}

bool is_no_downsample(double factor) {
  return factor == 1.0;
}

void append_level(FfaCudaTransformLevel& output,
                  const FfaCudaTransformLevel& input) {
  output.copy_ops.insert(output.copy_ops.end(), input.copy_ops.begin(),
                         input.copy_ops.end());
  output.merge_ops.insert(output.merge_ops.end(), input.merge_ops.begin(),
                          input.merge_ops.end());
}

std::vector<FfaCudaTransformLevel> build_transform_levels(
    FfaCudaBufferRole input_role,
    FfaCudaBufferRole scratch_role,
    FfaCudaBufferRole output_role,
    std::size_t row_begin,
    std::size_t rows) {
  if (rows == 1) {
    FfaCudaTransformLevel level;
    level.copy_ops.push_back(FfaCudaCopyOp{
        .input_role = input_role,
        .output_role = output_role,
        .input_begin_row = row_begin,
        .output_begin_row = row_begin,
        .rows = 1,
    });
    return {level};
  }

  if (rows == 2) {
    FfaCudaTransformLevel level;
    level.merge_ops.push_back(FfaCudaMergeOp{
        .head_role = input_role,
        .tail_role = input_role,
        .output_role = output_role,
        .head_begin_row = row_begin,
        .tail_begin_row = row_begin + 1,
        .output_begin_row = row_begin,
        .head_rows = 1,
        .tail_rows = 1,
        .output_rows = 2,
    });
    return {level};
  }

  const std::size_t head_rows = rows / 2;
  const std::size_t tail_rows = rows - head_rows;
  auto head_levels = build_transform_levels(
      input_role, output_role, scratch_role, row_begin, head_rows);
  auto tail_levels = build_transform_levels(
      input_role, output_role, scratch_role, row_begin + head_rows, tail_rows);

  std::vector<FfaCudaTransformLevel> levels(
      std::max(head_levels.size(), tail_levels.size()));
  for (std::size_t index = 0; index < head_levels.size(); ++index) {
    append_level(levels[index], head_levels[index]);
  }
  for (std::size_t index = 0; index < tail_levels.size(); ++index) {
    append_level(levels[index], tail_levels[index]);
  }

  FfaCudaTransformLevel merge_level;
  merge_level.merge_ops.push_back(FfaCudaMergeOp{
      .head_role = scratch_role,
      .tail_role = scratch_role,
      .output_role = output_role,
      .head_begin_row = row_begin,
      .tail_begin_row = row_begin + head_rows,
      .output_begin_row = row_begin,
      .head_rows = head_rows,
      .tail_rows = tail_rows,
      .output_rows = rows,
  });
  levels.push_back(std::move(merge_level));
  return levels;
}

std::vector<FfaCudaTransformLevel> build_transform_levels(
    FfaTransformShape shape) {
  return build_transform_levels(FfaCudaBufferRole::Input,
                                FfaCudaBufferRole::Scratch,
                                FfaCudaBufferRole::Output, 0, shape.rows);
}

std::size_t max_copy_elements(const FfaCudaTransformLevel& level,
                              std::size_t bins) {
  std::size_t max_elements = 0;
  for (const auto& op : level.copy_ops) {
    max_elements = std::max(max_elements, checked_multiply(
                                              op.rows, bins,
                                              "CUDA FFA copy op size overflow"));
  }
  return max_elements;
}

std::size_t max_merge_elements(const FfaCudaTransformLevel& level,
                               std::size_t bins) {
  std::size_t max_elements = 0;
  for (const auto& op : level.merge_ops) {
    max_elements = std::max(
        max_elements,
        checked_multiply(op.output_rows, bins,
                         "CUDA FFA merge op size overflow"));
  }
  return max_elements;
}

void validate_task_prepared_nsamples(const FfaSearchTask& task,
                                     const char* prefix) {
  if (task.input_nsamples == 0) {
    throw std::invalid_argument(std::string(prefix) +
                                " task input_nsamples must be > 0");
  }
  if (task.prepared_nsamples == 0) {
    throw std::invalid_argument(std::string(prefix) +
                                " task prepared_nsamples must be > 0");
  }

  if (is_no_downsample(task.downsample_factor)) {
    if (task.prepared_nsamples != task.input_nsamples) {
      throw std::invalid_argument(
          std::string(prefix) +
          " no-downsample task prepared_nsamples must match input nsamples");
    }
    return;
  }

  const std::size_t expected_prepared =
      downsampled_size(task.input_nsamples, task.downsample_factor);
  if (task.prepared_nsamples != expected_prepared) {
    throw std::invalid_argument(
        std::string(prefix) +
        " task prepared_nsamples must match downsampled_size");
  }
}

void validate_plan_for_workspace(const FfaSearchPlan& plan) {
  if (plan.tasks.empty()) {
    throw std::invalid_argument("CUDA FFA plan must contain at least one task");
  }
  if (plan.width_trials.empty()) {
    throw std::invalid_argument("CUDA FFA plan width_trials must not be empty");
  }
  for (const auto& task : plan.tasks) {
    if (task.bins <= 1) {
      throw std::invalid_argument("CUDA FFA task bins must be > 1");
    }
    if (task.rows == 0) {
      throw std::invalid_argument("CUDA FFA task rows must be > 0");
    }
    if (task.rows_eval == 0 || task.rows_eval > task.rows) {
      throw std::invalid_argument(
          "CUDA FFA task rows_eval must satisfy 0 < rows_eval <= rows");
    }
    if (task.prepared_nsamples == 0) {
      throw std::invalid_argument(
          "CUDA FFA task prepared_nsamples must be > 0");
    }
    const std::size_t task_elements = checked_multiply(
        task.rows, task.bins, "CUDA FFA task shape size overflow");
    if (task_elements > task.prepared_nsamples) {
      throw std::invalid_argument(
          "CUDA FFA task rows * bins must be <= prepared_nsamples");
    }
    if (!(task.effective_tsamp > 0.0) || !std::isfinite(task.effective_tsamp)) {
      throw std::invalid_argument(
          "CUDA FFA task effective_tsamp must be finite and > 0");
    }
    if (!std::isfinite(task.downsample_factor) ||
        task.downsample_factor < 1.0) {
      throw std::invalid_argument(
          "CUDA FFA task downsample_factor must be finite and >= 1");
    }
    validate_task_prepared_nsamples(task, "CUDA FFA");
  }
}

void validate_search_input(CudaTimeSeriesBatchView input,
                           const FfaSearchPlan& plan,
                           const FfaSearchOptions& search_options,
                           const CudaFfaExecutionOptions& cuda_options) {
  if (input.data == nullptr) {
    throw std::invalid_argument("CUDA FFA input data must not be null");
  }
  if (input.nseries == 0) {
    throw std::invalid_argument("CUDA FFA input nseries must be > 0");
  }
  if (input.nsamples == 0) {
    throw std::invalid_argument("CUDA FFA input nsamples must be > 0");
  }
  if (!(input.tsamp > 0.0) || !std::isfinite(input.tsamp)) {
    throw std::invalid_argument("CUDA FFA input tsamp must be finite and > 0");
  }
  if (!std::isfinite(search_options.snr_threshold)) {
    throw std::invalid_argument("CUDA FFA S/N threshold must be finite");
  }
  validate_execution_options(cuda_options);
  validate_plan_for_workspace(plan);
  for (const auto& task : plan.tasks) {
    if (task.input_nsamples != input.nsamples) {
      throw std::invalid_argument(
          "CUDA FFA task input_nsamples must match input nsamples");
    }
  }
}

void validate_prepare_arguments(CudaTimeSeriesBatchView input,
                                const FfaSearchTask& task,
                                CudaSpan<float> output,
                                const CudaLaunchOptions& options) {
  if (input.data == nullptr) {
    throw std::invalid_argument("CUDA FFA prepare input data must not be null");
  }
  if (input.nseries == 0) {
    throw std::invalid_argument("CUDA FFA prepare input nseries must be > 0");
  }
  if (input.nsamples == 0) {
    throw std::invalid_argument("CUDA FFA prepare input nsamples must be > 0");
  }
  if (!(input.tsamp > 0.0) || !std::isfinite(input.tsamp)) {
    throw std::invalid_argument(
        "CUDA FFA prepare input tsamp must be finite and > 0");
  }
  validate_launch_options(options);
  if (input.device_id != options.device_id ||
      output.device_id != options.device_id) {
    throw std::invalid_argument(
        "CUDA FFA prepare device ids must match launch device_id");
  }
  if (output.data == nullptr) {
    throw std::invalid_argument("CUDA FFA prepare output data must not be null");
  }
  if (task.input_nsamples != input.nsamples) {
    throw std::invalid_argument(
        "CUDA FFA prepare task input_nsamples must match input nsamples");
  }
  if (!std::isfinite(task.downsample_factor) ||
      task.downsample_factor < 1.0) {
    throw std::invalid_argument(
        "CUDA FFA prepare downsample_factor must be finite and >= 1");
  }
  validate_task_prepared_nsamples(task, "CUDA FFA prepare");
  const std::size_t expected_output = checked_multiply(
      input.nseries, task.prepared_nsamples,
      "CUDA FFA prepare output element count overflow");
  if (output.count != expected_output) {
    throw std::invalid_argument(
        "CUDA FFA prepare output size must match nseries * prepared_nsamples");
  }
}

std::size_t validate_transform_arguments(
    CudaFfaInput input,
    CudaFfaBuffer scratch,
    CudaFfaBuffer output,
    const CudaLaunchOptions& options) {
  if (input.data == nullptr) {
    throw std::invalid_argument("CUDA FFA transform input data must not be null");
  }
  if (input.nseries == 0) {
    throw std::invalid_argument("CUDA FFA transform nseries must be > 0");
  }
  if (input.nsamples == 0) {
    throw std::invalid_argument(
        "CUDA FFA transform input nsamples must be > 0");
  }
  if (input.stride < input.nsamples) {
    throw std::invalid_argument(
        "CUDA FFA transform input stride must be >= nsamples");
  }
  if (input.shape.rows == 0) {
    throw std::invalid_argument("CUDA FFA transform rows must be > 0");
  }
  if (input.shape.bins <= 1) {
    throw std::invalid_argument("CUDA FFA transform bins must be > 1");
  }
  validate_launch_options(options);
  if (input.device_id != options.device_id ||
      scratch.device_id != options.device_id ||
      output.device_id != options.device_id) {
    throw std::invalid_argument(
        "CUDA FFA transform device ids must match launch device_id");
  }
  if (scratch.nseries != input.nseries || output.nseries != input.nseries) {
    throw std::invalid_argument(
        "CUDA FFA transform scratch/output nseries must match input nseries");
  }
  if (scratch.shape.rows != input.shape.rows ||
      scratch.shape.bins != input.shape.bins ||
      output.shape.rows != input.shape.rows ||
      output.shape.bins != input.shape.bins) {
    throw std::invalid_argument(
        "CUDA FFA transform scratch/output shapes must match input shape");
  }
  if (scratch.data == nullptr) {
    throw std::invalid_argument(
        "CUDA FFA transform scratch data must not be null");
  }
  if (output.data == nullptr) {
    throw std::invalid_argument(
        "CUDA FFA transform output data must not be null");
  }
  if (input.data == scratch.data || input.data == output.data ||
      scratch.data == output.data) {
    throw std::invalid_argument(
        "CUDA FFA transform input, scratch, and output must not alias");
  }

  const std::size_t series_elements =
      checked_multiply(input.shape.rows, input.shape.bins,
                       "CUDA FFA transform series element count overflow");
  if (series_elements > input.nsamples) {
    throw std::invalid_argument(
        "CUDA FFA transform rows * bins must be <= input nsamples");
  }
  if (scratch.stride < series_elements) {
    throw std::invalid_argument(
        "CUDA FFA transform scratch stride is too small");
  }
  if (output.stride < series_elements) {
    throw std::invalid_argument("CUDA FFA transform output stride is too small");
  }
  return series_elements;
}

void validate_transform_matches_task(CudaFfaInput input,
                                     CudaFfaBuffer scratch,
                                     CudaFfaBuffer output,
                                     const CudaFfaTaskLayout& task) {
  if (input.shape.rows != task.shape.rows || input.shape.bins != task.shape.bins ||
      scratch.shape.rows != task.shape.rows ||
      scratch.shape.bins != task.shape.bins ||
      output.shape.rows != task.shape.rows ||
      output.shape.bins != task.shape.bins) {
    throw std::invalid_argument(
        "CUDA FFA program transform shapes must match selected task");
  }
  if (input.nsamples < task.transform_elements) {
    throw std::invalid_argument(
        "CUDA FFA program transform input nsamples is too small");
  }
}

void launch_copy_level(const FfaCudaTransformLevel& level,
                       const FfaCudaDeviceTransformLevel& device_level,
                       CudaFfaInput input,
                       CudaFfaBuffer scratch,
                       CudaFfaBuffer output,
                       const CudaLaunchOptions& options) {
  if (level.copy_ops.empty()) {
    return;
  }

  const auto threads = static_cast<unsigned int>(options.threads_per_block);
  const std::size_t x_blocks =
      (device_level.max_copy_elements + threads - 1) / threads;
  const dim3 block(threads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA copy grid x overflow"),
      checked_grid_dim(level.copy_ops.size(), "CUDA FFA copy grid y overflow"),
      checked_grid_dim(input.nseries, "CUDA FFA copy grid z overflow"));
  ffa_copy_level_kernel<<<grid, block, 0, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins, device_level.copy_ops.data(),
      device_level.copy_op_count, device_level.max_copy_elements);
  check_cuda(cudaGetLastError(), "ffa_transform_block_cuda copy kernel launch");
}

void launch_merge_level(const FfaCudaTransformLevel& level,
                        const FfaCudaDeviceTransformLevel& device_level,
                        CudaFfaInput input,
                        CudaFfaBuffer scratch,
                        CudaFfaBuffer output,
                        const CudaLaunchOptions& options) {
  if (level.merge_ops.empty()) {
    return;
  }

  const auto threads = static_cast<unsigned int>(options.threads_per_block);
  const std::size_t x_blocks =
      (device_level.max_merge_elements + threads - 1) / threads;
  const dim3 block(threads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA merge grid x overflow"),
      checked_grid_dim(level.merge_ops.size(), "CUDA FFA merge grid y overflow"),
      checked_grid_dim(input.nseries, "CUDA FFA merge grid z overflow"));
  ffa_merge_level_kernel<<<grid, block, 0, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins, device_level.merge_ops.data(),
      device_level.merge_op_count, device_level.max_merge_elements);
  check_cuda(cudaGetLastError(), "ffa_transform_block_cuda merge kernel launch");
}

std::vector<FfaCudaDeviceTransformLevel> materialize_device_levels(
    const std::vector<FfaCudaTransformLevel>& levels,
    FfaTransformShape shape) {
  std::vector<FfaCudaDeviceTransformLevel> device_levels;
  device_levels.reserve(levels.size());
  for (const auto& level : levels) {
    FfaCudaDeviceTransformLevel device_level{
        .copy_ops = CudaDeviceBuffer<FfaCudaCopyOp>(level.copy_ops.size()),
        .copy_op_count = level.copy_ops.size(),
        .max_copy_elements = max_copy_elements(level, shape.bins),
        .merge_ops = CudaDeviceBuffer<FfaCudaMergeOp>(level.merge_ops.size()),
        .merge_op_count = level.merge_ops.size(),
        .max_merge_elements = max_merge_elements(level, shape.bins),
    };
    if (!level.copy_ops.empty()) {
      check_cuda(cudaMemcpy(device_level.copy_ops.data(), level.copy_ops.data(),
                            level.copy_ops.size() * sizeof(FfaCudaCopyOp),
                            cudaMemcpyHostToDevice),
                 "ffa_transform_block_cuda copy ops H2D");
    }
    if (!level.merge_ops.empty()) {
      check_cuda(cudaMemcpy(device_level.merge_ops.data(),
                            level.merge_ops.data(),
                            level.merge_ops.size() * sizeof(FfaCudaMergeOp),
                            cudaMemcpyHostToDevice),
                 "ffa_transform_block_cuda merge ops H2D");
    }
    device_levels.push_back(std::move(device_level));
  }
  return device_levels;
}

void launch_transform_levels(
    const std::vector<FfaCudaTransformLevel>& levels,
    const std::vector<FfaCudaDeviceTransformLevel>& device_levels,
    CudaFfaInput input,
    CudaFfaBuffer scratch,
    CudaFfaBuffer output,
    const CudaLaunchOptions& options) {
  if (levels.size() != device_levels.size()) {
    throw std::invalid_argument(
        "CUDA FFA transform level metadata mismatch");
  }
  for (std::size_t index = 0; index < levels.size(); ++index) {
    launch_copy_level(levels[index], device_levels[index], input, scratch,
                      output, options);
    launch_merge_level(levels[index], device_levels[index], input, scratch,
                       output, options);
  }
  if (options.synchronize_after_call) {
    check_cuda(cudaStreamSynchronize(options.stream),
               "ffa_transform_block_cuda synchronize");
  }
}

struct FfaCudaProgramLevel {
  std::size_t copy_offset = 0;
  std::size_t copy_count = 0;
  std::size_t max_copy_elements = 0;
  std::size_t merge_offset = 0;
  std::size_t merge_count = 0;
  std::size_t max_merge_elements = 0;
};

struct FfaCudaProgramTask {
  std::size_t level_begin = 0;
  std::size_t level_count = 0;
};

struct FfaCudaProgramGroup {
  std::vector<FfaCudaProgramTask> tasks;
};

struct FfaCudaProgramLayout {
  std::vector<FfaCudaProgramGroup> groups;
  std::vector<FfaCudaProgramLevel> levels;
};

// Build-only host state. The operation vectors are copied once to device memory
// and intentionally do not remain resident in CudaFfaProgramImpl.
struct FfaCudaProgramBuildState {
  FfaCudaProgramLayout layout;
  std::vector<FfaCudaCopyOp> copy_ops;
  std::vector<FfaCudaMergeOp> merge_ops;
};

struct FfaCudaProgramOps {
  CudaDeviceBuffer<FfaCudaCopyOp> copy_ops;
  CudaDeviceBuffer<FfaCudaMergeOp> merge_ops;
};

void append_program_task_layout(FfaTransformShape shape,
                                FfaCudaProgramBuildState& build_state,
                                FfaCudaProgramGroup& group) {
  FfaCudaProgramTask task{
      .level_begin = build_state.layout.levels.size(),
  };
  const std::vector<FfaCudaTransformLevel> levels =
      build_transform_levels(shape);
  for (const auto& level : levels) {
    const FfaCudaProgramLevel program_level{
        .copy_offset = build_state.copy_ops.size(),
        .copy_count = level.copy_ops.size(),
        .max_copy_elements = max_copy_elements(level, shape.bins),
        .merge_offset = build_state.merge_ops.size(),
        .merge_count = level.merge_ops.size(),
        .max_merge_elements = max_merge_elements(level, shape.bins),
    };
    build_state.copy_ops.insert(build_state.copy_ops.end(),
                                level.copy_ops.begin(), level.copy_ops.end());
    build_state.merge_ops.insert(build_state.merge_ops.end(),
                                 level.merge_ops.begin(), level.merge_ops.end());
    build_state.layout.levels.push_back(program_level);
  }
  task.level_count = build_state.layout.levels.size() - task.level_begin;
  group.tasks.push_back(task);
}

FfaCudaProgramOps materialize_program_ops(
    const FfaCudaProgramBuildState& build_state) {
  FfaCudaProgramOps ops{
      .copy_ops = CudaDeviceBuffer<FfaCudaCopyOp>(build_state.copy_ops.size()),
      .merge_ops =
          CudaDeviceBuffer<FfaCudaMergeOp>(build_state.merge_ops.size()),
  };
  if (!build_state.copy_ops.empty()) {
    check_cuda(cudaMemcpy(ops.copy_ops.data(), build_state.copy_ops.data(),
                          build_state.copy_ops.size() * sizeof(FfaCudaCopyOp),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program copy ops H2D");
  }
  if (!build_state.merge_ops.empty()) {
    check_cuda(cudaMemcpy(ops.merge_ops.data(), build_state.merge_ops.data(),
                          build_state.merge_ops.size() * sizeof(FfaCudaMergeOp),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program merge ops H2D");
  }
  return ops;
}

void launch_program_copy_level(const FfaCudaProgramLevel& level,
                               const FfaCudaProgramOps& ops,
                               CudaFfaInput input,
                               CudaFfaBuffer scratch,
                               CudaFfaBuffer output,
                               const CudaLaunchOptions& options) {
  if (level.copy_count == 0) {
    return;
  }
  const auto threads = static_cast<unsigned int>(options.threads_per_block);
  const std::size_t x_blocks =
      (level.max_copy_elements + threads - 1) / threads;
  const dim3 block(threads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA program copy grid x overflow"),
      checked_grid_dim(level.copy_count,
                       "CUDA FFA program copy grid y overflow"),
      checked_grid_dim(input.nseries,
                       "CUDA FFA program copy grid z overflow"));
  ffa_copy_level_kernel<<<grid, block, 0, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins,
      ops.copy_ops.data() + level.copy_offset, level.copy_count,
      level.max_copy_elements);
  check_cuda(cudaGetLastError(), "CUDA FFA program copy kernel launch");
}

void launch_program_merge_level(const FfaCudaProgramLevel& level,
                                const FfaCudaProgramOps& ops,
                                CudaFfaInput input,
                                CudaFfaBuffer scratch,
                                CudaFfaBuffer output,
                                const CudaLaunchOptions& options) {
  if (level.merge_count == 0) {
    return;
  }
  const auto threads = static_cast<unsigned int>(options.threads_per_block);
  const std::size_t x_blocks =
      (level.max_merge_elements + threads - 1) / threads;
  const dim3 block(threads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA program merge grid x overflow"),
      checked_grid_dim(level.merge_count,
                       "CUDA FFA program merge grid y overflow"),
      checked_grid_dim(input.nseries,
                       "CUDA FFA program merge grid z overflow"));
  ffa_merge_level_kernel<<<grid, block, 0, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins,
      ops.merge_ops.data() + level.merge_offset, level.merge_count,
      level.max_merge_elements);
  check_cuda(cudaGetLastError(), "CUDA FFA program merge kernel launch");
}

void launch_program_transform(const FfaCudaProgramTask& task,
                              const FfaCudaProgramLayout& layout,
                              const FfaCudaProgramOps& ops,
                              CudaFfaInput input,
                              CudaFfaBuffer scratch,
                              CudaFfaBuffer output,
                              const CudaLaunchOptions& options) {
  if (task.level_begin > layout.levels.size() ||
      task.level_count > layout.levels.size() - task.level_begin) {
    throw std::logic_error("CUDA FFA program level layout is invalid");
  }
  for (std::size_t index = 0; index < task.level_count; ++index) {
    const auto& level = layout.levels[task.level_begin + index];
    launch_program_copy_level(level, ops, input, scratch, output, options);
    launch_program_merge_level(level, ops, input, scratch, output, options);
  }
  if (options.synchronize_after_call) {
    check_cuda(cudaStreamSynchronize(options.stream),
               "CUDA FFA program transform synchronize");
  }
}

struct CudaFfaWorkspace {
  CudaFfaWorkspace(const CudaFfaExecutionPlan& plan,
                   const CudaFfaExecutionOptions& options,
                   int device_id)
      : shape(estimate_ffa_cuda_workspace(plan, options)) {
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
    prepared = CudaDeviceBuffer<float>(shape.prepared_bytes / sizeof(float));
    scratch = CudaDeviceBuffer<float>(shape.ping_bytes / sizeof(float));
    output = CudaDeviceBuffer<float>(shape.pong_bytes / sizeof(float));
  }

  CudaSpan<float> prepared_span(std::size_t nseries,
                                std::size_t prepared_nsamples,
                                int device_id) {
    const std::size_t count = checked_multiply(
        nseries, prepared_nsamples, "CUDA FFA prepared tile size overflow");
    if (nseries == 0 || nseries > shape.series_tile_size ||
        prepared_nsamples > shape.max_prepared_nsamples ||
        count > prepared.size()) {
      throw std::invalid_argument("CUDA FFA prepared tile exceeds workspace");
    }
    return CudaSpan<float>{
        .data = prepared.data(),
        .count = count,
        .device_id = device_id,
    };
  }

  CudaFfaInput prepared_input(std::size_t nseries,
                              const CudaFfaPrepareGroup& group,
                              const CudaFfaTaskLayout& task,
                              int device_id) {
    (void)prepared_span(nseries, group.prepared_nsamples, device_id);
    return CudaFfaInput{
        .data = prepared.data(),
        .nseries = nseries,
        .nsamples = group.prepared_nsamples,
        .stride = group.prepared_nsamples,
        .shape = task.shape,
        .device_id = device_id,
    };
  }

  CudaFfaBuffer transform_buffer(CudaDeviceBuffer<float>& buffer,
                                 std::size_t nseries,
                                 const CudaFfaTaskLayout& task,
                                 int device_id) {
    const std::size_t count = checked_multiply(
        nseries, task.transform_elements,
        "CUDA FFA transform tile size overflow");
    if (nseries == 0 || nseries > shape.series_tile_size ||
        task.transform_elements > shape.max_task_elements ||
        count > buffer.size()) {
      throw std::invalid_argument("CUDA FFA transform tile exceeds workspace");
    }
    return CudaFfaBuffer{
        .data = buffer.data(),
        .nseries = nseries,
        .stride = task.transform_elements,
        .shape = task.shape,
        .device_id = device_id,
    };
  }

  CudaFfaWorkspaceShape shape;
  CudaDeviceBuffer<float> prepared;
  CudaDeviceBuffer<float> scratch;
  CudaDeviceBuffer<float> output;
};

}  // namespace

struct CudaFfaProgramImpl {
  explicit CudaFfaProgramImpl(CudaFfaExecutionPlan execution_plan)
      : execution_plan(std::move(execution_plan)) {}

  int device_id = 0;
  CudaFfaExecutionPlan execution_plan;
  FfaCudaProgramLayout layout;
  FfaCudaProgramOps ops;
};

CudaFfaExecutionPlan::CudaFfaExecutionPlan(
    std::vector<CudaFfaPrepareGroup> groups,
    std::size_t max_prepared_nsamples,
    std::size_t max_transform_elements)
    : groups_(std::move(groups)),
      max_prepared_nsamples_(max_prepared_nsamples),
      max_transform_elements_(max_transform_elements) {}

std::span<const CudaFfaPrepareGroup> CudaFfaExecutionPlan::groups() const
    noexcept {
  return groups_;
}

std::size_t CudaFfaExecutionPlan::max_prepared_nsamples() const noexcept {
  return max_prepared_nsamples_;
}

std::size_t CudaFfaExecutionPlan::max_transform_elements() const noexcept {
  return max_transform_elements_;
}

CudaFfaProgram::CudaFfaProgram(CudaFfaExecutionPlan execution_plan,
                               const CudaFfaProgramOptions& options)
    : impl_(std::make_unique<CudaFfaProgramImpl>(std::move(execution_plan))) {
  validate_program_options(options);
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");
  impl_->device_id = options.device_id;
  FfaCudaProgramBuildState build_state;
  build_state.layout.groups.reserve(impl_->execution_plan.groups().size());

  for (const auto& group : impl_->execution_plan.groups()) {
    FfaCudaProgramGroup program_group;
    program_group.tasks.reserve(group.tasks.size());
    for (const auto& task : group.tasks) {
      append_program_task_layout(task.shape, build_state, program_group);
    }
    build_state.layout.groups.push_back(std::move(program_group));
  }
  impl_->ops = materialize_program_ops(build_state);
  impl_->layout = std::move(build_state.layout);
}

CudaFfaProgram::CudaFfaProgram(const FfaSearchPlan& plan,
                               const CudaFfaProgramOptions& options)
    : CudaFfaProgram(make_ffa_cuda_execution_plan(plan), options) {
}

CudaFfaProgram::~CudaFfaProgram() = default;

CudaFfaProgram::CudaFfaProgram(CudaFfaProgram&&) noexcept = default;

CudaFfaProgram& CudaFfaProgram::operator=(CudaFfaProgram&&) noexcept = default;

bool CudaFfaProgram::empty() const noexcept {
  return impl_ == nullptr;
}

const CudaFfaExecutionPlan& CudaFfaProgram::execution_plan() const {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  return impl_->execution_plan;
}

int CudaFfaProgram::device_id() const {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  return impl_->device_id;
}

std::size_t CudaFfaProgram::device_metadata_bytes() const {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  return checked_add(impl_->ops.copy_ops.bytes(), impl_->ops.merge_ops.bytes(),
                     "CUDA FFA program metadata byte size overflow");
}

CudaFfaExecutionPlan make_ffa_cuda_execution_plan(const FfaSearchPlan& plan) {
  validate_plan_for_workspace(plan);

  const std::size_t input_nsamples = plan.tasks.front().input_nsamples;
  std::vector<CudaFfaPrepareGroup> groups;
  std::size_t max_prepared_nsamples = 0;
  std::size_t max_transform_elements = 0;
  for (const auto& task : plan.tasks) {
    if (task.input_nsamples != input_nsamples) {
      throw std::invalid_argument(
          "CUDA FFA plan tasks must share input_nsamples");
    }
    const std::size_t transform_elements = checked_multiply(
        task.rows, task.bins, "CUDA FFA task layout size overflow");
    const CudaFfaTaskLayout task_layout{
        .task = task,
        .shape = FfaTransformShape{
            .rows = task.rows,
            .bins = task.bins,
        },
        .transform_elements = transform_elements,
    };

    auto group = std::find_if(
        groups.begin(), groups.end(),
        [&](const CudaFfaPrepareGroup& candidate) {
          return candidate.prepare_key.input_nsamples == task.input_nsamples &&
                 candidate.prepare_key.downsample_factor ==
                     task.downsample_factor;
        });
    if (group == groups.end()) {
      group = groups.insert(
          groups.end(),
          CudaFfaPrepareGroup{
              .prepare_key = CudaFfaPrepareKey{
                  .input_nsamples = task.input_nsamples,
                  .downsample_factor = task.downsample_factor,
              },
              .prepared_nsamples = task.prepared_nsamples,
          });
    }

    if (group->prepared_nsamples != task.prepared_nsamples) {
      throw std::logic_error(
          "CUDA FFA prepare group has inconsistent prepared_nsamples");
    }
    group->tasks.push_back(task_layout);
    max_prepared_nsamples =
        std::max(max_prepared_nsamples, task.prepared_nsamples);
    max_transform_elements =
        std::max(max_transform_elements, transform_elements);
  }

  return CudaFfaExecutionPlan(std::move(groups), max_prepared_nsamples,
                              max_transform_elements);
}

CudaFfaWorkspaceShape estimate_ffa_cuda_workspace(
    const CudaFfaExecutionPlan& plan,
    const CudaFfaExecutionOptions& options) {
  validate_execution_options(options);
  if (plan.groups().empty()) {
    throw std::invalid_argument(
        "CUDA FFA execution plan must contain at least one group");
  }
  if (plan.max_prepared_nsamples() == 0) {
    throw std::invalid_argument(
        "CUDA FFA execution plan max_prepared_nsamples must be > 0");
  }
  if (plan.max_transform_elements() == 0) {
    throw std::invalid_argument(
        "CUDA FFA execution plan max_task_elements must be > 0");
  }

  const std::size_t prepared_count = checked_multiply(
      options.series_tile_size, plan.max_prepared_nsamples(),
      "CUDA FFA prepared workspace element count overflow");
  const std::size_t task_count = checked_multiply(
      options.series_tile_size, plan.max_transform_elements(),
      "CUDA FFA transform workspace element count overflow");

  CudaFfaWorkspaceShape shape{
      .series_tile_size = options.series_tile_size,
      .max_prepared_nsamples = plan.max_prepared_nsamples(),
      .max_task_elements = plan.max_transform_elements(),
      .prepared_bytes = checked_float_bytes(
          prepared_count, "CUDA FFA prepared workspace byte size overflow"),
      .ping_bytes = checked_float_bytes(
          task_count, "CUDA FFA ping workspace byte size overflow"),
      .pong_bytes = checked_float_bytes(
          task_count, "CUDA FFA pong workspace byte size overflow"),
  };
  shape.total_bytes = checked_add(
      checked_add(shape.prepared_bytes, shape.ping_bytes,
                  "CUDA FFA workspace byte size overflow"),
      shape.pong_bytes, "CUDA FFA workspace byte size overflow");

  if (options.workspace_bytes_limit != 0 &&
      shape.total_bytes > options.workspace_bytes_limit) {
    throw std::runtime_error("CUDA FFA workspace exceeds workspace_bytes_limit");
  }

  return shape;
}

void prepare_ffa_input_cuda(CudaTimeSeriesBatchView input,
                            const FfaSearchTask& task,
                            CudaSpan<float> output,
                            const CudaLaunchOptions& options) {
  validate_prepare_arguments(input, task, output, options);
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");

  const auto threads = static_cast<unsigned int>(options.threads_per_block);
  const std::size_t x_blocks =
      (task.prepared_nsamples + threads - 1) / threads;
  const dim3 block(threads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA prepare grid x overflow"),
      checked_grid_dim(input.nseries, "CUDA FFA prepare grid y overflow"));

  if (task.downsample_factor == 1.0) {
    prepare_copy_float_kernel<<<grid, block, 0, options.stream>>>(
        input.data, input.nsamples, output.data, task.prepared_nsamples);
    check_cuda(cudaGetLastError(), "prepare_ffa_input_cuda kernel launch");
    if (options.synchronize_after_call) {
      check_cuda(cudaStreamSynchronize(options.stream),
                 "prepare_ffa_input_cuda synchronize");
    }
  } else {
    downsample_weighted_sum_cuda(input, task.downsample_factor, output,
                                 options);
    return;
  }
}

void ffa_transform_block_cuda(CudaFfaInput input,
                              CudaFfaBuffer scratch,
                              CudaFfaBuffer output,
                              const CudaLaunchOptions& options) {
  (void)validate_transform_arguments(input, scratch, output, options);
  if (!options.synchronize_after_call) {
    throw std::invalid_argument(
        "CUDA FFA materialized transform requires synchronize_after_call=true");
  }
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");

  const std::vector<FfaCudaTransformLevel> levels =
      build_transform_levels(input.shape);
  const std::vector<FfaCudaDeviceTransformLevel> device_levels =
      materialize_device_levels(levels, input.shape);
  launch_transform_levels(levels, device_levels, input, scratch, output,
                          options);
}

void ffa_transform_block_cuda(const CudaFfaProgram& program,
                              std::size_t group_index,
                              std::size_t task_index,
                              CudaFfaInput input,
                              CudaFfaBuffer scratch,
                              CudaFfaBuffer output,
                              const CudaLaunchOptions& options) {
  if (program.empty()) {
    throw std::invalid_argument("CUDA FFA program must not be empty");
  }
  validate_launch_options(options);
  if (program.device_id() != options.device_id) {
    throw std::invalid_argument(
        "CUDA FFA program device_id must match launch device_id");
  }
  if (group_index >= program.impl_->layout.groups.size()) {
    throw std::out_of_range("CUDA FFA program group_index is out of range");
  }
  const auto& group = program.impl_->layout.groups[group_index];
  const auto& plan_group = program.impl_->execution_plan.groups()[group_index];
  if (task_index >= group.tasks.size()) {
    throw std::out_of_range("CUDA FFA program task_index is out of range");
  }

  const auto& program_task = group.tasks[task_index];
  const auto& task_layout = plan_group.tasks[task_index];
  (void)validate_transform_arguments(input, scratch, output, options);
  validate_transform_matches_task(input, scratch, output, task_layout);
  if (program_task.level_count == 0) {
    throw std::logic_error("CUDA FFA program task has no transform levels");
  }

  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");
  launch_program_transform(program_task, program.impl_->layout,
                           program.impl_->ops, input, scratch, output,
                           options);
}

namespace {

// Owns only temporary storage for one execution-plan instance. A tile is a
// dense [series][sample] device batch, normally a DM tile. Transforms are
// intentionally not materialized beyond one task: later GPU detection will
// consume workspace.output on the same stream before that buffer is reused.
class [[maybe_unused]] CudaFfaExecutor {
 public:
  CudaFfaExecutor(const CudaFfaProgram& program,
                  CudaFfaExecutionOptions options)
      : program_(program), options_(std::move(options)),
        workspace_(program.execution_plan(), options_, program.device_id()) {
    if (program_.empty()) {
      throw std::invalid_argument("CUDA FFA executor program must not be empty");
    }
    validate_execution_options(options_);
    check_cuda(cudaSetDevice(program_.device_id()), "cudaSetDevice");
  }

  [[maybe_unused]] void run_tile(CudaTimeSeriesBatchView input) {
    validate_tile(input);
    const CudaLaunchOptions launch =
        options_.async_launch_options(program_.device_id());
    const auto groups = program_.execution_plan().groups();
    for (std::size_t group_index = 0; group_index < groups.size();
         ++group_index) {
      const auto& group = groups[group_index];
      const auto prepared = workspace_.prepared_span(
          input.nseries, group.prepared_nsamples, program_.device_id());
      prepare_ffa_input_cuda(input, group.tasks.front().task, prepared, launch);

      for (std::size_t task_index = 0; task_index < group.tasks.size();
           ++task_index) {
        const auto& task = group.tasks[task_index];
        const CudaFfaInput prepared_input = workspace_.prepared_input(
            input.nseries, group, task, program_.device_id());
        CudaFfaBuffer scratch = workspace_.transform_buffer(
            workspace_.scratch, input.nseries, task, program_.device_id());
        CudaFfaBuffer output = workspace_.transform_buffer(
            workspace_.output, input.nseries, task, program_.device_id());

        // One stream orders prepare -> transform and every successive task.
        // Future detection must enqueue its consumer kernel here, before the
        // next task overwrites output.
        ffa_transform_block_cuda(program_, group_index, task_index,
                                 prepared_input, scratch, output, launch);
      }
    }
    check_cuda(cudaStreamSynchronize(options_.stream),
               "CUDA FFA executor synchronize");
  }

 private:
  void validate_tile(CudaTimeSeriesBatchView input) const {
    if (input.data == nullptr) {
      throw std::invalid_argument("CUDA FFA executor input data must not be null");
    }
    if (input.nseries == 0 || input.nseries > options_.series_tile_size) {
      throw std::invalid_argument(
          "CUDA FFA executor input nseries must be within series_tile_size");
    }
    if (input.device_id != program_.device_id()) {
      throw std::invalid_argument(
          "CUDA FFA executor input device_id must match program device_id");
    }
    if (!(input.tsamp > 0.0) || !std::isfinite(input.tsamp)) {
      throw std::invalid_argument(
          "CUDA FFA executor input tsamp must be finite and > 0");
    }
    for (const auto& group : program_.execution_plan().groups()) {
      if (group.prepare_key.input_nsamples != input.nsamples) {
        throw std::invalid_argument(
            "CUDA FFA executor tile nsamples must match execution plan");
      }
    }
  }

  const CudaFfaProgram& program_;
  CudaFfaExecutionOptions options_;
  CudaFfaWorkspace workspace_;
};

}  // namespace

FfaSearchResult search_ffa_cuda(CudaTimeSeriesBatchView input,
                                const FfaSearchPlan& plan,
                                const FfaSearchOptions& search_options,
                                const CudaFfaExecutionOptions& cuda_options) {
  validate_search_input(input, plan, search_options, cuda_options);
  const CudaFfaExecutionPlan execution_plan =
      make_ffa_cuda_execution_plan(plan);

  check_cuda(cudaSetDevice(input.device_id), "cudaSetDevice");
  const CudaFfaWorkspace workspace(execution_plan, cuda_options,
                                   input.device_id);
  (void)workspace;

  throw std::runtime_error("CUDA FFA search is not implemented yet");
}

}  // namespace gaffa
