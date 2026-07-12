#include "gaffa/ffa_cuda.h"

#include "gaffa/cuda_memory.h"
#include "gaffa/time_series.h"
#include "gaffa/time_series_cuda.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

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

// One descriptor per bounded subtree, never one descriptor per output row.
// The schedule is pre-expanded on the host; the device kernel is iterative.
struct FfaCudaSubtreeOp {
  FfaCudaBufferRole input_role = FfaCudaBufferRole::Input;
  FfaCudaBufferRole output_role = FfaCudaBufferRole::Output;
  std::size_t input_begin_row = 0;
  std::size_t output_begin_row = 0;
  std::size_t rows = 0;
  std::size_t schedule_index = 0;
};

struct FfaCudaSubtreeLevel {
  std::size_t copy_offset = 0;
  std::size_t copy_count = 0;
  std::size_t merge_offset = 0;
  std::size_t merge_count = 0;
};

struct FfaCudaSubtreeSchedule {
  std::size_t rows = 0;
  std::size_t level_offset = 0;
  std::size_t level_count = 0;
};

struct FfaCudaTransformLevel {
  std::vector<FfaCudaSubtreeOp> subtree_ops;
  std::vector<FfaCudaCopyOp> copy_ops;
  std::vector<FfaCudaMergeOp> merge_ops;
};

struct FfaCudaDeviceTransformLevel {
  CudaDeviceBuffer<FfaCudaCopyOp> copy_ops;
  std::size_t copy_op_count = 0;
  std::size_t max_copy_elements = 0;
  CudaDeviceBuffer<FfaCudaMergeOp> merge_ops;
  std::size_t merge_op_count = 0;
  std::size_t max_merge_rows = 0;
};

// Device-only representation: every omitted FfaPeak field is immutable task
// metadata or can be reconstructed from these indices on the host.
struct FfaCudaPeak {
  std::uint32_t task_index = 0;
  std::uint32_t series_index = 0;
  std::uint32_t shift = 0;
  std::uint32_t phase = 0;
  std::uint32_t width_index = 0;
  float snr = 0.0F;
};

static_assert(sizeof(FfaCudaPeak) == 24);

struct FfaCudaBoxcarTrial {
  std::uint32_t width = 0;
  std::uint32_t width_index = 0;
  float height = 0.0F;
  float baseline = 0.0F;
};

__device__ void append_ffa_peak(FfaCudaPeak peak,
                                FfaCudaPeak* peaks,
                                std::size_t capacity,
                                unsigned int* count,
                                unsigned int* overflow) {
  const unsigned int index = atomicAdd(count, 1U);
  if (index < capacity) {
    peaks[index] = peak;
    return;
  }
  atomicExch(overflow, 1U);
}

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

constexpr unsigned int kFfaMergeThreads = 256;
constexpr unsigned int kFfaMergeWarpWidth = 32;
constexpr unsigned int kFfaMergeWarpsPerBlock =
    kFfaMergeThreads / kFfaMergeWarpWidth;
static_assert(kFfaMergeThreads % kFfaMergeWarpWidth == 0);

__global__ void ffa_merge_profiles_kernel(const float* input,
                                          float* scratch,
                                          float* output,
                                          std::size_t input_stride,
                                          std::size_t scratch_stride,
                                          std::size_t output_stride,
                                          std::size_t bins,
                                          const FfaCudaMergeOp* ops,
                                          std::size_t op_count,
                                          std::size_t max_output_rows) {
  const std::size_t op_index = blockIdx.y;
  const std::size_t series = blockIdx.z;
  if (op_index >= op_count) {
    return;
  }

  __shared__ FfaCudaMergeOp shared_op;
  if (threadIdx.x == 0) {
    shared_op = ops[op_index];
  }
  __syncthreads();
  const FfaCudaMergeOp op = shared_op;
  const unsigned int lane = threadIdx.x % kFfaMergeWarpWidth;
  const unsigned int warp = threadIdx.x / kFfaMergeWarpWidth;
  const std::size_t shift =
      blockIdx.x * static_cast<std::size_t>(kFfaMergeWarpsPerBlock) + warp;
  if (shift >= op.output_rows || shift >= max_output_rows) {
    return;
  }

  std::size_t head_shift = 0;
  std::size_t tail_shift = 0;
  std::size_t rolled = 0;
  if (lane == 0) {
    const float head_scale = static_cast<float>(op.head_rows - 1) /
                             static_cast<float>(op.output_rows - 1);
    const float tail_scale = static_cast<float>(op.tail_rows - 1) /
                             static_cast<float>(op.output_rows - 1);
    head_shift = static_cast<std::size_t>(
        head_scale * static_cast<float>(shift) + 0.5F);
    tail_shift = static_cast<std::size_t>(
        tail_scale * static_cast<float>(shift) + 0.5F);
    const std::size_t compensation = shift - (head_shift + tail_shift);
    rolled = (head_shift + compensation) % bins;
  }
  head_shift = __shfl_sync(0xffffffffU, head_shift, 0);
  tail_shift = __shfl_sync(0xffffffffU, tail_shift, 0);
  rolled = __shfl_sync(0xffffffffU, rolled, 0);

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

  for (std::size_t bin = lane; bin < bins; bin += kFfaMergeWarpWidth) {
    const std::size_t tail_bin =
        bin + rolled < bins ? bin + rolled : bin + rolled - bins;
    output_base[(op.output_begin_row + shift) * bins + bin] =
        head_base[(op.head_begin_row + head_shift) * bins + bin] +
        tail_base[(op.tail_begin_row + tail_shift) * bins + tail_bin];
  }
}

__device__ const float* select_shared_const_buffer(FfaCudaBufferRole role,
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

__device__ float* select_shared_mutable_buffer(FfaCudaBufferRole role,
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

__device__ void execute_shared_copy(const FfaCudaCopyOp& op,
                                    const float* input,
                                    float* scratch,
                                    float* output,
                                    std::size_t bins) {
  const float* source =
      select_shared_const_buffer(op.input_role, input, scratch, output);
  float* destination =
      select_shared_mutable_buffer(op.output_role, scratch, output);
  const std::size_t count = op.rows * bins;
  for (std::size_t index = threadIdx.x; index < count;
       index += blockDim.x) {
    const std::size_t row = index / bins;
    const std::size_t bin = index % bins;
    destination[(op.output_begin_row + row) * bins + bin] =
        source[(op.input_begin_row + row) * bins + bin];
  }
  __syncthreads();
}

__device__ void execute_shared_merge(const FfaCudaMergeOp& op,
                                     const float* input,
                                     float* scratch,
                                     float* output,
                                     std::size_t bins) {
  const float* head =
      select_shared_const_buffer(op.head_role, input, scratch, output);
  const float* tail =
      select_shared_const_buffer(op.tail_role, input, scratch, output);
  float* destination =
      select_shared_mutable_buffer(op.output_role, scratch, output);
  const float head_scale = static_cast<float>(op.head_rows - 1) /
                           static_cast<float>(op.output_rows - 1);
  const float tail_scale = static_cast<float>(op.tail_rows - 1) /
                           static_cast<float>(op.output_rows - 1);
  for (std::size_t shift = 0; shift < op.output_rows; ++shift) {
    const auto head_shift = static_cast<std::size_t>(
        head_scale * static_cast<float>(shift) + 0.5F);
    const auto tail_shift = static_cast<std::size_t>(
        tail_scale * static_cast<float>(shift) + 0.5F);
    const std::size_t compensation = shift - (head_shift + tail_shift);
    const std::size_t roll = (head_shift + compensation) % bins;
    for (std::size_t bin = threadIdx.x; bin < bins; bin += blockDim.x) {
      const std::size_t tail_bin =
          bin + roll < bins ? bin + roll : bin + roll - bins;
      destination[(op.output_begin_row + shift) * bins + bin] =
          head[(op.head_begin_row + head_shift) * bins + bin] +
          tail[(op.tail_begin_row + tail_shift) * bins + tail_bin];
    }
  }
  __syncthreads();
}

__global__ void ffa_subtree_transform_kernel(
    const float* input,
    float* scratch,
    float* output,
    std::size_t input_stride,
    std::size_t scratch_stride,
    std::size_t output_stride,
    std::size_t bins,
    const FfaCudaSubtreeOp* ops,
    std::size_t op_count,
    const FfaCudaSubtreeSchedule* schedules,
    const FfaCudaSubtreeLevel* levels,
    const FfaCudaCopyOp* copy_ops,
    const FfaCudaMergeOp* merge_ops,
    std::size_t subtree_rows_capacity) {
  const std::size_t op_index = blockIdx.x;
  const std::size_t series = blockIdx.y;
  if (op_index >= op_count) {
    return;
  }

  const FfaCudaSubtreeOp op = ops[op_index];
  const FfaCudaSubtreeSchedule schedule = schedules[op.schedule_index];
  if (op.rows == 0 || op.rows > subtree_rows_capacity ||
      schedule.rows != op.rows) {
    return;
  }
  const std::size_t input_stride_for_role =
      op.input_role == FfaCudaBufferRole::Input
          ? input_stride
          : (op.input_role == FfaCudaBufferRole::Scratch ? scratch_stride
                                                         : output_stride);
  const std::size_t output_stride_for_role =
      op.output_role == FfaCudaBufferRole::Scratch ? scratch_stride
                                                    : output_stride;
  const float* source =
      select_const_buffer(op.input_role, input, scratch, output) +
      series * input_stride_for_role + op.input_begin_row * bins;
  float* destination =
      select_mutable_buffer(op.output_role, scratch, output) +
      series * output_stride_for_role + op.output_begin_row * bins;

  extern __shared__ float shared[];
  const std::size_t subtree_elements = subtree_rows_capacity * bins;
  float* shared_input = shared;
  float* shared_scratch = shared_input + subtree_elements;
  float* shared_output = shared_scratch + subtree_elements;
  const std::size_t op_elements = op.rows * bins;
  for (std::size_t index = threadIdx.x; index < op_elements;
       index += blockDim.x) {
    shared_input[index] = source[index];
  }
  __syncthreads();

  for (std::size_t level_index = 0; level_index < schedule.level_count;
       ++level_index) {
    const FfaCudaSubtreeLevel level =
        levels[schedule.level_offset + level_index];
    for (std::size_t copy_index = 0; copy_index < level.copy_count;
         ++copy_index) {
      execute_shared_copy(copy_ops[level.copy_offset + copy_index],
                          shared_input, shared_scratch, shared_output, bins);
    }
    for (std::size_t merge_index = 0; merge_index < level.merge_count;
         ++merge_index) {
      execute_shared_merge(merge_ops[level.merge_offset + merge_index],
                           shared_input, shared_scratch, shared_output, bins);
    }
  }

  for (std::size_t index = threadIdx.x; index < op_elements;
       index += blockDim.x) {
    destination[index] = shared_output[index];
  }
}

constexpr unsigned int kFfaDetectionThreads = 256;
using FfaDetectionBlockScan = cub::BlockScan<float, kFfaDetectionThreads>;

__device__ void build_circular_prefix_parallel(
    const float* profile,
    float* prefix,
    std::size_t bins,
    std::size_t max_width,
    FfaDetectionBlockScan::TempStorage& scan_storage,
    float& scan_carry,
    float& tile_carry) {
  if (threadIdx.x == 0) {
    prefix[0] = 0.0F;
    scan_carry = 0.0F;
  }
  __syncthreads();

  for (std::size_t base = 0; base < bins;
       base += kFfaDetectionThreads) {
    const std::size_t index = base + threadIdx.x;
    const float value = index < bins ? profile[index] : 0.0F;
    float inclusive = 0.0F;
    FfaDetectionBlockScan(scan_storage).InclusiveSum(value, inclusive);
    if (threadIdx.x == kFfaDetectionThreads - 1) {
      tile_carry = inclusive;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const float tile_total = tile_carry;
      tile_carry = scan_carry;
      scan_carry += tile_total;
    }
    __syncthreads();
    if (index < bins) {
      prefix[index + 1] = tile_carry + inclusive;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scan_carry = prefix[bins];
  }
  __syncthreads();
  for (std::size_t base = 0; base < max_width;
       base += kFfaDetectionThreads) {
    const std::size_t index = base + threadIdx.x;
    const float value = index < max_width ? profile[index] : 0.0F;
    float inclusive = 0.0F;
    FfaDetectionBlockScan(scan_storage).InclusiveSum(value, inclusive);
    if (threadIdx.x == kFfaDetectionThreads - 1) {
      tile_carry = inclusive;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const float tile_total = tile_carry;
      tile_carry = scan_carry;
      scan_carry += tile_total;
    }
    __syncthreads();
    if (index < max_width) {
      prefix[bins + index + 1] = tile_carry + inclusive;
    }
    __syncthreads();
  }
}

__global__ void detect_ffa_rows_kernel(
    const float* transform,
    std::size_t transform_stride,
    std::size_t rows_eval,
    std::size_t bins,
    const FfaCudaBoxcarTrial* trials,
    std::size_t trial_count,
    std::size_t max_width,
    float stdnoise,
    float snr_threshold,
    std::uint32_t task_index,
    FfaCudaPeak* peaks,
    std::size_t peak_capacity,
    unsigned int* peak_count,
    unsigned int* peak_overflow) {
  const std::size_t shift = blockIdx.x;
  const std::size_t series = blockIdx.y;
  if (shift >= rows_eval) {
    return;
  }

  extern __shared__ float prefix[];

  const float* profile = transform + series * transform_stride + shift * bins;
  if (threadIdx.x == 0) {
    prefix[0] = 0.0F;
    for (std::size_t index = 0; index < bins; ++index) {
      prefix[index + 1] = prefix[index] + profile[index];
    }
    for (std::size_t index = 0; index < max_width; ++index) {
      prefix[bins + index + 1] = prefix[bins + index] + profile[index];
    }
  }
  __syncthreads();

  const float profile_sum = prefix[bins];
  constexpr unsigned int kWarpWidth = 32;
  constexpr unsigned int kWarpsPerBlock =
      kFfaDetectionThreads / kWarpWidth;
  static_assert(kFfaDetectionThreads % kWarpWidth == 0);

  // Each warp owns one width trial. A lane scans every 32nd phase, so this
  // covers any bins value; bins does not need to be warp- or block-aligned.
  const unsigned int lane = threadIdx.x % kWarpWidth;
  const unsigned int warp = threadIdx.x / kWarpWidth;
  for (std::size_t trial_base = 0; trial_base < trial_count;
       trial_base += kWarpsPerBlock) {
    const std::size_t trial_index = trial_base + warp;
    if (trial_index >= trial_count) {
      continue;
    }
    const FfaCudaBoxcarTrial trial = trials[trial_index];
    float best_sum = -INFINITY;
    std::uint32_t best_phase = 0;
    for (std::size_t phase = lane; phase < bins; phase += kWarpWidth) {
      const float sum = prefix[phase + trial.width] - prefix[phase];
      if (sum > best_sum || (sum == best_sum && phase < best_phase)) {
        best_sum = sum;
        best_phase = static_cast<std::uint32_t>(phase);
      }
    }
    for (unsigned int offset = kWarpWidth / 2; offset > 0; offset /= 2) {
      const float other_sum = __shfl_down_sync(0xffffffffU, best_sum, offset);
      const std::uint32_t other_phase =
          __shfl_down_sync(0xffffffffU, best_phase, offset);
      if (other_sum > best_sum ||
          (other_sum == best_sum && other_phase < best_phase)) {
        best_sum = other_sum;
        best_phase = other_phase;
      }
    }

    if (lane == 0) {
      const float snr =
          ((trial.height + trial.baseline) * best_sum -
           trial.baseline * profile_sum) /
          stdnoise;
      if (snr < snr_threshold) {
        continue;
      }
      append_ffa_peak(FfaCudaPeak{
          .task_index = task_index,
          .series_index = static_cast<std::uint32_t>(series),
          .shift = static_cast<std::uint32_t>(shift),
          .phase = best_phase,
          .width_index = trial.width_index,
          .snr = snr,
      }, peaks, peak_capacity, peak_count, peak_overflow);
    }
  }
}

__device__ __forceinline__ void detect_terminal_boxcar_trial(
    const float* prefix,
    std::size_t bins,
    FfaCudaBoxcarTrial trial,
    float profile_sum,
    float stdnoise,
    float snr_threshold,
    std::uint32_t task_index,
    std::size_t series,
    std::size_t shift,
    FfaCudaPeak* peaks,
    std::size_t peak_capacity,
    unsigned int* peak_count,
    unsigned int* peak_overflow) {
  constexpr unsigned int kWarpWidth = 32;
  const unsigned int lane = threadIdx.x % kWarpWidth;
  float best_sum = -INFINITY;
  std::uint32_t best_phase = 0;
  for (std::size_t phase = lane; phase < bins; phase += kWarpWidth) {
    const float sum = prefix[phase + trial.width] - prefix[phase];
    if (sum > best_sum || (sum == best_sum && phase < best_phase)) {
      best_sum = sum;
      best_phase = static_cast<std::uint32_t>(phase);
    }
  }
  for (unsigned int offset = kWarpWidth / 2; offset > 0; offset /= 2) {
    const float other_sum = __shfl_down_sync(0xffffffffU, best_sum, offset);
    const std::uint32_t other_phase =
        __shfl_down_sync(0xffffffffU, best_phase, offset);
    if (other_sum > best_sum ||
        (other_sum == best_sum && other_phase < best_phase)) {
      best_sum = other_sum;
      best_phase = other_phase;
    }
  }
  if (lane == 0) {
    const float snr =
        ((trial.height + trial.baseline) * best_sum -
         trial.baseline * profile_sum) /
        stdnoise;
    if (snr >= snr_threshold) {
      append_ffa_peak(FfaCudaPeak{
          .task_index = task_index,
          .series_index = static_cast<std::uint32_t>(series),
          .shift = static_cast<std::uint32_t>(shift),
          .phase = best_phase,
          .width_index = trial.width_index,
          .snr = snr,
      }, peaks, peak_capacity, peak_count, peak_overflow);
    }
  }
}

// Production-only terminal consumer. It replaces the final materialized
// merge plus the immediately following detection pass when the final FFA
// level is one merge operation.
template <int FixedTrialCount>
__global__ void detect_ffa_terminal_merge_kernel(
    const float* input,
    const float* scratch,
    const float* output,
    std::size_t input_stride,
    std::size_t scratch_stride,
    std::size_t output_stride,
    std::size_t rows_eval,
    std::size_t bins,
    const FfaCudaMergeOp* terminal_op,
    const FfaCudaBoxcarTrial* trials,
    std::size_t trial_count,
    std::size_t max_width,
    float stdnoise,
    float snr_threshold,
    std::uint32_t task_index,
    FfaCudaPeak* peaks,
    std::size_t peak_capacity,
    unsigned int* peak_count,
    unsigned int* peak_overflow) {
  const std::size_t shift = blockIdx.x;
  const std::size_t series = blockIdx.y;
  if (shift >= rows_eval) {
    return;
  }

  __shared__ FfaCudaMergeOp op;
  __shared__ std::size_t head_shift;
  __shared__ std::size_t tail_shift;
  __shared__ std::size_t rolled;
  __shared__ FfaDetectionBlockScan::TempStorage scan_storage;
  __shared__ float scan_carry;
  __shared__ float tile_carry;
  if (threadIdx.x == 0) {
    op = *terminal_op;
  }
  __syncthreads();

  extern __shared__ float shared[];
  float* profile = shared;
  float* prefix = profile + bins;
  const float* head_base =
      select_const_buffer(op.head_role, input, scratch, output) +
      series * (op.head_role == FfaCudaBufferRole::Input
                    ? input_stride
                    : (op.head_role == FfaCudaBufferRole::Scratch
                           ? scratch_stride
                           : output_stride));
  const float* tail_base =
      select_const_buffer(op.tail_role, input, scratch, output) +
      series * (op.tail_role == FfaCudaBufferRole::Input
                    ? input_stride
                    : (op.tail_role == FfaCudaBufferRole::Scratch
                           ? scratch_stride
                           : output_stride));
  if (threadIdx.x == 0) {
    const float head_scale = static_cast<float>(op.head_rows - 1) /
                             static_cast<float>(op.output_rows - 1);
    const float tail_scale = static_cast<float>(op.tail_rows - 1) /
                             static_cast<float>(op.output_rows - 1);
    head_shift = static_cast<std::size_t>(
        head_scale * static_cast<float>(shift) + 0.5F);
    tail_shift = static_cast<std::size_t>(
        tail_scale * static_cast<float>(shift) + 0.5F);
    const std::size_t compensation = shift - (head_shift + tail_shift);
    rolled = (head_shift + compensation) % bins;
  }
  __syncthreads();
  for (std::size_t bin = threadIdx.x; bin < bins; bin += blockDim.x) {
    const std::size_t tail_bin =
        bin + rolled < bins ? bin + rolled : bin + rolled - bins;
    profile[bin] =
        head_base[(op.head_begin_row + head_shift) * bins + bin] +
        tail_base[(op.tail_begin_row + tail_shift) * bins + tail_bin];
  }
  __syncthreads();

  build_circular_prefix_parallel(profile, prefix, bins, max_width,
                                 scan_storage, scan_carry, tile_carry);

  const float profile_sum = prefix[bins];
  constexpr unsigned int kWarpsPerBlock = kFfaDetectionThreads / 32;
  const unsigned int warp = threadIdx.x / 32;
  if constexpr (FixedTrialCount > 0) {
    static_assert(FixedTrialCount <= kWarpsPerBlock);
    if (warp < FixedTrialCount) {
      detect_terminal_boxcar_trial(
          prefix, bins, trials[warp], profile_sum, stdnoise, snr_threshold,
          task_index, series, shift, peaks, peak_capacity, peak_count,
          peak_overflow);
    }
  } else {
    for (std::size_t trial_base = 0; trial_base < trial_count;
         trial_base += kWarpsPerBlock) {
      const std::size_t trial_index = trial_base + warp;
      if (trial_index < trial_count) {
        detect_terminal_boxcar_trial(
            prefix, bins, trials[trial_index], profile_sum, stdnoise,
            snr_threshold, task_index, series, shift, peaks, peak_capacity,
            peak_count, peak_overflow);
      }
    }
  }
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

constexpr std::size_t kPreferredSharedSubtreeRows = 16;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::size_t subtree_shared_bytes(std::size_t rows, std::size_t bins) {
  return checked_float_bytes(
      checked_multiply(
          checked_multiply(rows, bins,
                           "CUDA FFA subtree shared element count overflow"),
          3, "CUDA FFA subtree shared element count overflow"),
      "CUDA FFA subtree shared byte size overflow");
}

std::size_t select_shared_subtree_rows(std::size_t bins, int device_id) {
  int optin_shared_bytes = 0;
  check_cuda(cudaDeviceGetAttribute(&optin_shared_bytes,
                                    cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    device_id),
             "cudaDeviceGetAttribute max opt-in shared memory");
  if (optin_shared_bytes <= 0) {
    return 0;
  }

  const std::size_t rows_by_capacity =
      static_cast<std::size_t>(optin_shared_bytes) /
      (3 * bins * sizeof(float));
  if (rows_by_capacity < 2) {
    return 0;
  }
  return std::min(kPreferredSharedSubtreeRows, rows_by_capacity);
}

void configure_subtree_dynamic_shared_memory(std::size_t shared_bytes) {
  if (shared_bytes == 0) {
    return;
  }
  if (shared_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("CUDA FFA subtree shared memory size overflow");
  }
  check_cuda(cudaFuncSetAttribute(
                 ffa_subtree_transform_kernel,
                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                 static_cast<int>(shared_bytes)),
             "cudaFuncSetAttribute CUDA FFA subtree shared memory");
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
  if (options.peak_buffer_bytes == 0) {
    throw std::invalid_argument("CUDA FFA peak_buffer_bytes must be > 0");
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
  output.subtree_ops.insert(output.subtree_ops.end(), input.subtree_ops.begin(),
                            input.subtree_ops.end());
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
    std::size_t rows,
    std::size_t subtree_rows) {
  if (subtree_rows != 0 && rows <= subtree_rows) {
    FfaCudaTransformLevel level;
    level.subtree_ops.push_back(FfaCudaSubtreeOp{
        .input_role = input_role,
        .output_role = output_role,
        .input_begin_row = row_begin,
        .output_begin_row = row_begin,
        .rows = rows,
    });
    return {level};
  }
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
      input_role, output_role, scratch_role, row_begin, head_rows,
      subtree_rows);
  auto tail_levels = build_transform_levels(
      input_role, output_role, scratch_role, row_begin + head_rows, tail_rows,
      subtree_rows);

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
    FfaTransformShape shape,
    std::size_t subtree_rows = 0) {
  return build_transform_levels(FfaCudaBufferRole::Input,
                                FfaCudaBufferRole::Scratch,
                                FfaCudaBufferRole::Output, 0, shape.rows,
                                subtree_rows);
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

std::size_t max_subtree_rows(const FfaCudaTransformLevel& level) {
  std::size_t max_rows = 0;
  for (const auto& op : level.subtree_ops) {
    max_rows = std::max(max_rows, op.rows);
  }
  return max_rows;
}

std::size_t max_merge_rows(const FfaCudaTransformLevel& level) {
  std::size_t max_rows = 0;
  for (const auto& op : level.merge_ops) {
    max_rows = std::max(max_rows, op.output_rows);
  }
  return max_rows;
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

void validate_search_options(const FfaSearchOptions& search_options) {
  if (!std::isfinite(search_options.snr_threshold)) {
    throw std::invalid_argument("CUDA FFA S/N threshold must be finite");
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

  const std::size_t x_blocks =
      (device_level.max_merge_rows + kFfaMergeWarpsPerBlock - 1) /
      kFfaMergeWarpsPerBlock;
  const dim3 block(kFfaMergeThreads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA merge grid x overflow"),
      checked_grid_dim(level.merge_ops.size(), "CUDA FFA merge grid y overflow"),
      checked_grid_dim(input.nseries, "CUDA FFA merge grid z overflow"));
  ffa_merge_profiles_kernel<<<grid, block, 0, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins, device_level.merge_ops.data(),
      device_level.merge_op_count, device_level.max_merge_rows);
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
        .max_merge_rows = max_merge_rows(level),
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
  std::size_t subtree_offset = 0;
  std::size_t subtree_count = 0;
  std::size_t max_subtree_rows = 0;
  std::size_t copy_offset = 0;
  std::size_t copy_count = 0;
  std::size_t max_copy_elements = 0;
  std::size_t merge_offset = 0;
  std::size_t merge_count = 0;
  std::size_t max_merge_rows = 0;
};

struct FfaCudaProgramTask {
  std::size_t level_begin = 0;
  std::size_t level_count = 0;
  std::size_t detection_trial_offset = 0;
  std::size_t detection_trial_count = 0;
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
  std::vector<FfaCudaSubtreeOp> subtree_ops;
  std::vector<FfaCudaSubtreeSchedule> subtree_schedules;
  std::vector<FfaCudaSubtreeLevel> subtree_levels;
  std::vector<FfaCudaCopyOp> subtree_copy_ops;
  std::vector<FfaCudaMergeOp> subtree_merge_ops;
  std::vector<FfaCudaCopyOp> copy_ops;
  std::vector<FfaCudaMergeOp> merge_ops;
  std::vector<FfaCudaBoxcarTrial> detection_trials;
  std::size_t max_subtree_shared_bytes = 0;
};

struct FfaCudaProgramOps {
  CudaDeviceBuffer<FfaCudaSubtreeOp> subtree_ops;
  CudaDeviceBuffer<FfaCudaSubtreeSchedule> subtree_schedules;
  CudaDeviceBuffer<FfaCudaSubtreeLevel> subtree_levels;
  CudaDeviceBuffer<FfaCudaCopyOp> subtree_copy_ops;
  CudaDeviceBuffer<FfaCudaMergeOp> subtree_merge_ops;
  CudaDeviceBuffer<FfaCudaCopyOp> copy_ops;
  CudaDeviceBuffer<FfaCudaMergeOp> merge_ops;
  CudaDeviceBuffer<FfaCudaBoxcarTrial> detection_trials;
};

std::size_t find_or_append_subtree_schedule(
    std::size_t rows,
    FfaCudaProgramBuildState& build_state) {
  for (std::size_t index = 0; index < build_state.subtree_schedules.size();
       ++index) {
    if (build_state.subtree_schedules[index].rows == rows) {
      return index;
    }
  }

  const auto schedule_levels = build_transform_levels(
      FfaCudaBufferRole::Input, FfaCudaBufferRole::Scratch,
      FfaCudaBufferRole::Output, 0, rows, 0);
  const FfaCudaSubtreeSchedule schedule{
      .rows = rows,
      .level_offset = build_state.subtree_levels.size(),
      .level_count = schedule_levels.size(),
  };
  for (const auto& level : schedule_levels) {
    build_state.subtree_levels.push_back(FfaCudaSubtreeLevel{
        .copy_offset = build_state.subtree_copy_ops.size(),
        .copy_count = level.copy_ops.size(),
        .merge_offset = build_state.subtree_merge_ops.size(),
        .merge_count = level.merge_ops.size(),
    });
    build_state.subtree_copy_ops.insert(build_state.subtree_copy_ops.end(),
                                        level.copy_ops.begin(),
                                        level.copy_ops.end());
    build_state.subtree_merge_ops.insert(build_state.subtree_merge_ops.end(),
                                         level.merge_ops.begin(),
                                         level.merge_ops.end());
  }
  build_state.subtree_schedules.push_back(schedule);
  return build_state.subtree_schedules.size() - 1;
}

void append_program_task_layout(const CudaFfaTaskLayout& task_layout,
                                FfaCudaProgramBuildState& build_state,
                                FfaCudaProgramGroup& group,
                                int device_id) {
  FfaCudaProgramTask task{
      .level_begin = build_state.layout.levels.size(),
      .detection_trial_offset = build_state.detection_trials.size(),
  };
  const FfaTransformShape shape = task_layout.shape;
  const std::vector<FfaCudaTransformLevel> levels =
      build_transform_levels(shape,
                             select_shared_subtree_rows(shape.bins, device_id));
  for (const auto& level : levels) {
    const FfaCudaProgramLevel program_level{
        .subtree_offset = build_state.subtree_ops.size(),
        .subtree_count = level.subtree_ops.size(),
        .max_subtree_rows = max_subtree_rows(level),
        .copy_offset = build_state.copy_ops.size(),
        .copy_count = level.copy_ops.size(),
        .max_copy_elements = max_copy_elements(level, shape.bins),
        .merge_offset = build_state.merge_ops.size(),
        .merge_count = level.merge_ops.size(),
        .max_merge_rows = max_merge_rows(level),
    };
    for (auto subtree : level.subtree_ops) {
      subtree.schedule_index =
          find_or_append_subtree_schedule(subtree.rows, build_state);
      build_state.max_subtree_shared_bytes = std::max(
          build_state.max_subtree_shared_bytes,
          subtree_shared_bytes(subtree.rows, shape.bins));
      build_state.subtree_ops.push_back(subtree);
    }
    build_state.copy_ops.insert(build_state.copy_ops.end(),
                                level.copy_ops.begin(), level.copy_ops.end());
    build_state.merge_ops.insert(build_state.merge_ops.end(),
                                 level.merge_ops.begin(), level.merge_ops.end());
    build_state.layout.levels.push_back(program_level);
  }
  task.level_count = build_state.layout.levels.size() - task.level_begin;
  for (const FfaBoxcarTrial& trial : task_layout.detection_plan.boxcar_trials) {
    if (trial.width > std::numeric_limits<std::uint32_t>::max() ||
        trial.width_index > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA FFA detection trial index overflow");
    }
    build_state.detection_trials.push_back(FfaCudaBoxcarTrial{
        .width = static_cast<std::uint32_t>(trial.width),
        .width_index = static_cast<std::uint32_t>(trial.width_index),
        .height = trial.height,
        .baseline = trial.baseline,
    });
  }
  task.detection_trial_count = build_state.detection_trials.size() -
                               task.detection_trial_offset;
  group.tasks.push_back(task);
}

FfaCudaProgramOps materialize_program_ops(
    const FfaCudaProgramBuildState& build_state) {
  FfaCudaProgramOps ops{
      .subtree_ops = CudaDeviceBuffer<FfaCudaSubtreeOp>(
          build_state.subtree_ops.size()),
      .subtree_schedules = CudaDeviceBuffer<FfaCudaSubtreeSchedule>(
          build_state.subtree_schedules.size()),
      .subtree_levels = CudaDeviceBuffer<FfaCudaSubtreeLevel>(
          build_state.subtree_levels.size()),
      .subtree_copy_ops = CudaDeviceBuffer<FfaCudaCopyOp>(
          build_state.subtree_copy_ops.size()),
      .subtree_merge_ops = CudaDeviceBuffer<FfaCudaMergeOp>(
          build_state.subtree_merge_ops.size()),
      .copy_ops = CudaDeviceBuffer<FfaCudaCopyOp>(build_state.copy_ops.size()),
      .merge_ops =
          CudaDeviceBuffer<FfaCudaMergeOp>(build_state.merge_ops.size()),
      .detection_trials = CudaDeviceBuffer<FfaCudaBoxcarTrial>(
          build_state.detection_trials.size()),
  };
  if (!build_state.subtree_ops.empty()) {
    check_cuda(cudaMemcpy(ops.subtree_ops.data(),
                          build_state.subtree_ops.data(),
                          build_state.subtree_ops.size() *
                              sizeof(FfaCudaSubtreeOp),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program subtree ops H2D");
  }
  if (!build_state.subtree_schedules.empty()) {
    check_cuda(cudaMemcpy(ops.subtree_schedules.data(),
                          build_state.subtree_schedules.data(),
                          build_state.subtree_schedules.size() *
                              sizeof(FfaCudaSubtreeSchedule),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program subtree schedules H2D");
  }
  if (!build_state.subtree_levels.empty()) {
    check_cuda(cudaMemcpy(ops.subtree_levels.data(),
                          build_state.subtree_levels.data(),
                          build_state.subtree_levels.size() *
                              sizeof(FfaCudaSubtreeLevel),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program subtree levels H2D");
  }
  if (!build_state.subtree_copy_ops.empty()) {
    check_cuda(cudaMemcpy(ops.subtree_copy_ops.data(),
                          build_state.subtree_copy_ops.data(),
                          build_state.subtree_copy_ops.size() *
                              sizeof(FfaCudaCopyOp),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program subtree copy ops H2D");
  }
  if (!build_state.subtree_merge_ops.empty()) {
    check_cuda(cudaMemcpy(ops.subtree_merge_ops.data(),
                          build_state.subtree_merge_ops.data(),
                          build_state.subtree_merge_ops.size() *
                              sizeof(FfaCudaMergeOp),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program subtree merge ops H2D");
  }
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
  if (!build_state.detection_trials.empty()) {
    check_cuda(cudaMemcpy(ops.detection_trials.data(),
                          build_state.detection_trials.data(),
                          build_state.detection_trials.size() *
                              sizeof(FfaCudaBoxcarTrial),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program detection trials H2D");
  }
  return ops;
}

void launch_program_subtree_level(const FfaCudaProgramLevel& level,
                                  const FfaCudaProgramOps& ops,
                                  CudaFfaInput input,
                                  CudaFfaBuffer scratch,
                                  CudaFfaBuffer output,
                                  const CudaLaunchOptions& options) {
  if (level.subtree_count == 0) {
    return;
  }
  const std::size_t shared_elements = checked_multiply(
      checked_multiply(level.max_subtree_rows, input.shape.bins,
                       "CUDA FFA subtree shared element count overflow"),
      3, "CUDA FFA subtree shared element count overflow");
  const std::size_t shared_bytes = checked_float_bytes(
      shared_elements, "CUDA FFA subtree shared byte size overflow");
  const dim3 block(static_cast<unsigned int>(options.threads_per_block));
  const dim3 grid(
      checked_grid_dim(level.subtree_count,
                       "CUDA FFA program subtree grid x overflow"),
      checked_grid_dim(input.nseries,
                       "CUDA FFA program subtree grid y overflow"));
  ffa_subtree_transform_kernel<<<grid, block, shared_bytes, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins,
      ops.subtree_ops.data() + level.subtree_offset, level.subtree_count,
      ops.subtree_schedules.data(), ops.subtree_levels.data(),
      ops.subtree_copy_ops.data(), ops.subtree_merge_ops.data(),
      level.max_subtree_rows);
  check_cuda(cudaGetLastError(), "CUDA FFA program subtree kernel launch");
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
  const std::size_t x_blocks =
      (level.max_merge_rows + kFfaMergeWarpsPerBlock - 1) /
      kFfaMergeWarpsPerBlock;
  const dim3 block(kFfaMergeThreads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA program merge grid x overflow"),
      checked_grid_dim(level.merge_count,
                       "CUDA FFA program merge grid y overflow"),
      checked_grid_dim(input.nseries,
                       "CUDA FFA program merge grid z overflow"));
  ffa_merge_profiles_kernel<<<grid, block, 0, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins,
      ops.merge_ops.data() + level.merge_offset, level.merge_count,
      level.max_merge_rows);
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
    launch_program_subtree_level(level, ops, input, scratch, output, options);
    launch_program_copy_level(level, ops, input, scratch, output, options);
    launch_program_merge_level(level, ops, input, scratch, output, options);
  }
  if (options.synchronize_after_call) {
    check_cuda(cudaStreamSynchronize(options.stream),
               "CUDA FFA program transform synchronize");
  }
}

bool can_fuse_terminal_merge(const FfaCudaProgramTask& task,
                             const FfaCudaProgramLayout& layout) {
  if (task.level_count == 0 || task.level_begin >= layout.levels.size() ||
      task.level_count > layout.levels.size() - task.level_begin) {
    return false;
  }
  const FfaCudaProgramLevel& terminal =
      layout.levels[task.level_begin + task.level_count - 1];
  return terminal.subtree_count == 0 && terminal.copy_count == 0 &&
         terminal.merge_count == 1;
}

void launch_program_transform_before_terminal(
    const FfaCudaProgramTask& task,
    const FfaCudaProgramLayout& layout,
    const FfaCudaProgramOps& ops,
    CudaFfaInput input,
    CudaFfaBuffer scratch,
    CudaFfaBuffer output,
    const CudaLaunchOptions& options) {
  if (!can_fuse_terminal_merge(task, layout)) {
    throw std::logic_error("CUDA FFA task has no fusable terminal merge");
  }
  for (std::size_t index = 0; index + 1 < task.level_count; ++index) {
    const auto& level = layout.levels[task.level_begin + index];
    launch_program_subtree_level(level, ops, input, scratch, output, options);
    launch_program_copy_level(level, ops, input, scratch, output, options);
    launch_program_merge_level(level, ops, input, scratch, output, options);
  }
}

struct CudaFfaWorkspace {
  CudaFfaWorkspace(const CudaFfaExecutionPlan& plan,
                   const CudaFfaExecutionOptions& options,
                   int device_id)
      : shape(estimate_ffa_cuda_workspace(plan, options)) {
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
    prepared = CudaDeviceBuffer<float>(shape.prepared_bytes / sizeof(float));
    scratch = CudaDeviceBuffer<float>(shape.scratch_bytes / sizeof(float));
    output = CudaDeviceBuffer<float>(shape.output_bytes / sizeof(float));
    detection_compact = CudaDeviceBuffer<FfaCudaPeak>(
        shape.detection_compact_bytes / sizeof(FfaCudaPeak));
    detection_peak_count = CudaDeviceBuffer<unsigned int>(1);
    detection_peak_overflow = CudaDeviceBuffer<unsigned int>(1);
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
  CudaDeviceBuffer<FfaCudaPeak> detection_compact;
  CudaDeviceBuffer<unsigned int> detection_peak_count;
  CudaDeviceBuffer<unsigned int> detection_peak_overflow;
};

}  // namespace

struct CudaFfaProgramImpl {
  explicit CudaFfaProgramImpl(CudaFfaExecutionPlan execution_plan,
                              CudaFfaExecutionOptions execution_options)
      : execution_plan(std::move(execution_plan)),
        execution_options(std::move(execution_options)) {}

  int device_id = 0;
  CudaFfaExecutionPlan execution_plan;
  CudaFfaExecutionOptions execution_options;
  FfaCudaProgramLayout layout;
  FfaCudaProgramOps ops;
  std::unique_ptr<CudaFfaWorkspace> workspace;
};

CudaFfaExecutionPlan::CudaFfaExecutionPlan(
    std::vector<CudaFfaPrepareGroup> groups,
    std::size_t max_prepared_nsamples,
    std::size_t max_transform_elements,
    std::size_t max_detection_slots_per_series)
    : groups_(std::move(groups)),
      max_prepared_nsamples_(max_prepared_nsamples),
      max_transform_elements_(max_transform_elements),
      max_detection_slots_per_series_(max_detection_slots_per_series) {}

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

std::size_t CudaFfaExecutionPlan::max_detection_slots_per_series() const
    noexcept {
  return max_detection_slots_per_series_;
}

CudaFfaProgram::CudaFfaProgram(
    CudaFfaExecutionPlan execution_plan,
    const CudaFfaProgramOptions& program_options,
    const CudaFfaExecutionOptions& execution_options)
    : impl_(std::make_unique<CudaFfaProgramImpl>(
          std::move(execution_plan), execution_options)) {
  validate_program_options(program_options);
  validate_execution_options(execution_options);
  check_cuda(cudaSetDevice(program_options.device_id), "cudaSetDevice");
  impl_->device_id = program_options.device_id;
  FfaCudaProgramBuildState build_state;
  build_state.layout.groups.reserve(impl_->execution_plan.groups().size());

  for (const auto& group : impl_->execution_plan.groups()) {
    FfaCudaProgramGroup program_group;
    program_group.tasks.reserve(group.tasks.size());
    for (const auto& task : group.tasks) {
      append_program_task_layout(task, build_state, program_group,
                                 impl_->device_id);
    }
    build_state.layout.groups.push_back(std::move(program_group));
  }
  configure_subtree_dynamic_shared_memory(
      build_state.max_subtree_shared_bytes);
  impl_->ops = materialize_program_ops(build_state);
  impl_->layout = std::move(build_state.layout);
  impl_->workspace = std::make_unique<CudaFfaWorkspace>(
      impl_->execution_plan, impl_->execution_options, impl_->device_id);
}

CudaFfaProgram::CudaFfaProgram(
    const FfaSearchPlan& plan,
    const CudaFfaProgramOptions& program_options,
    const CudaFfaExecutionOptions& execution_options)
    : CudaFfaProgram(make_ffa_cuda_execution_plan(plan), program_options,
                     execution_options) {
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

std::size_t CudaFfaProgram::tile_capacity() const {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  return impl_->execution_options.series_tile_size;
}

const CudaFfaWorkspaceShape& CudaFfaProgram::workspace_shape() const {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  return impl_->workspace->shape;
}

std::size_t CudaFfaProgram::device_metadata_bytes() const {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  return checked_add(
      checked_add(
          checked_add(impl_->ops.subtree_ops.bytes(),
                      impl_->ops.subtree_schedules.bytes(),
                      "CUDA FFA program metadata byte size overflow"),
          checked_add(
              checked_add(impl_->ops.subtree_levels.bytes(),
                          impl_->ops.subtree_copy_ops.bytes(),
                          "CUDA FFA program metadata byte size overflow"),
              impl_->ops.subtree_merge_ops.bytes(),
              "CUDA FFA program metadata byte size overflow"),
          "CUDA FFA program metadata byte size overflow"),
      checked_add(
          checked_add(impl_->ops.copy_ops.bytes(), impl_->ops.merge_ops.bytes(),
                      "CUDA FFA program metadata byte size overflow"),
          impl_->ops.detection_trials.bytes(),
          "CUDA FFA program metadata byte size overflow"),
      "CUDA FFA program metadata byte size overflow");
}

CudaFfaExecutionPlan make_ffa_cuda_execution_plan(const FfaSearchPlan& plan) {
  validate_plan_for_workspace(plan);

  const std::size_t input_nsamples = plan.tasks.front().input_nsamples;
  std::vector<CudaFfaPrepareGroup> groups;
  std::size_t max_prepared_nsamples = 0;
  std::size_t max_transform_elements = 0;
  std::size_t max_detection_slots_per_series = 0;
  for (const auto& task : plan.tasks) {
    if (task.input_nsamples != input_nsamples) {
      throw std::invalid_argument(
          "CUDA FFA plan tasks must share input_nsamples");
    }
    const std::size_t transform_elements = checked_multiply(
        task.rows, task.bins, "CUDA FFA task layout size overflow");
    const FfaDetectionPlan detection_plan =
        make_ffa_detection_plan(plan.width_trials, task.bins);
    const std::size_t detection_slots_per_series = checked_multiply(
        task.rows_eval, detection_plan.boxcar_trials.size(),
        "CUDA FFA detection slot count overflow");
    const CudaFfaTaskLayout task_layout{
        .task = task,
        .shape = FfaTransformShape{
            .rows = task.rows,
            .bins = task.bins,
        },
        .detection_plan = detection_plan,
        .transform_elements = transform_elements,
        .detection_slots_per_series = detection_slots_per_series,
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
    max_detection_slots_per_series = std::max(
        max_detection_slots_per_series, detection_slots_per_series);
  }

  return CudaFfaExecutionPlan(std::move(groups), max_prepared_nsamples,
                              max_transform_elements,
                              max_detection_slots_per_series);
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
  if (plan.max_detection_slots_per_series() == 0) {
    throw std::invalid_argument(
        "CUDA FFA execution plan max_detection_slots_per_series must be > 0");
  }

  const std::size_t prepared_count = checked_multiply(
      options.series_tile_size, plan.max_prepared_nsamples(),
      "CUDA FFA prepared workspace element count overflow");
  const std::size_t task_count = checked_multiply(
      options.series_tile_size, plan.max_transform_elements(),
      "CUDA FFA transform workspace element count overflow");
  const std::size_t detection_count = checked_multiply(
      options.series_tile_size, plan.max_detection_slots_per_series(),
      "CUDA FFA detection workspace element count overflow");
  const std::size_t full_peak_bytes = checked_multiply(
      detection_count, sizeof(FfaCudaPeak),
      "CUDA FFA detection peak workspace byte size overflow");

  CudaFfaWorkspaceShape shape{
      .series_tile_size = options.series_tile_size,
      .max_prepared_nsamples = plan.max_prepared_nsamples(),
      .max_task_elements = plan.max_transform_elements(),
      .max_detection_slots_per_series =
          plan.max_detection_slots_per_series(),
      .prepared_bytes = checked_float_bytes(
          prepared_count, "CUDA FFA prepared workspace byte size overflow"),
      .scratch_bytes = checked_float_bytes(
          task_count, "CUDA FFA scratch workspace byte size overflow"),
      .output_bytes = checked_float_bytes(
          task_count, "CUDA FFA output workspace byte size overflow"),
      .detection_raw_bytes = 0,
      .detection_compact_bytes =
          std::min(full_peak_bytes, options.peak_buffer_bytes),
      .detection_flags_bytes = 0,
      .detection_select_temp_bytes = 0,
  };
  shape.total_bytes = checked_add(
      checked_add(
          checked_add(shape.prepared_bytes, shape.scratch_bytes,
                      "CUDA FFA workspace byte size overflow"),
          shape.output_bytes, "CUDA FFA workspace byte size overflow"),
      checked_add(
          checked_add(shape.detection_raw_bytes, shape.detection_compact_bytes,
                      "CUDA FFA workspace byte size overflow"),
          checked_add(shape.detection_flags_bytes,
                      shape.detection_select_temp_bytes,
                      "CUDA FFA workspace byte size overflow"),
          "CUDA FFA workspace byte size overflow"),
      "CUDA FFA workspace byte size overflow");

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

namespace detail {

// Executes one dense [series][sample] tile using workspace owned by a
// long-lived CudaFfaProgram. It intentionally owns no CUDA allocations.
class CudaFfaTileRunner {
 private:
  static CudaFfaWorkspace& workspace_for(CudaFfaProgram& program) {
    if (program.empty() || program.impl_->workspace == nullptr) {
      throw std::invalid_argument("CUDA FFA program must not be empty");
    }
    return *program.impl_->workspace;
  }

 public:
  CudaFfaTileRunner(CudaFfaProgram& program, FfaSearchOptions search_options)
      : program_(program), search_options_(search_options),
        workspace_(workspace_for(program)) {
    validate_search_options(search_options_);
    check_cuda(cudaSetDevice(program_.device_id()), "cudaSetDevice");
  }

  FfaBatchSearchResult run_tile(CudaTimeSeriesBatchView input) {
    validate_tile(input);
    FfaBatchSearchResult result;
    std::vector<std::size_t> peak_counts(input.nseries, 0);
    const CudaLaunchOptions launch =
        program_.impl_->execution_options.async_launch_options(
            program_.device_id());
    const auto groups = program_.execution_plan().groups();
    for (std::size_t group_index = 0; group_index < groups.size();
         ++group_index) {
      const auto& group = groups[group_index];
      reset_group_peak_buffer();
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
        const auto& program_task =
            program_.impl_->layout.groups[group_index].tasks[task_index];

        // One stream orders prepare -> transform and every successive task.
        // A terminal merge can feed detection directly, avoiding one full
        // materialized transform write/read pair.
        const bool fuse_terminal =
            can_fuse_terminal_merge(program_task, program_.impl_->layout);
        if (fuse_terminal) {
          launch_program_transform_before_terminal(
              program_task, program_.impl_->layout, program_.impl_->ops,
              prepared_input, scratch, output, launch);
        } else {
          ffa_transform_block_cuda(program_, group_index, task_index,
                                   prepared_input, scratch, output, launch);
        }
        enqueue_task_detection(group_index, task_index, input.nseries,
                               prepared_input, scratch, output, fuse_terminal);
      }
      collect_group_peaks(group_index, input.nseries, peak_counts, result);
    }
    std::sort(result.peaks.begin(), result.peaks.end(),
              [](const FfaBatchPeak& lhs, const FfaBatchPeak& rhs) {
                if (lhs.series_index != rhs.series_index) {
                  return lhs.series_index < rhs.series_index;
                }
                return is_better_ffa_peak(lhs.peak, rhs.peak);
              });
    return result;
  }

 private:
  void reset_group_peak_buffer() {
    check_cuda(cudaMemsetAsync(workspace_.detection_peak_count.data(), 0,
                               sizeof(unsigned int),
                               program_.impl_->execution_options.stream),
               "CUDA FFA peak count reset");
    check_cuda(cudaMemsetAsync(workspace_.detection_peak_overflow.data(), 0,
                               sizeof(unsigned int),
                               program_.impl_->execution_options.stream),
               "CUDA FFA peak overflow reset");
  }

  void enqueue_task_detection(std::size_t group_index,
                              std::size_t task_index,
                              std::size_t nseries,
                              CudaFfaInput input,
                              CudaFfaBuffer scratch,
                              CudaFfaBuffer transform,
                              bool fuse_terminal) {
    const auto& program_group = program_.impl_->layout.groups[group_index];
    const auto& program_task = program_group.tasks[task_index];
    const auto& task =
        program_.impl_->execution_plan.groups()[group_index].tasks[task_index];
    if (task_index > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA FFA task index overflow");
    }
    if (program_task.detection_trial_count !=
        task.detection_plan.boxcar_trials.size()) {
      throw std::logic_error("CUDA FFA detection metadata mismatch");
    }
    const std::size_t profile_and_prefix_elements =
        fuse_terminal
            ? checked_add(
                  task.shape.bins,
                  checked_add(task.shape.bins,
                              task.detection_plan.max_width + 1,
                              "CUDA FFA terminal detection shared size overflow"),
                  "CUDA FFA terminal detection shared size overflow")
            : task.shape.bins + task.detection_plan.max_width + 1;
    const std::size_t shared_bytes = checked_float_bytes(
        profile_and_prefix_elements,
        "CUDA FFA detection shared-memory size overflow");
    const dim3 grid(
        checked_grid_dim(task.task.rows_eval,
                         "CUDA FFA detection grid x overflow"),
        checked_grid_dim(nseries, "CUDA FFA detection grid y overflow"));
    if (fuse_terminal) {
      const FfaCudaProgramLevel& terminal = program_.impl_->layout.levels[
          program_task.level_begin + program_task.level_count - 1];
      const auto launch_terminal = [&]<int TrialCount>() {
        detect_ffa_terminal_merge_kernel<TrialCount><<<
            grid, kFfaDetectionThreads, shared_bytes,
            program_.impl_->execution_options.stream>>>(
            input.data, scratch.data, transform.data, input.stride,
            scratch.stride, transform.stride, task.task.rows_eval,
            task.shape.bins,
            program_.impl_->ops.merge_ops.data() + terminal.merge_offset,
            program_.impl_->ops.detection_trials.data() +
                program_task.detection_trial_offset,
            program_task.detection_trial_count, task.detection_plan.max_width,
            ffa_task_stdnoise(task.task), search_options_.snr_threshold,
            static_cast<std::uint32_t>(task_index),
            workspace_.detection_compact.data(),
            workspace_.detection_compact.size(),
            workspace_.detection_peak_count.data(),
            workspace_.detection_peak_overflow.data());
      };
      switch (program_task.detection_trial_count) {
        case 1:
          launch_terminal.template operator()<1>();
          break;
        case 2:
          launch_terminal.template operator()<2>();
          break;
        case 3:
          launch_terminal.template operator()<3>();
          break;
        case 4:
          launch_terminal.template operator()<4>();
          break;
        case 5:
          launch_terminal.template operator()<5>();
          break;
        case 6:
          launch_terminal.template operator()<6>();
          break;
        case 7:
          launch_terminal.template operator()<7>();
          break;
        case 8:
          launch_terminal.template operator()<8>();
          break;
        default:
          launch_terminal.template operator()<0>();
          break;
      }
    } else {
      detect_ffa_rows_kernel<<<grid, kFfaDetectionThreads, shared_bytes,
                               program_.impl_->execution_options.stream>>>(
          transform.data, transform.stride, task.task.rows_eval, task.shape.bins,
          program_.impl_->ops.detection_trials.data() +
              program_task.detection_trial_offset,
          program_task.detection_trial_count, task.detection_plan.max_width,
          ffa_task_stdnoise(task.task), search_options_.snr_threshold,
          static_cast<std::uint32_t>(task_index),
          workspace_.detection_compact.data(),
          workspace_.detection_compact.size(),
          workspace_.detection_peak_count.data(),
          workspace_.detection_peak_overflow.data());
    }
    check_cuda(cudaGetLastError(), "CUDA FFA detection kernel launch");
  }

  void collect_group_peaks(std::size_t group_index,
                           std::size_t nseries,
                           std::vector<std::size_t>& peak_counts,
                           FfaBatchSearchResult& result) {
    check_cuda(cudaStreamSynchronize(program_.impl_->execution_options.stream),
               "CUDA FFA group synchronize");

    unsigned int selected_count = 0;
    unsigned int overflow = 0;
    check_cuda(cudaMemcpy(&selected_count, workspace_.detection_peak_count.data(),
                          sizeof(selected_count), cudaMemcpyDeviceToHost),
               "CUDA FFA peak count D2H");
    check_cuda(cudaMemcpy(&overflow, workspace_.detection_peak_overflow.data(),
                          sizeof(overflow), cudaMemcpyDeviceToHost),
               "CUDA FFA peak overflow D2H");
    if (overflow != 0 || selected_count > workspace_.detection_compact.size()) {
      throw std::runtime_error(
          "CUDA FFA compact peak buffer overflow; increase "
          "CudaFfaExecutionOptions::peak_buffer_bytes or search with a "
          "higher SNR threshold");
    }
    std::vector<FfaCudaPeak> compact_peaks(
        selected_count);
    if (!compact_peaks.empty()) {
      check_cuda(cudaMemcpy(compact_peaks.data(),
                            workspace_.detection_compact.data(),
                            compact_peaks.size() * sizeof(FfaCudaPeak),
                            cudaMemcpyDeviceToHost),
                 "CUDA FFA detection compact peaks D2H");
    }
    for (const FfaCudaPeak& compact_peak : compact_peaks) {
      const auto& group = program_.impl_->execution_plan.groups()[group_index];
      if (compact_peak.task_index >= group.tasks.size()) {
        throw std::logic_error("CUDA FFA peak task index is invalid");
      }
      const auto& task = group.tasks[compact_peak.task_index];
      if (compact_peak.series_index >= nseries ||
          compact_peak.shift >= task.task.rows_eval ||
          compact_peak.width_index >= task.detection_plan.boxcar_trials.size() ||
          !std::isfinite(compact_peak.snr)) {
        throw std::logic_error("CUDA FFA detection peak metadata is invalid");
      }
      std::size_t& count = peak_counts[compact_peak.series_index];
      if (search_options_.max_peaks != 0 && count >= search_options_.max_peaks) {
        throw std::runtime_error(
            "CUDA FFA detection peak count exceeded max_peaks safety guard");
      }
      const FfaBoxcarTrial& trial = task.detection_plan.boxcar_trials[
          compact_peak.width_index];
      result.peaks.push_back(FfaBatchPeak{
          .series_index = compact_peak.series_index,
          .peak = make_ffa_peak(task.task, trial, compact_peak.shift,
                                compact_peak.phase, compact_peak.snr),
      });
      ++count;
    }
  }

  void validate_tile(CudaTimeSeriesBatchView input) const {
    if (input.data == nullptr) {
      throw std::invalid_argument("CUDA FFA executor input data must not be null");
    }
    if (input.nseries == 0 || input.nseries > program_.tile_capacity()) {
      throw std::invalid_argument(
          "CUDA FFA tile nseries must be within program tile_capacity");
    }
    if (input.device_id != program_.device_id()) {
      throw std::invalid_argument(
          "CUDA FFA executor input device_id must match program device_id");
    }
    for (const auto& group : program_.execution_plan().groups()) {
      if (group.prepare_key.input_nsamples != input.nsamples) {
        throw std::invalid_argument(
            "CUDA FFA executor tile nsamples must match execution plan");
      }
    }
  }

  CudaFfaProgram& program_;
  FfaSearchOptions search_options_;
  CudaFfaWorkspace& workspace_;
};

}  // namespace detail

FfaBatchSearchResult run_ffa_batch_cuda(
    CudaFfaProgram& program,
    CudaTimeSeriesBatchView batch,
    const FfaSearchOptions& options) {
  detail::CudaFfaTileRunner runner(program, options);
  return runner.run_tile(batch);
}

FfaSearchResult search_ffa_cuda(CudaFfaProgram& program,
                                 CudaSpan<const float> time_series,
                                 const FfaSearchOptions& options) {
  if (time_series.data == nullptr || time_series.count == 0) {
    throw std::invalid_argument("CUDA FFA time series must not be empty");
  }
  if (time_series.device_id != program.device_id()) {
    throw std::invalid_argument(
        "CUDA FFA time series device_id must match program device_id");
  }
  const FfaBatchSearchResult batch = run_ffa_batch_cuda(
      program,
      CudaTimeSeriesBatchView{
          .data = time_series.data,
          .nseries = 1,
          .nsamples = time_series.count,
          .device_id = time_series.device_id,
      },
      options);
  FfaSearchResult result;
  result.peaks.reserve(batch.peaks.size());
  for (const FfaBatchPeak& peak : batch.peaks) {
    if (peak.series_index != 0) {
      throw std::logic_error("CUDA FFA single-series result has invalid index");
    }
    result.peaks.push_back(peak.peak);
  }
  return result;
}

FfaSearchResult search_ffa_cuda(
    CudaSpan<const float> time_series,
    const FfaSearchPlan& plan,
    const FfaSearchOptions& options,
    const CudaFfaProgramOptions& program_options,
    const CudaFfaExecutionOptions& execution_options) {
  if (time_series.data == nullptr || time_series.count == 0) {
    throw std::invalid_argument("CUDA FFA time series must not be empty");
  }
  if (time_series.device_id != program_options.device_id) {
    throw std::invalid_argument(
        "CUDA FFA time series device_id must match program device_id");
  }
  CudaFfaProgram program(plan, program_options, execution_options);
  return search_ffa_cuda(program, time_series, options);
}

}  // namespace gaffa
