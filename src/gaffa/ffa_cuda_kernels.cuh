// Private kernel data model and device kernels. This fragment is included once
// by ffa_cuda.cu so CUDA code generation remains in one translation unit.
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
  __shared__ FfaDetectionBlockScan::TempStorage scan_storage;
  __shared__ float scan_carry;
  __shared__ float tile_carry;

  const float* profile = transform + series * transform_stride + shift * bins;
  build_circular_prefix_parallel(profile, prefix, bins, max_width,
                                 scan_storage, scan_carry, tile_carry);

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
