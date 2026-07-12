#include "gaffa/preprocessing_cuda.h"

#include "gaffa/cuda_memory.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gaffa {
namespace {

constexpr std::size_t kThreadsPerBlock = 256;
constexpr std::size_t kWarpSize = 32;
constexpr std::size_t kScrunchWarpsPerBlock =
    kThreadsPerBlock / kWarpSize;
constexpr std::size_t kStatsItemsPerThread = 8;
constexpr std::size_t kStatsItemsPerBlock =
    kThreadsPerBlock * kStatsItemsPerThread;
constexpr std::size_t kMedianOutputsPerBlock = 16;

enum class CudaPreprocessStatus : int {
  Ok = 0,
  NonFiniteInput = 1,
  ConstantInput = 2,
};

struct CudaPartialStats {
  double sum = 0.0;
  double sum_squares = 0.0;
};

struct CudaSeriesStats {
  double mean = 0.0;
  double inverse_stddev = 0.0;
};

struct CudaDetrendLayout {
  DetrendRunningMedianOptions options{};
  std::size_t scrunch_factor = 1;
  std::size_t median_window = 0;
};

struct CudaPreprocessStepLayout {
  PreprocessStepKind kind = PreprocessStepKind::Normalise;
  CudaDetrendLayout detrend{};
  NormaliseOptions normalise{};
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::size_t checked_add(std::size_t lhs,
                        std::size_t rhs,
                        const char* message) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::overflow_error(message);
  }
  return lhs + rhs;
}

std::size_t checked_multiply(std::size_t lhs,
                             std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

std::size_t checked_bytes(std::size_t count,
                          std::size_t element_size,
                          const char* message) {
  return checked_multiply(count, element_size, message);
}

unsigned int checked_grid_dim(std::size_t value, const char* message) {
  if (value == 0 ||
      value > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::overflow_error(message);
  }
  return static_cast<unsigned int>(value);
}

std::size_t ceil_div(std::size_t value, std::size_t divisor) {
  return value / divisor + (value % divisor != 0 ? 1 : 0);
}

void validate_detrend_options(DetrendRunningMedianOptions options) {
  if (options.window_samples == 0 || options.window_samples % 2 == 0) {
    throw std::invalid_argument(
        "CUDA running median window_samples must be odd and > 0");
  }
  if (options.min_points == 0 || options.min_points % 2 == 0) {
    throw std::invalid_argument(
        "CUDA running median min_points must be odd and > 0");
  }
}

CudaDetrendLayout make_detrend_layout(DetrendRunningMedianOptions options) {
  validate_detrend_options(options);
  const std::size_t factor = std::max<std::size_t>(
      1, options.window_samples / options.min_points);
  const std::size_t median_window =
      factor == 1 ? options.window_samples : options.min_points;
  if (median_window > 256) {
    throw std::invalid_argument(
        "CUDA running median effective window must be <= 256");
  }
  return CudaDetrendLayout{
      .options = options,
      .scrunch_factor = factor,
      .median_window = median_window,
  };
}

void validate_program_options(const CudaPreprocessProgramOptions& options) {
  if (options.device_id < 0) {
    throw std::invalid_argument(
        "CUDA preprocessing program device_id must be >= 0");
  }
}

void validate_stream_for_current_device(cudaStream_t stream) {
  if (stream == nullptr) {
    return;
  }

  CUcontext current_context = nullptr;
  CUcontext stream_context = nullptr;
  const CUresult current_status = cuCtxGetCurrent(&current_context);
  const CUresult stream_status = cuStreamGetCtx(stream, &stream_context);
  if (current_status != CUDA_SUCCESS || stream_status != CUDA_SUCCESS ||
      current_context == nullptr || stream_context != current_context) {
    throw std::invalid_argument(
        "CUDA preprocessing stream must belong to the program device");
  }
}

void validate_execution_options(const CudaPreprocessExecutionOptions& options) {
  if (options.series_tile_size == 0) {
    throw std::invalid_argument(
        "CUDA preprocessing series_tile_size must be > 0");
  }
  if (options.max_nsamples == 0) {
    throw std::invalid_argument(
        "CUDA preprocessing max_nsamples must be > 0");
  }
  if (options.threads_per_block != kThreadsPerBlock) {
    throw std::invalid_argument(
        "CUDA preprocessing threads_per_block must be 256");
  }
}

struct CudaPreprocessBuild {
  std::vector<CudaPreprocessStepLayout> steps;
  CudaPreprocessWorkspaceShape workspace_shape;
  std::size_t scrunched_count = 0;
  std::size_t baseline_count = 0;
  std::size_t partial_count = 0;
  std::size_t series_stats_count = 0;
};

CudaPreprocessBuild compile_preprocess_layout(
    const PreprocessPlan& plan,
    const CudaPreprocessExecutionOptions& execution_options) {
  validate_execution_options(execution_options);
  if (plan.steps.empty()) {
    throw std::invalid_argument("CUDA preprocess plan must contain at least one step");
  }

  std::size_t max_scrunched_per_series = 0;
  std::size_t max_baseline_per_series = 0;
  bool has_normalise = false;
  CudaPreprocessBuild build;
  build.steps.reserve(plan.steps.size());
  for (const PreprocessStep& step : plan.steps) {
    CudaPreprocessStepLayout layout{
        .kind = step.kind,
        .normalise = step.normalise,
    };
    switch (step.kind) {
      case PreprocessStepKind::DetrendRunningMedian: {
        layout.detrend = make_detrend_layout(step.detrend_running_median);
        const std::size_t reduced = execution_options.max_nsamples /
                                    layout.detrend.scrunch_factor;
        if (layout.detrend.scrunch_factor == 1) {
          max_baseline_per_series = std::max(
              max_baseline_per_series, execution_options.max_nsamples);
        } else {
          max_scrunched_per_series =
              std::max(max_scrunched_per_series, reduced);
          max_baseline_per_series =
              std::max(max_baseline_per_series, reduced);
        }
        break;
      }
      case PreprocessStepKind::Normalise:
        has_normalise = true;
        break;
    }
    build.steps.push_back(layout);
  }

  const std::size_t partials_per_series =
      has_normalise
          ? ceil_div(execution_options.max_nsamples, kStatsItemsPerBlock)
          : 0;
  build.scrunched_count = checked_multiply(
      execution_options.series_tile_size, max_scrunched_per_series,
      "CUDA preprocessing scrunched workspace size overflow");
  build.baseline_count = checked_multiply(
      execution_options.series_tile_size, max_baseline_per_series,
      "CUDA preprocessing baseline workspace size overflow");
  build.partial_count = checked_multiply(
      execution_options.series_tile_size, partials_per_series,
      "CUDA preprocessing stats workspace size overflow");
  build.series_stats_count =
      has_normalise ? execution_options.series_tile_size : 0;

  build.workspace_shape = CudaPreprocessWorkspaceShape{
      .series_tile_size = execution_options.series_tile_size,
      .max_nsamples = execution_options.max_nsamples,
      .scrunched_bytes = checked_bytes(
          build.scrunched_count, sizeof(float),
          "CUDA preprocessing scrunched workspace byte size overflow"),
      .baseline_bytes = checked_bytes(
          build.baseline_count, sizeof(float),
          "CUDA preprocessing baseline workspace byte size overflow"),
      .partial_stats_bytes = checked_bytes(
          build.partial_count, sizeof(CudaPartialStats),
          "CUDA preprocessing partial stats byte size overflow"),
      .series_stats_bytes = checked_bytes(
          build.series_stats_count, sizeof(CudaSeriesStats),
          "CUDA preprocessing series stats byte size overflow"),
      .status_bytes = sizeof(int),
  };
  build.workspace_shape.total_bytes = checked_add(
      checked_add(build.workspace_shape.scrunched_bytes,
                  build.workspace_shape.baseline_bytes,
                  "CUDA preprocessing workspace byte size overflow"),
      checked_add(
          checked_add(build.workspace_shape.partial_stats_bytes,
                      build.workspace_shape.series_stats_bytes,
                      "CUDA preprocessing workspace byte size overflow"),
          build.workspace_shape.status_bytes,
          "CUDA preprocessing workspace byte size overflow"),
      "CUDA preprocessing workspace byte size overflow");
  if (execution_options.workspace_bytes_limit != 0 &&
      build.workspace_shape.total_bytes >
          execution_options.workspace_bytes_limit) {
    throw std::runtime_error(
        "CUDA preprocessing workspace exceeds workspace_bytes_limit");
  }
  return build;
}

__device__ void set_status_once(int* status, CudaPreprocessStatus value) {
  atomicCAS(status, static_cast<int>(CudaPreprocessStatus::Ok),
            static_cast<int>(value));
}

__global__ void validate_finite_kernel(const float* input,
                                       std::size_t count,
                                       int* status) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < count && !isfinite(input[index])) {
    set_status_once(status, CudaPreprocessStatus::NonFiniteInput);
  }
}

__global__ void mean_scrunch_kernel(const float* input,
                                    std::size_t input_nsamples,
                                    float* output,
                                    std::size_t output_nsamples,
                                    std::size_t factor) {
  const std::size_t warp = threadIdx.x / kWarpSize;
  const std::size_t lane = threadIdx.x % kWarpSize;
  const std::size_t output_index =
      blockIdx.x * kScrunchWarpsPerBlock + warp;
  const std::size_t series = blockIdx.y;
  if (output_index >= output_nsamples) {
    return;
  }

  const float* source = input + series * input_nsamples + output_index * factor;
  double sum = 0.0;
  for (std::size_t index = lane; index < factor; index += kWarpSize) {
    sum += static_cast<double>(source[index]);
  }
  for (unsigned int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  if (lane == 0) {
    output[series * output_nsamples + output_index] =
        static_cast<float>(sum / static_cast<double>(factor));
  }
}

template <unsigned int SortSize>
__global__ void running_median_kernel(const float* input,
                                      std::size_t input_nsamples,
                                      float* baseline,
                                      std::size_t output_nsamples,
                                      std::size_t window_samples) {
  static_assert(kThreadsPerBlock % SortSize == 0);
  constexpr unsigned int kGroups = kThreadsPerBlock / SortSize;
  __shared__ float values[kThreadsPerBlock];

  const unsigned int group = threadIdx.x / SortSize;
  const unsigned int lane = threadIdx.x % SortSize;
  const std::size_t series = blockIdx.y;
  const std::size_t half_window = window_samples / 2;
  const float* series_input = input + series * input_nsamples;
  float* series_baseline = baseline + series * output_nsamples;

  for (std::size_t output_offset = 0;
       output_offset < kMedianOutputsPerBlock;
       output_offset += kGroups) {
    const std::size_t output_index =
        blockIdx.x * kMedianOutputsPerBlock + output_offset + group;
    const std::size_t shared_index = group * SortSize + lane;
    float value = INFINITY;
    if (output_index < output_nsamples && lane < window_samples) {
      const long long centered = static_cast<long long>(output_index) +
                                 static_cast<long long>(lane) -
                                 static_cast<long long>(half_window);
      const std::size_t nonnegative_index =
          centered <= 0 ? 0 : static_cast<std::size_t>(centered);
      const std::size_t source_index =
          nonnegative_index >= input_nsamples ? input_nsamples - 1
                                               : nonnegative_index;
      value = series_input[source_index];
    }
    values[shared_index] = value;
    __syncthreads();

    for (unsigned int length = 2; length <= SortSize; length <<= 1) {
      for (unsigned int stride = length >> 1; stride > 0; stride >>= 1) {
        const unsigned int other_lane = lane ^ stride;
        if (lane < other_lane) {
          const std::size_t other_index = group * SortSize + other_lane;
          const float self = values[shared_index];
          const float other = values[other_index];
          const bool ascending = (lane & length) == 0;
          values[shared_index] = ascending ? fminf(self, other)
                                           : fmaxf(self, other);
          values[other_index] = ascending ? fmaxf(self, other)
                                          : fminf(self, other);
        }
        __syncthreads();
      }
    }

    if (output_index < output_nsamples && lane == window_samples / 2) {
      series_baseline[output_index] = values[shared_index];
    }
    __syncthreads();
  }
}

__global__ void interpolate_subtract_kernel(float* data,
                                             std::size_t nsamples,
                                             const float* low_res_baseline,
                                             std::size_t low_res_nsamples,
                                             std::size_t scrunch_factor) {
  const std::size_t sample =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t series = blockIdx.y;
  if (sample >= nsamples) {
    return;
  }

  const float* baseline = low_res_baseline + series * low_res_nsamples;
  const double offset = 0.5 * static_cast<double>(scrunch_factor - 1);
  const double position =
      (static_cast<double>(sample) - offset) /
      static_cast<double>(scrunch_factor);
  float value = baseline[0];
  if (position > 0.0) {
    const auto lower = static_cast<std::size_t>(floor(position));
    if (lower >= low_res_nsamples - 1) {
      value = baseline[low_res_nsamples - 1];
    } else {
      const double fraction = position - static_cast<double>(lower);
      value = static_cast<float>(
          (1.0 - fraction) * static_cast<double>(baseline[lower]) +
          fraction * static_cast<double>(baseline[lower + 1]));
    }
  }
  data[series * nsamples + sample] -= value;
}

__global__ void partial_stats_kernel(const float* input,
                                     std::size_t nsamples,
                                     CudaPartialStats* partials,
                                     std::size_t partials_per_series) {
  const std::size_t partial_index = blockIdx.x;
  const std::size_t series = blockIdx.y;
  const std::size_t begin = partial_index * kStatsItemsPerBlock;
  const std::size_t index = begin + threadIdx.x;
  const float* source = input + series * nsamples;

  double sum = 0.0;
  double sum_squares = 0.0;
  for (std::size_t item = 0; item < kStatsItemsPerThread; ++item) {
    const std::size_t sample = index + item * kThreadsPerBlock;
    if (sample < nsamples) {
      const double value = static_cast<double>(source[sample]);
      sum += value;
      sum_squares += value * value;
    }
  }

  __shared__ double shared_sum[kThreadsPerBlock];
  __shared__ double shared_squares[kThreadsPerBlock];
  shared_sum[threadIdx.x] = sum;
  shared_squares[threadIdx.x] = sum_squares;
  __syncthreads();
  for (unsigned int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
      shared_squares[threadIdx.x] += shared_squares[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[series * partials_per_series + partial_index] = CudaPartialStats{
        .sum = shared_sum[0],
        .sum_squares = shared_squares[0],
    };
  }
}

__global__ void final_stats_kernel(const CudaPartialStats* partials,
                                   std::size_t partials_per_series,
                                   std::size_t nsamples,
                                   CudaSeriesStats* stats,
                                   int reject_constant,
                                   int* status) {
  const std::size_t series = blockIdx.x;
  double sum = 0.0;
  double sum_squares = 0.0;
  for (std::size_t index = threadIdx.x; index < partials_per_series;
       index += blockDim.x) {
    const CudaPartialStats partial = partials[series * partials_per_series + index];
    sum += partial.sum;
    sum_squares += partial.sum_squares;
  }

  __shared__ double shared_sum[kThreadsPerBlock];
  __shared__ double shared_squares[kThreadsPerBlock];
  shared_sum[threadIdx.x] = sum;
  shared_squares[threadIdx.x] = sum_squares;
  __syncthreads();
  for (unsigned int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
      shared_squares[threadIdx.x] += shared_squares[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const double count = static_cast<double>(nsamples);
    const double mean = shared_sum[0] / count;
    double variance = shared_squares[0] / count - mean * mean;
    if (variance < 0.0 && variance > -1.0e-12) {
      variance = 0.0;
    }
    if (!(variance >= 0.0) || !isfinite(variance)) {
      set_status_once(status, CudaPreprocessStatus::NonFiniteInput);
      stats[series] = CudaSeriesStats{};
      return;
    }
    if (!(variance > 0.0)) {
      if (reject_constant != 0) {
        set_status_once(status, CudaPreprocessStatus::ConstantInput);
      }
      stats[series] = CudaSeriesStats{.mean = mean, .inverse_stddev = 0.0};
      return;
    }
    stats[series] = CudaSeriesStats{
        .mean = mean,
        .inverse_stddev = 1.0 / sqrt(variance),
    };
  }
}

__global__ void normalise_inplace_kernel(float* data,
                                         std::size_t nsamples,
                                         const CudaSeriesStats* stats) {
  const std::size_t sample =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t series = blockIdx.y;
  if (sample >= nsamples) {
    return;
  }
  const CudaSeriesStats series_stats = stats[series];
  data[series * nsamples + sample] = static_cast<float>(
      (static_cast<double>(data[series * nsamples + sample]) -
       series_stats.mean) *
      series_stats.inverse_stddev);
}

template <unsigned int SortSize>
void launch_running_median(const float* input,
                           std::size_t input_nsamples,
                           float* baseline,
                           std::size_t output_nsamples,
                           std::size_t window_samples,
                           std::size_t nseries,
                           cudaStream_t stream) {
  const dim3 grid(
      checked_grid_dim(ceil_div(output_nsamples, kMedianOutputsPerBlock),
                       "CUDA running median grid x overflow"),
      checked_grid_dim(nseries, "CUDA running median grid y overflow"));
  running_median_kernel<SortSize><<<grid, kThreadsPerBlock, 0, stream>>>(
      input, input_nsamples, baseline, output_nsamples, window_samples);
  check_cuda(cudaGetLastError(), "CUDA running median kernel launch");
}

void launch_running_median_dispatch(const float* input,
                                    std::size_t input_nsamples,
                                    float* baseline,
                                    std::size_t output_nsamples,
                                    std::size_t window_samples,
                                    std::size_t nseries,
                                    cudaStream_t stream) {
  if (window_samples <= 32) {
    launch_running_median<32>(input, input_nsamples, baseline,
                              output_nsamples, window_samples, nseries, stream);
  } else if (window_samples <= 64) {
    launch_running_median<64>(input, input_nsamples, baseline,
                              output_nsamples, window_samples, nseries, stream);
  } else if (window_samples <= 128) {
    launch_running_median<128>(input, input_nsamples, baseline,
                               output_nsamples, window_samples, nseries, stream);
  } else {
    launch_running_median<256>(input, input_nsamples, baseline,
                               output_nsamples, window_samples, nseries, stream);
  }
}

}  // namespace

struct CudaPreprocessProgramImpl {
  explicit CudaPreprocessProgramImpl(
      CudaPreprocessExecutionOptions execution_options)
      : execution_options(std::move(execution_options)) {}

  CudaPreprocessExecutionOptions execution_options;
  int device_id = 0;
  std::vector<CudaPreprocessStepLayout> steps;
  CudaPreprocessWorkspaceShape workspace_shape;
  CudaDeviceBuffer<float> scrunched;
  CudaDeviceBuffer<float> baseline;
  CudaDeviceBuffer<CudaPartialStats> partial_stats;
  CudaDeviceBuffer<CudaSeriesStats> series_stats;
  CudaDeviceBuffer<int> status;
  enum class RunState {
    Idle,
    Pending,
    Poisoned,
  };

  RunState run_state = RunState::Idle;
};

namespace {

void validate_batch(const CudaPreprocessProgram& program,
                    MutableCudaTimeSeriesBatchView batch) {
  if (program.empty()) {
    throw std::invalid_argument("CUDA preprocessing program must not be empty");
  }
  if (batch.data == nullptr) {
    throw std::invalid_argument("CUDA preprocessing batch data must not be null");
  }
  if (batch.nseries == 0 || batch.nseries > program.tile_capacity()) {
    throw std::invalid_argument(
        "CUDA preprocessing batch nseries must fit program tile_capacity");
  }
  if (batch.nsamples == 0 || batch.nsamples > program.max_nsamples()) {
    throw std::invalid_argument(
        "CUDA preprocessing batch nsamples must fit program max_nsamples");
  }
  if (batch.device_id != program.device_id()) {
    throw std::invalid_argument(
        "CUDA preprocessing batch device_id must match program device_id");
  }
}

void validate_plan_for_batch(
    const CudaPreprocessProgramImpl& impl,
    MutableCudaTimeSeriesBatchView batch) {
  for (const CudaPreprocessStepLayout& step : impl.steps) {
    if (step.kind != PreprocessStepKind::DetrendRunningMedian) {
      continue;
    }

    const std::size_t factor = step.detrend.scrunch_factor;
    const std::size_t reduced_nsamples = batch.nsamples / factor;
    if (factor == 1) {
      if (step.detrend.options.window_samples >= batch.nsamples) {
        throw std::invalid_argument(
            "CUDA running median window_samples must be < input size");
      }
    } else if (reduced_nsamples <= step.detrend.options.min_points) {
      throw std::invalid_argument(
          "CUDA running median low-resolution sample count must be > min_points");
    }
  }
}

void run_detrend(CudaPreprocessProgramImpl& impl,
                  MutableCudaTimeSeriesBatchView batch,
                  const CudaDetrendLayout& layout) {
  const std::size_t factor = layout.scrunch_factor;
  const std::size_t reduced_nsamples = batch.nsamples / factor;

  const float* median_input = batch.data;
  std::size_t median_input_nsamples = batch.nsamples;
  std::size_t median_output_nsamples = batch.nsamples;
  if (factor > 1) {
    const dim3 grid(
        checked_grid_dim(ceil_div(reduced_nsamples, kScrunchWarpsPerBlock),
                         "CUDA mean scrunch grid x overflow"),
        checked_grid_dim(batch.nseries, "CUDA mean scrunch grid y overflow"));
    mean_scrunch_kernel<<<grid, kThreadsPerBlock, 0,
                         impl.execution_options.stream>>>(
        batch.data, batch.nsamples, impl.scrunched.data(), reduced_nsamples,
        factor);
    check_cuda(cudaGetLastError(), "CUDA mean scrunch kernel launch");
    median_input = impl.scrunched.data();
    median_input_nsamples = reduced_nsamples;
    median_output_nsamples = reduced_nsamples;
  }

  launch_running_median_dispatch(
      median_input, median_input_nsamples, impl.baseline.data(),
      median_output_nsamples, layout.median_window, batch.nseries,
      impl.execution_options.stream);

  if (factor == 1) {
    const dim3 grid(
        checked_grid_dim(ceil_div(batch.nsamples, kThreadsPerBlock),
                         "CUDA exact detrend grid x overflow"),
        checked_grid_dim(batch.nseries, "CUDA exact detrend grid y overflow"));
    interpolate_subtract_kernel<<<grid, kThreadsPerBlock, 0,
                                  impl.execution_options.stream>>>(
        batch.data, batch.nsamples, impl.baseline.data(), batch.nsamples, 1);
  } else {
    const dim3 grid(
        checked_grid_dim(ceil_div(batch.nsamples, kThreadsPerBlock),
                         "CUDA detrend interpolation grid x overflow"),
        checked_grid_dim(batch.nseries,
                         "CUDA detrend interpolation grid y overflow"));
    interpolate_subtract_kernel<<<grid, kThreadsPerBlock, 0,
                                  impl.execution_options.stream>>>(
        batch.data, batch.nsamples, impl.baseline.data(), reduced_nsamples,
        factor);
  }
  check_cuda(cudaGetLastError(), "CUDA detrend interpolation kernel launch");
}

void run_normalise(CudaPreprocessProgramImpl& impl,
                   MutableCudaTimeSeriesBatchView batch,
                   NormaliseOptions options) {
  const std::size_t partials_per_series =
      ceil_div(batch.nsamples, kStatsItemsPerBlock);
  const dim3 partial_grid(
      checked_grid_dim(partials_per_series,
                       "CUDA normalise partial grid x overflow"),
      checked_grid_dim(batch.nseries,
                       "CUDA normalise partial grid y overflow"));
  partial_stats_kernel<<<partial_grid, kThreadsPerBlock, 0,
                         impl.execution_options.stream>>>(
      batch.data, batch.nsamples, impl.partial_stats.data(), partials_per_series);
  check_cuda(cudaGetLastError(), "CUDA normalise partial stats kernel launch");

  const dim3 final_grid(
      checked_grid_dim(batch.nseries, "CUDA normalise final grid x overflow"));
  final_stats_kernel<<<final_grid, kThreadsPerBlock, 0,
                       impl.execution_options.stream>>>(
      impl.partial_stats.data(), partials_per_series, batch.nsamples,
      impl.series_stats.data(), options.reject_constant ? 1 : 0,
      impl.status.data());
  check_cuda(cudaGetLastError(), "CUDA normalise final stats kernel launch");

  const dim3 normalise_grid(
      checked_grid_dim(ceil_div(batch.nsamples, kThreadsPerBlock),
                       "CUDA normalise grid x overflow"),
      checked_grid_dim(batch.nseries, "CUDA normalise grid y overflow"));
  normalise_inplace_kernel<<<normalise_grid, kThreadsPerBlock, 0,
                             impl.execution_options.stream>>>(
      batch.data, batch.nsamples, impl.series_stats.data());
  check_cuda(cudaGetLastError(), "CUDA normalise kernel launch");
}

}  // namespace

CudaPreprocessWorkspaceShape estimate_cuda_preprocess_workspace(
    const PreprocessPlan& plan,
    const CudaPreprocessExecutionOptions& options) {
  return compile_preprocess_layout(plan, options).workspace_shape;
}

CudaPreprocessProgram::CudaPreprocessProgram(
    PreprocessPlan plan, const CudaPreprocessProgramOptions& program_options,
    const CudaPreprocessExecutionOptions& execution_options)
    : impl_(std::make_unique<CudaPreprocessProgramImpl>(execution_options)) {
  validate_program_options(program_options);
  const CudaPreprocessBuild build =
      compile_preprocess_layout(plan, execution_options);
  impl_->steps = build.steps;
  impl_->workspace_shape = build.workspace_shape;

  check_cuda(cudaSetDevice(program_options.device_id), "cudaSetDevice");
  validate_stream_for_current_device(execution_options.stream);
  impl_->device_id = program_options.device_id;
  impl_->scrunched = CudaDeviceBuffer<float>(build.scrunched_count);
  impl_->baseline = CudaDeviceBuffer<float>(build.baseline_count);
  impl_->partial_stats = CudaDeviceBuffer<CudaPartialStats>(build.partial_count);
  impl_->series_stats =
      CudaDeviceBuffer<CudaSeriesStats>(build.series_stats_count);
  impl_->status = CudaDeviceBuffer<int>(1);
}

CudaPreprocessProgram::~CudaPreprocessProgram() = default;
CudaPreprocessProgram::CudaPreprocessProgram(CudaPreprocessProgram&&) noexcept =
    default;
CudaPreprocessProgram& CudaPreprocessProgram::operator=(
    CudaPreprocessProgram&&) noexcept = default;

bool CudaPreprocessProgram::empty() const noexcept {
  return impl_ == nullptr;
}

int CudaPreprocessProgram::device_id() const {
  if (empty()) {
    throw std::logic_error("CUDA preprocessing program must not be empty");
  }
  return impl_->device_id;
}

std::size_t CudaPreprocessProgram::tile_capacity() const {
  if (empty()) {
    throw std::logic_error("CUDA preprocessing program must not be empty");
  }
  return impl_->execution_options.series_tile_size;
}

std::size_t CudaPreprocessProgram::max_nsamples() const {
  if (empty()) {
    throw std::logic_error("CUDA preprocessing program must not be empty");
  }
  return impl_->execution_options.max_nsamples;
}

const CudaPreprocessWorkspaceShape& CudaPreprocessProgram::workspace_shape()
    const {
  if (empty()) {
    throw std::logic_error("CUDA preprocessing program must not be empty");
  }
  return impl_->workspace_shape;
}

void CudaPreprocessProgram::synchronize() {
  if (empty()) {
    throw std::logic_error("CUDA preprocessing program must not be empty");
  }
  if (impl_->run_state == CudaPreprocessProgramImpl::RunState::Poisoned) {
    throw std::logic_error(
        "CUDA preprocessing program is poisoned by an earlier CUDA failure; recreate it");
  }
  if (impl_->run_state == CudaPreprocessProgramImpl::RunState::Idle) {
    return;
  }

  int status = static_cast<int>(CudaPreprocessStatus::Ok);
  try {
    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice");
    check_cuda(cudaStreamSynchronize(impl_->execution_options.stream),
               "CUDA preprocessing synchronize");
    check_cuda(cudaMemcpy(&status, impl_->status.data(), sizeof(status),
                          cudaMemcpyDeviceToHost),
               "CUDA preprocessing status D2H");
  } catch (...) {
    impl_->run_state = CudaPreprocessProgramImpl::RunState::Poisoned;
    throw;
  }
  impl_->run_state = CudaPreprocessProgramImpl::RunState::Idle;
  switch (static_cast<CudaPreprocessStatus>(status)) {
    case CudaPreprocessStatus::Ok:
      return;
    case CudaPreprocessStatus::NonFiniteInput:
      throw std::invalid_argument(
          "CUDA preprocessing input samples must be finite");
    case CudaPreprocessStatus::ConstantInput:
      throw std::invalid_argument(
          "CUDA normalise input standard deviation must be finite and > 0");
  }
  throw std::logic_error("CUDA preprocessing status is invalid");
}

void preprocess_time_series_batch_inplace_cuda(
    CudaPreprocessProgram& program, MutableCudaTimeSeriesBatchView batch) {
  validate_batch(program, batch);
  CudaPreprocessProgramImpl& impl = *program.impl_;
  if (impl.run_state == CudaPreprocessProgramImpl::RunState::Poisoned) {
    throw std::logic_error(
        "CUDA preprocessing program is poisoned by an earlier CUDA failure; recreate it");
  }
  if (impl.run_state == CudaPreprocessProgramImpl::RunState::Pending) {
    throw std::logic_error(
        "CUDA preprocessing program already has an active run; call synchronize() before reuse");
  }
  validate_plan_for_batch(impl, batch);
  const std::size_t count = checked_multiply(
      batch.nseries, batch.nsamples,
      "CUDA preprocessing batch element count overflow");
  check_cuda(cudaSetDevice(impl.device_id), "cudaSetDevice");
  impl.run_state = CudaPreprocessProgramImpl::RunState::Pending;
  try {
    check_cuda(cudaMemsetAsync(impl.status.data(), 0, sizeof(int),
                              impl.execution_options.stream),
               "CUDA preprocessing status reset");
    validate_finite_kernel<<<
        checked_grid_dim(ceil_div(count, kThreadsPerBlock),
                         "CUDA preprocessing finite-check grid overflow"),
        kThreadsPerBlock, 0, impl.execution_options.stream>>>(batch.data,
                                                                 count,
                                                                 impl.status.data());
    check_cuda(cudaGetLastError(),
               "CUDA preprocessing finite-check kernel launch");

    for (const CudaPreprocessStepLayout& step : impl.steps) {
      switch (step.kind) {
        case PreprocessStepKind::DetrendRunningMedian:
          run_detrend(impl, batch, step.detrend);
          break;
        case PreprocessStepKind::Normalise:
          run_normalise(impl, batch, step.normalise);
          break;
      }
    }
  } catch (...) {
    impl.run_state = CudaPreprocessProgramImpl::RunState::Poisoned;
    throw;
  }
}

}  // namespace gaffa
