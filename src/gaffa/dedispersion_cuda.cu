#include "gaffa/dedispersion_cuda.h"
#include "gaffa/internal/dedispersion_delay.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace gaffa {
namespace {

template <typename T>
struct DedispersedValue;

template <>
struct DedispersedValue<std::uint8_t> {
  using type = std::uint32_t;
};

template <>
struct DedispersedValue<std::uint16_t> {
  using type = std::uint32_t;
};

template <>
struct DedispersedValue<float> {
  using type = float;
};

template <typename T>
using DedispersedValueT = typename DedispersedValue<T>::type;

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char* description) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(std::string(description) + " size overflow");
  }
  return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs,
                        const char* description) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::overflow_error(std::string(description) + " size overflow");
  }
  return lhs + rhs;
}

std::size_t ceil_div(std::size_t value, std::size_t divisor) {
  return value / divisor + static_cast<std::size_t>(value % divisor != 0);
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void launch_barrier(const char* kernel_name) {
  check_cuda(cudaGetLastError(), kernel_name);
  check_cuda(cudaDeviceSynchronize(), kernel_name);
}

void check_launch(const char* kernel_name) {
  check_cuda(cudaGetLastError(), kernel_name);
}

void synchronize_stream(cudaStream_t stream, const char* operation) {
  check_cuda(cudaStreamSynchronize(stream), operation);
}

struct CudaStreamOwner {
  cudaStream_t stream = nullptr;
  int device_id = -1;

  CudaStreamOwner() = default;

  CudaStreamOwner(const CudaStreamOwner&) = delete;
  CudaStreamOwner& operator=(const CudaStreamOwner&) = delete;

  ~CudaStreamOwner() noexcept {
    if (stream == nullptr) {
      return;
    }
    int previous_device = -1;
    if (cudaGetDevice(&previous_device) == cudaSuccess) {
      cudaSetDevice(device_id);
      cudaStreamDestroy(stream);
      cudaSetDevice(previous_device);
    } else {
      cudaStreamDestroy(stream);
    }
  }
};

template <typename T>
struct DeviceInputTileView {
  CudaSpan<const T> data;
  SampleShape shape{};
  int device_id = -1;
};

template <typename T>
struct DeviceOutputTileView {
  CudaSpan<T> data;
  DedispersedShape shape{};
  int device_id = -1;
};

struct CudaDedispersionWorkspaceBase {
  CudaStreamOwner stream;

  explicit CudaDedispersionWorkspaceBase(int device_id) {
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
    stream.device_id = device_id;
    check_cuda(cudaStreamCreateWithFlags(&stream.stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }

  CudaDedispersionWorkspaceBase(const CudaDedispersionWorkspaceBase&) = delete;
  CudaDedispersionWorkspaceBase& operator=(
      const CudaDedispersionWorkspaceBase&) = delete;
};

std::size_t block_count(std::size_t total, std::size_t threads_per_block);

template <typename InT, typename OutT>
struct CudaDedispersionWorkspace : CudaDedispersionWorkspaceBase {
  CudaDeviceBuffer<InT> input_tile;
  CudaDeviceBuffer<OutT> output_tile;
  CudaDeviceBuffer<OutT> intermediate_tile;

  explicit CudaDedispersionWorkspace(int device_id)
      : CudaDedispersionWorkspaceBase(device_id) {}
};

void validate_options(const CudaDedispersionOptions& options) {
  if (options.device_id < 0) {
    throw std::invalid_argument("CUDA device_id must be non-negative");
  }
  if (options.threads_per_block == 0 ||
      options.threads_per_block >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("CUDA threads_per_block is invalid");
  }
  if (options.memory_budget_bytes != 0 &&
      options.memory_budget_bytes < sizeof(std::uint32_t)) {
    throw std::invalid_argument("CUDA memory_budget_bytes is too small");
  }
}

template <typename T>
void validate_samples(HostSampleView<T> samples,
                      std::span<const double> frequency_mhz) {
  if (samples.data.empty()) {
    throw std::invalid_argument("dedispersion samples data must not be null");
  }
  if (samples.shape.nifs != 1) {
    throw std::invalid_argument("dedispersion requires nifs == 1");
  }
  if (samples.shape.nsamples == 0) {
    throw std::invalid_argument("dedispersion requires at least one sample");
  }
  if (samples.shape.nchans == 0) {
    throw std::invalid_argument("dedispersion requires at least one channel");
  }
  if (samples.data.size() != samples.size()) {
    throw std::invalid_argument(
        "dedispersion sample data size does not match shape");
  }
  if (frequency_mhz.size() != samples.shape.nchans) {
    throw std::invalid_argument(
        "frequency table length must match sample channel count");
  }
  for (double frequency : frequency_mhz) {
    if (!std::isfinite(frequency) || !(frequency > 0.0)) {
      throw std::invalid_argument("frequency values must be finite and positive");
    }
  }
}

void validate_channel_range(std::size_t chan_begin, std::size_t chan_end,
                            std::size_t nchans) {
  if (chan_begin >= chan_end) {
    throw std::invalid_argument("channel range must be non-empty");
  }
  if (chan_end > nchans) {
    throw std::invalid_argument("channel range exceeds sample channel count");
  }
}

void validate_plan_values(double ref_frequency_mhz, double tsamp) {
  if (!std::isfinite(ref_frequency_mhz) || !(ref_frequency_mhz > 0.0)) {
    throw std::invalid_argument("reference frequency must be positive");
  }
  if (!std::isfinite(tsamp) || !(tsamp > 0.0)) {
    throw std::invalid_argument("tsamp must be positive");
  }
}

template <typename T>
void validate_single_plan(HostSampleView<T> samples,
                          std::span<const double> frequency_mhz,
                          const SingleDmDedispersionPlan& plan,
                          const CudaDedispersionOptions& options) {
  validate_options(options);
  validate_samples(samples, frequency_mhz);
  validate_plan_values(plan.ref_frequency_mhz, plan.tsamp);
  validate_channel_range(plan.chan_begin, plan.chan_end, samples.shape.nchans);
  if (!std::isfinite(plan.dm) || plan.dm < 0.0) {
    throw std::invalid_argument("dm must be finite and non-negative");
  }
  internal::validate_nonnegative_delay_range(
      frequency_mhz, plan.ref_frequency_mhz, plan.chan_begin, plan.chan_end);
}

template <typename T>
void validate_multi_plan(HostSampleView<T> samples,
                         std::span<const double> frequency_mhz,
                         const MultiDmDedispersionPlan& plan,
                         const CudaDedispersionOptions& options) {
  validate_options(options);
  validate_samples(samples, frequency_mhz);
  validate_plan_values(plan.ref_frequency_mhz, plan.tsamp);
  if (!std::isfinite(plan.dm_low) || plan.dm_low < 0.0) {
    throw std::invalid_argument("dm_low must be finite and non-negative");
  }
  if (plan.ndm == 0) {
    throw std::invalid_argument("dedispersion requires at least one DM");
  }
  if (!std::isfinite(plan.dm_step) || !(plan.dm_step > 0.0)) {
    throw std::invalid_argument("dm_step must be positive");
  }
  validate_channel_range(plan.chan_begin, plan.chan_end, samples.shape.nchans);
  internal::validate_nonnegative_delay_range(
      frequency_mhz, plan.ref_frequency_mhz, plan.chan_begin, plan.chan_end);
}

void validate_subband_options(const SubbandDedispersionOptions& options) {
  if (options.subband_channels == 0) {
    throw std::invalid_argument("subband_channels must be positive");
  }
  if (options.ndm_per_nominal == 0) {
    throw std::invalid_argument("ndm_per_nominal must be positive");
  }
}

template <typename T>
std::vector<T> copy_span_to_vector(std::span<const T> values) {
  return std::vector<T>(values.begin(), values.end());
}

template <typename T>
CudaDeviceBuffer<T> copy_to_device(std::span<const T> values,
                               const char* operation) {
  CudaDeviceBuffer<T> buffer(values.size());
  check_cuda(cudaMemcpy(buffer.get(), values.data(), values.size() * sizeof(T),
                        cudaMemcpyHostToDevice),
             operation);
  return buffer;
}

__device__ std::int32_t device_delay_bins(double dm, double frequency_mhz,
                                          double ref_frequency_mhz,
                                          double tsamp) {
  const double frequency2 = frequency_mhz * frequency_mhz;
  const double ref2 = ref_frequency_mhz * ref_frequency_mhz;
  const double delay =
      internal::dispersion_delay_ms * dm * (1.0 / frequency2 - 1.0 / ref2);
  return static_cast<std::int32_t>(llround(delay / tsamp));
}

__global__ void compute_single_dm_delay_kernel(
    std::int32_t* delays, const double* frequency_mhz, double dm,
    double ref_frequency_mhz, double tsamp, std::size_t chan_begin,
    std::size_t channel_count) {
  const std::size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= channel_count) {
    return;
  }
  const std::size_t channel = chan_begin + offset;
  delays[offset] =
      device_delay_bins(dm, frequency_mhz[channel], ref_frequency_mhz, tsamp);
}

__global__ void compute_multi_dm_delay_kernel(
    std::int32_t* delays, const double* frequency_mhz, double dm_low,
    double dm_step, std::size_t ndm, double ref_frequency_mhz, double tsamp,
    std::size_t chan_begin, std::size_t channel_count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = ndm * channel_count;
  if (index >= total) {
    return;
  }
  const std::size_t dm_index = index / channel_count;
  const std::size_t offset = index % channel_count;
  const std::size_t channel = chan_begin + offset;
  const double dm = dm_low + static_cast<double>(dm_index) * dm_step;
  delays[index] =
      device_delay_bins(dm, frequency_mhz[channel], ref_frequency_mhz, tsamp);
}

template <typename InT, typename OutT>
__global__ void single_dm_kernel(OutT* output, const InT* input,
                                 const std::int32_t* delays,
                                 std::size_t input_nsamples,
                                 std::size_t output_nsamples,
                                 std::size_t nchans,
                                 std::size_t chan_begin,
                                 std::size_t channel_count) {
  const std::size_t time = blockIdx.x * blockDim.x + threadIdx.x;
  if (time >= output_nsamples) {
    return;
  }
  OutT sum = 0;
  for (std::size_t offset = 0; offset < channel_count; ++offset) {
    const std::size_t channel = chan_begin + offset;
    const std::int64_t shifted_time =
        static_cast<std::int64_t>(time) + delays[offset];
    if (shifted_time >= 0 &&
        shifted_time < static_cast<std::int64_t>(input_nsamples)) {
      sum += static_cast<OutT>(
          input[static_cast<std::size_t>(shifted_time) * nchans + channel]);
    }
  }
  output[time] = sum;
}

template <typename T>
__global__ void spectrum_single_dm_kernel(
    T* output, const T* input, const std::int32_t* delays,
    std::size_t input_nsamples, std::size_t output_nsamples,
    std::size_t input_nchans, std::size_t chan_begin,
    std::size_t channel_count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = output_nsamples * channel_count;
  if (index >= total) {
    return;
  }

  const std::size_t time = index / channel_count;
  const std::size_t channel_offset = index % channel_count;
  const std::size_t input_channel = chan_begin + channel_offset;
  const std::int64_t input_time =
      static_cast<std::int64_t>(time) + delays[channel_offset];
  if (input_time >= 0 &&
      input_time < static_cast<std::int64_t>(input_nsamples)) {
    output[index] =
        input[static_cast<std::size_t>(input_time) * input_nchans +
              input_channel];
  }
}

template <typename InT, typename OutT>
__global__ void multi_dm_kernel(OutT* output, const InT* input,
                                const std::int32_t* delays,
                                std::size_t ndm, std::size_t input_nsamples,
                                std::size_t output_nsamples,
                                std::size_t nchans, std::size_t chan_begin,
                                std::size_t channel_count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = ndm * output_nsamples;
  if (index >= total) {
    return;
  }
  const std::size_t dm_index = index / output_nsamples;
  const std::size_t time = index % output_nsamples;
  const std::int32_t* dm_delays = delays + dm_index * channel_count;

  OutT sum = 0;
  for (std::size_t offset = 0; offset < channel_count; ++offset) {
    const std::size_t channel = chan_begin + offset;
    const std::int64_t shifted_time =
        static_cast<std::int64_t>(time) + dm_delays[offset];
    if (shifted_time >= 0 &&
        shifted_time < static_cast<std::int64_t>(input_nsamples)) {
      sum += static_cast<OutT>(
          input[static_cast<std::size_t>(shifted_time) * nchans + channel]);
    }
  }
  output[index] = sum;
}

template <typename InT, typename OutT>
__global__ void subband_stage1_kernel(
    OutT* intermediate, const InT* input, const std::int32_t* coarse_delays,
    std::size_t nominal_dm_count, std::size_t subband_count,
    std::size_t subband_channels, std::size_t input_nsamples,
    std::size_t nchans, std::size_t chan_begin, std::size_t chan_end,
    std::size_t tile1_len, std::size_t channel_count) {
  const std::size_t time_in_tile = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t nominal_index = blockIdx.y;
  const std::size_t subband = blockIdx.z;
  if (time_in_tile >= tile1_len || nominal_index >= nominal_dm_count ||
      subband >= subband_count) {
    return;
  }

  const std::size_t subband_begin =
      chan_begin + subband * subband_channels;
  const std::size_t subband_end =
      min(subband_begin + subband_channels, chan_end);
  const std::int32_t* nominal_delays =
      coarse_delays + nominal_index * channel_count;
  OutT sum = 0;
  for (std::size_t channel = subband_begin; channel < subband_end; ++channel) {
    const std::int64_t shifted_time =
        static_cast<std::int64_t>(time_in_tile) +
        nominal_delays[channel - chan_begin];
    if (shifted_time >= 0 &&
        shifted_time < static_cast<std::int64_t>(input_nsamples)) {
      sum += static_cast<OutT>(
          input[static_cast<std::size_t>(shifted_time) * nchans + channel]);
    }
  }
  intermediate[(nominal_index * subband_count + subband) * tile1_len +
               time_in_tile] = sum;
}

template <typename OutT>
__global__ void subband_stage2_kernel(
    OutT* output, const OutT* intermediate, const std::int32_t* residual_delays,
    std::size_t ndm, std::size_t ndm_per_nominal, std::size_t subband_count,
    std::size_t tile_len, std::size_t tile1_len) {
  const std::size_t time_in_tile = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t dm_index = blockIdx.y;
  if (time_in_tile >= tile_len || dm_index >= ndm) {
    return;
  }

  const std::size_t nominal_index = dm_index / ndm_per_nominal;
  const OutT* inter_base =
      intermediate + nominal_index * subband_count * tile1_len;
  const std::int32_t* residual_base =
      residual_delays + dm_index * subband_count;

  OutT sum = 0;
  for (std::size_t subband = 0; subband < subband_count; ++subband) {
    const std::int64_t shifted_time =
        static_cast<std::int64_t>(time_in_tile) + residual_base[subband];
    if (shifted_time >= 0 &&
        shifted_time < static_cast<std::int64_t>(tile1_len)) {
      sum += inter_base[subband * tile1_len +
                        static_cast<std::size_t>(shifted_time)];
    }
  }
  output[dm_index * tile_len + time_in_tile] = sum;
}

struct SingleDmDeviceMetadata {
  CudaDeviceBuffer<double> frequency;
  CudaDeviceBuffer<std::int32_t> delays;
  std::size_t channel_count = 0;
  std::size_t max_delay = 0;
  int device_id = -1;
};

struct MultiDmDeviceMetadata {
  CudaDeviceBuffer<double> frequency;
  CudaDeviceBuffer<std::int32_t> delays;
  std::size_t ndm = 0;
  std::size_t channel_count = 0;
  std::size_t max_delay = 0;
  int device_id = -1;
};

struct SubbandDeviceMetadata {
  CudaDeviceBuffer<std::int32_t> coarse_delays;
  CudaDeviceBuffer<std::int32_t> residual_delays;
  std::size_t nominal_dm_count = 0;
  std::size_t ndm = 0;
  std::size_t subband_count = 0;
  std::size_t channel_count = 0;
  std::size_t max_coarse_delay = 0;
  std::size_t max_residual_delay = 0;
  std::size_t max_delay = 0;
  int device_id = -1;
};

template <typename T>
DeviceInputTileView<T> make_input_tile_view(
    CudaDeviceBuffer<T>& buffer, std::size_t input_nsamples,
    std::size_t nchans, int device_id) {
  return DeviceInputTileView<T>{
      .data = CudaSpan<const T>{
          .data = buffer.data(),
          .count = sample_element_count(
              SampleShape{input_nsamples, 1, nchans}),
          .device_id = device_id,
      },
      .shape = SampleShape{input_nsamples, 1, nchans},
      .device_id = device_id,
  };
}

template <typename T>
DeviceOutputTileView<T> make_output_tile_view(
    CudaDeviceBuffer<T>& buffer, std::size_t ndm,
    std::size_t output_nsamples, int device_id) {
  return DeviceOutputTileView<T>{
      .data = CudaSpan<T>{
          .data = buffer.data(),
          .count = dedispersed_element_count(
              DedispersedShape{ndm, output_nsamples}),
          .device_id = device_id,
      },
      .shape = DedispersedShape{ndm, output_nsamples},
      .device_id = device_id,
  };
}

template <typename T>
void validate_input_tile_view(const DeviceInputTileView<T>& input,
                              std::size_t expected_nsamples,
                              std::size_t expected_nchans, int device_id) {
  if (input.device_id != device_id || input.data.device_id != device_id) {
    throw std::invalid_argument("CUDA input tile belongs to another device");
  }
  if (input.shape.nifs != 1 || input.shape.nsamples != expected_nsamples ||
      input.shape.nchans != expected_nchans ||
      input.data.size() != sample_element_count(input.shape)) {
    throw std::invalid_argument("CUDA input tile shape is invalid");
  }
}

template <typename T>
void validate_output_tile_view(const DeviceOutputTileView<T>& output,
                               std::size_t expected_ndm,
                               std::size_t expected_nsamples, int device_id) {
  if (output.device_id != device_id || output.data.device_id != device_id) {
    throw std::invalid_argument("CUDA output tile belongs to another device");
  }
  const DedispersedShape expected_shape{expected_ndm, expected_nsamples};
  if (output.shape.ndm != expected_shape.ndm ||
      output.shape.nsamples != expected_shape.nsamples ||
      output.data.size() != dedispersed_element_count(expected_shape)) {
    throw std::invalid_argument("CUDA output tile shape is invalid");
  }
}

template <typename InT, typename OutT>
void launch_single_dm_tile_cuda(
    const DeviceInputTileView<InT>& input,
    const DeviceOutputTileView<OutT>& output,
    const SingleDmDeviceMetadata& metadata, std::size_t chan_begin,
    std::size_t original_nchans, std::size_t threads_per_block,
    cudaStream_t stream) {
  validate_input_tile_view(input, input.shape.nsamples, original_nchans,
                           metadata.device_id);
  validate_output_tile_view(output, 1, output.shape.nsamples,
                            metadata.device_id);
  const int threads = static_cast<int>(threads_per_block);
  single_dm_kernel<InT, OutT>
      <<<block_count(output.shape.nsamples, threads_per_block), threads, 0,
         stream>>>(output.data.data, input.data.data, metadata.delays.get(),
                   input.shape.nsamples, output.shape.nsamples, original_nchans,
                   chan_begin, metadata.channel_count);
  check_launch("single_dm_tile_kernel");
}

template <typename InT, typename OutT>
void launch_multi_dm_tile_cuda(
    const DeviceInputTileView<InT>& input,
    const DeviceOutputTileView<OutT>& output,
    const MultiDmDeviceMetadata& metadata, std::size_t chan_begin,
    std::size_t original_nchans, std::size_t threads_per_block,
    cudaStream_t stream) {
  validate_input_tile_view(input, input.shape.nsamples, original_nchans,
                           metadata.device_id);
  validate_output_tile_view(output, metadata.ndm, output.shape.nsamples,
                            metadata.device_id);
  const int threads = static_cast<int>(threads_per_block);
  const std::size_t output_size =
      checked_multiply(metadata.ndm, output.shape.nsamples,
                       "multi-DM tile output");
  multi_dm_kernel<InT, OutT>
      <<<block_count(output_size, threads_per_block), threads, 0, stream>>>(
          output.data.data, input.data.data, metadata.delays.get(),
          metadata.ndm, input.shape.nsamples, output.shape.nsamples,
          original_nchans, chan_begin, metadata.channel_count);
  check_launch("multi_dm_tile_kernel");
}

template <typename InT, typename OutT>
void launch_subband_tile_cuda(
    const DeviceInputTileView<InT>& input,
    const DeviceOutputTileView<OutT>& output,
    const SubbandDeviceMetadata& metadata,
    const SubbandDedispersionOptions& options, std::size_t chan_begin,
    std::size_t chan_end, std::size_t original_nchans,
    std::size_t tile1_len, std::size_t threads_per_block,
    cudaStream_t stream, CudaDeviceBuffer<OutT>& intermediate) {
  validate_input_tile_view(input, input.shape.nsamples, original_nchans,
                           metadata.device_id);
  validate_output_tile_view(output, metadata.ndm, output.shape.nsamples,
                            metadata.device_id);
  const int threads = static_cast<int>(threads_per_block);
  const dim3 stage1_grid(
      static_cast<unsigned int>(block_count(tile1_len, threads_per_block)),
      static_cast<unsigned int>(metadata.nominal_dm_count),
      static_cast<unsigned int>(metadata.subband_count));
  subband_stage1_kernel<InT, OutT>
      <<<stage1_grid, threads, 0, stream>>>(
          intermediate.get(), input.data.data, metadata.coarse_delays.get(),
          metadata.nominal_dm_count, metadata.subband_count,
          options.subband_channels, input.shape.nsamples, original_nchans,
          chan_begin, chan_end, tile1_len, metadata.channel_count);
  check_launch("subband_stage1_tile_kernel");

  const dim3 stage2_grid(
      static_cast<unsigned int>(
          block_count(output.shape.nsamples, threads_per_block)),
      static_cast<unsigned int>(metadata.ndm));
  subband_stage2_kernel<OutT>
      <<<stage2_grid, threads, 0, stream>>>(
          output.data.data, intermediate.get(),
          metadata.residual_delays.get(), metadata.ndm,
          options.ndm_per_nominal, metadata.subband_count,
          output.shape.nsamples, tile1_len);
  check_launch("subband_stage2_tile_kernel");
}

std::size_t block_count(std::size_t total, std::size_t threads_per_block) {
  return ceil_div(total, threads_per_block);
}

std::vector<double> make_subband_frequency(std::span<const double> frequency_mhz,
                                           const MultiDmDedispersionPlan& plan,
                                           const SubbandDedispersionOptions& options,
                                           std::size_t subband_count) {
  std::vector<double> subband_frequency(subband_count);
  for (std::size_t subband = 0; subband < subband_count; ++subband) {
    const std::size_t begin =
        plan.chan_begin + subband * options.subband_channels;
    const std::size_t end =
        std::min(begin + options.subband_channels, plan.chan_end);
    subband_frequency[subband] = frequency_mhz[(begin + end - 1) / 2];
  }
  return subband_frequency;
}

SingleDmDeviceMetadata prepare_single_dm_metadata(
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan, int device_id,
    std::size_t threads_per_block, cudaStream_t stream) {
  SingleDmDeviceMetadata metadata;
  metadata.channel_count = plan.chan_end - plan.chan_begin;
  metadata.max_delay = static_cast<std::size_t>(
      internal::single_dm_max_delay(frequency_mhz, plan));
  metadata.device_id = device_id;

  const std::vector<double> frequency_copy = copy_span_to_vector(frequency_mhz);
  metadata.frequency = copy_to_device<double>(
      std::span<const double>(frequency_copy), "copy frequency table");
  metadata.delays = CudaDeviceBuffer<std::int32_t>(metadata.channel_count);

  const int threads = static_cast<int>(threads_per_block);
  compute_single_dm_delay_kernel<<<
      block_count(metadata.channel_count, threads_per_block), threads, 0,
      stream>>>(metadata.delays.get(), metadata.frequency.get(), plan.dm,
                plan.ref_frequency_mhz, plan.tsamp, plan.chan_begin,
                metadata.channel_count);
  check_launch("compute_single_dm_delay_kernel");
  return metadata;
}

MultiDmDeviceMetadata prepare_multi_dm_metadata(
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan, int device_id,
    std::size_t threads_per_block, cudaStream_t stream) {
  MultiDmDeviceMetadata metadata;
  metadata.ndm = plan.ndm;
  metadata.channel_count = plan.chan_end - plan.chan_begin;
  metadata.max_delay = static_cast<std::size_t>(
      internal::multi_dm_max_delay(frequency_mhz, plan));
  metadata.device_id = device_id;

  const std::size_t delay_size = checked_multiply(
      metadata.ndm, metadata.channel_count, "multi-DM delay table");
  const std::vector<double> frequency_copy = copy_span_to_vector(frequency_mhz);
  metadata.frequency = copy_to_device<double>(
      std::span<const double>(frequency_copy), "copy frequency table");
  metadata.delays = CudaDeviceBuffer<std::int32_t>(delay_size);

  const int threads = static_cast<int>(threads_per_block);
  compute_multi_dm_delay_kernel<<<block_count(delay_size, threads_per_block),
                                  threads, 0, stream>>>(
      metadata.delays.get(), metadata.frequency.get(), plan.dm_low,
      plan.dm_step, plan.ndm, plan.ref_frequency_mhz, plan.tsamp,
      plan.chan_begin, metadata.channel_count);
  check_launch("compute_multi_dm_delay_kernel");
  return metadata;
}

SubbandDeviceMetadata prepare_subband_metadata(
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options, int device_id) {
  SubbandDeviceMetadata metadata;
  metadata.channel_count = plan.chan_end - plan.chan_begin;
  metadata.subband_count =
      ceil_div(metadata.channel_count, subband_options.subband_channels);
  metadata.nominal_dm_count =
      ceil_div(plan.ndm, subband_options.ndm_per_nominal);
  metadata.ndm = plan.ndm;
  metadata.max_delay = static_cast<std::size_t>(
      internal::multi_dm_max_delay(frequency_mhz, plan));
  metadata.device_id = device_id;

  const std::vector<double> subband_frequency = make_subband_frequency(
      frequency_mhz, plan, subband_options, metadata.subband_count);
  const std::vector<std::int32_t> coarse_delays =
      internal::make_subband_coarse_delay_table(
          frequency_mhz, plan, subband_options, metadata.nominal_dm_count);
  const std::vector<std::int32_t> residual_delays =
      internal::make_subband_residual_delay_table(
          std::span<const double>(subband_frequency), plan, subband_options);
  metadata.max_coarse_delay = static_cast<std::size_t>(
      internal::max_nonnegative_delay(std::span<const std::int32_t>(
          coarse_delays)));
  metadata.max_residual_delay = static_cast<std::size_t>(
      internal::max_nonnegative_delay(std::span<const std::int32_t>(
          residual_delays)));

  metadata.coarse_delays = copy_to_device<std::int32_t>(
      std::span<const std::int32_t>(coarse_delays),
      "copy subband coarse delay table");
  metadata.residual_delays = copy_to_device<std::int32_t>(
      std::span<const std::int32_t>(residual_delays),
      "copy subband residual delay table");

  // The metadata is copied before any tile is launched, so all tiles share
  // one immutable delay table.
  return metadata;
}

std::size_t tile_workspace_budget(const CudaDedispersionOptions& options) {
  if (options.memory_budget_bytes != 0) {
    return options.memory_budget_bytes;
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  (void)total_bytes;
  return free_bytes - free_bytes / 5;
}

template <typename Estimate>
std::size_t choose_time_tile_samples(
    std::size_t output_nsamples, std::size_t requested,
    const CudaDedispersionOptions& options, Estimate&& estimate) {
  const std::size_t budget = tile_workspace_budget(options);
  if (requested != 0) {
    const std::size_t tile = std::min(requested, output_nsamples);
    if (estimate(tile) > budget) {
      throw std::runtime_error(
          "CUDA time_tile_samples exceeds the device memory budget");
    }
    return tile;
  }

  std::size_t low = 1;
  std::size_t high = output_nsamples;
  std::size_t best = 0;
  while (low <= high) {
    const std::size_t middle = low + (high - low) / 2;
    if (estimate(middle) <= budget) {
      best = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  if (best == 0) {
    throw std::runtime_error(
        "CUDA device memory budget cannot hold one dedispersion tile");
  }
  return best;
}

template <typename InT>
void copy_input_tile_async(HostSampleView<InT> samples,
                           std::size_t output_begin,
                           std::size_t input_tile_nsamples,
                           CudaDeviceBuffer<InT>& device_input,
                           cudaStream_t stream) {
  const std::size_t row_size = samples.shape.nchans;
  const std::size_t element_offset =
      checked_multiply(output_begin, row_size, "input tile offset");
  const std::size_t element_count = checked_multiply(
      input_tile_nsamples, row_size, "input tile element count");
  check_cuda(cudaMemcpyAsync(
                 device_input.get(), samples.data.data() + element_offset,
                 checked_multiply(element_count, sizeof(InT),
                                  "input tile bytes"),
                 cudaMemcpyHostToDevice, stream),
             "copy input tile");
}

template <typename T>
void copy_output_tile_async(const CudaDeviceBuffer<T>& device_output,
                            std::size_t ndm, std::size_t tile_nsamples,
                            std::size_t output_nsamples,
                            std::size_t output_begin, std::vector<T>& host,
                            cudaStream_t stream) {
  const std::size_t width_bytes =
      checked_multiply(tile_nsamples, sizeof(T), "output tile bytes");
  const std::size_t host_pitch =
      checked_multiply(output_nsamples, sizeof(T), "host output pitch");
  check_cuda(cudaMemcpy2DAsync(
                 host.data() + output_begin, host_pitch, device_output.get(),
                 width_bytes, width_bytes, ndm, cudaMemcpyDeviceToHost,
                 stream),
             "copy output tile");
}

template <typename InT, typename OutT>
std::size_t direct_tile_workspace_bytes(
    std::size_t tile_nsamples, std::size_t input_nsamples,
    std::size_t max_delay, std::size_t nchans, std::size_t ndm) {
  const std::size_t input_nsamples_tile = std::min(
      checked_add(tile_nsamples, max_delay, "direct input tile"),
      input_nsamples);
  const std::size_t input_bytes = checked_multiply(
      checked_multiply(input_nsamples_tile, nchans, "direct input tile"),
      sizeof(InT), "direct input tile");
  const std::size_t output_bytes = checked_multiply(
      checked_multiply(tile_nsamples, ndm, "direct output tile"),
      sizeof(OutT), "direct output tile");
  return checked_add(input_bytes, output_bytes, "direct tile workspace");
}

template <typename InT, typename OutT>
std::size_t subband_tile_workspace_bytes(
    std::size_t tile_nsamples, std::size_t input_nsamples,
    const SubbandDeviceMetadata& metadata,
    const SubbandDedispersionOptions& options) {
  const std::size_t tile1_len = std::min(
      checked_add(tile_nsamples, metadata.max_residual_delay,
                  "subband intermediate tile"),
      input_nsamples);
  const std::size_t input_nsamples_tile = std::min(
      checked_add(tile1_len, metadata.max_coarse_delay,
                  "subband input tile"),
      input_nsamples);
  const std::size_t input_bytes = checked_multiply(
      checked_multiply(input_nsamples_tile, metadata.channel_count,
                       "subband input tile"),
      sizeof(InT), "subband input tile");
  const std::size_t output_bytes = checked_multiply(
      checked_multiply(tile_nsamples, metadata.ndm, "subband output tile"),
      sizeof(OutT), "subband output tile");
  const std::size_t intermediate_bytes = checked_multiply(
      checked_multiply(
          checked_multiply(metadata.nominal_dm_count, metadata.subband_count,
                           "subband intermediate tile"),
          tile1_len, "subband intermediate tile"),
      sizeof(OutT), "subband intermediate tile");
  (void)options;
  return checked_add(checked_add(input_bytes, output_bytes,
                                 "subband tile workspace"),
                     intermediate_bytes, "subband tile workspace");
}

template <typename InT, typename OutT>
DedispersedResult<OutT> dedisperse_single_dm_cuda_host_tiled_impl(
    HostSampleView<InT> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  validate_single_plan(samples, frequency_mhz, plan, options);
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");

  CudaDedispersionWorkspace<InT, OutT> workspace(options.device_id);
  SingleDmDeviceMetadata metadata = prepare_single_dm_metadata(
      frequency_mhz, plan, options.device_id, options.threads_per_block,
      workspace.stream.stream);
  const std::size_t output_nsamples = internal::valid_output_nsamples(
      samples.shape.nsamples, static_cast<std::int32_t>(metadata.max_delay));

  DedispersedResult<OutT> result;
  result.shape = DedispersedShape{1, output_nsamples};
  result.data.resize(output_nsamples);

  const auto estimate = [&](std::size_t tile_nsamples) {
    return direct_tile_workspace_bytes<InT, OutT>(
        tile_nsamples, samples.shape.nsamples, metadata.max_delay,
        samples.shape.nchans, 1);
  };
  const std::size_t tile_nsamples = choose_time_tile_samples(
      output_nsamples, options.time_tile_samples, options, estimate);
  const std::size_t max_input_nsamples = std::min(
      checked_add(tile_nsamples, metadata.max_delay, "single-DM input tile"),
      samples.shape.nsamples);
  workspace.input_tile = CudaDeviceBuffer<InT>(checked_multiply(
      max_input_nsamples, samples.shape.nchans, "single-DM input tile"));
  workspace.output_tile = CudaDeviceBuffer<OutT>(tile_nsamples);

  for (std::size_t output_begin = 0; output_begin < output_nsamples;
       output_begin += tile_nsamples) {
    const std::size_t output_count =
        std::min(tile_nsamples, output_nsamples - output_begin);
    const std::size_t input_count = std::min(
        checked_add(output_count, metadata.max_delay,
                    "single-DM input tile"),
        samples.shape.nsamples - output_begin);
    copy_input_tile_async(samples, output_begin, input_count,
                          workspace.input_tile, workspace.stream.stream);
    const auto input_view = make_input_tile_view(
        workspace.input_tile, input_count, samples.shape.nchans,
        options.device_id);
    const auto output_view = make_output_tile_view(
        workspace.output_tile, 1, output_count, options.device_id);
    launch_single_dm_tile_cuda<InT, OutT>(
        input_view, output_view, metadata, plan.chan_begin,
        samples.shape.nchans, options.threads_per_block,
        workspace.stream.stream);
    copy_output_tile_async(workspace.output_tile, 1, output_count,
                           output_nsamples, output_begin, result.data,
                           workspace.stream.stream);
  }
  synchronize_stream(workspace.stream.stream, "single-DM CUDA tiles");
  return result;
}

template <typename InT, typename OutT>
DedispersedResult<OutT> dedisperse_multi_dm_cuda_host_tiled_impl(
    HostSampleView<InT> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  validate_multi_plan(samples, frequency_mhz, plan, options);
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");

  CudaDedispersionWorkspace<InT, OutT> workspace(options.device_id);
  MultiDmDeviceMetadata metadata = prepare_multi_dm_metadata(
      frequency_mhz, plan, options.device_id, options.threads_per_block,
      workspace.stream.stream);
  const std::size_t output_nsamples = internal::valid_output_nsamples(
      samples.shape.nsamples, static_cast<std::int32_t>(metadata.max_delay));

  DedispersedResult<OutT> result;
  result.shape = DedispersedShape{plan.ndm, output_nsamples};
  result.data.resize(
      checked_multiply(plan.ndm, output_nsamples, "dedispersed output"));

  const auto estimate = [&](std::size_t tile_nsamples) {
    return direct_tile_workspace_bytes<InT, OutT>(
        tile_nsamples, samples.shape.nsamples, metadata.max_delay,
        samples.shape.nchans, plan.ndm);
  };
  const std::size_t tile_nsamples = choose_time_tile_samples(
      output_nsamples, options.time_tile_samples, options, estimate);
  const std::size_t max_input_nsamples = std::min(
      checked_add(tile_nsamples, metadata.max_delay, "multi-DM input tile"),
      samples.shape.nsamples);
  workspace.input_tile = CudaDeviceBuffer<InT>(checked_multiply(
      max_input_nsamples, samples.shape.nchans, "multi-DM input tile"));
  workspace.output_tile = CudaDeviceBuffer<OutT>(checked_multiply(
      plan.ndm, tile_nsamples, "multi-DM output tile"));

  for (std::size_t output_begin = 0; output_begin < output_nsamples;
       output_begin += tile_nsamples) {
    const std::size_t output_count =
        std::min(tile_nsamples, output_nsamples - output_begin);
    const std::size_t input_count = std::min(
        checked_add(output_count, metadata.max_delay,
                    "multi-DM input tile"),
        samples.shape.nsamples - output_begin);
    copy_input_tile_async(samples, output_begin, input_count,
                          workspace.input_tile, workspace.stream.stream);
    const auto input_view = make_input_tile_view(
        workspace.input_tile, input_count, samples.shape.nchans,
        options.device_id);
    const auto output_view = make_output_tile_view(
        workspace.output_tile, plan.ndm, output_count, options.device_id);
    launch_multi_dm_tile_cuda<InT, OutT>(
        input_view, output_view, metadata, plan.chan_begin,
        samples.shape.nchans, options.threads_per_block,
        workspace.stream.stream);
    copy_output_tile_async(workspace.output_tile, plan.ndm, output_count,
                           output_nsamples, output_begin, result.data,
                           workspace.stream.stream);
  }
  synchronize_stream(workspace.stream.stream, "multi-DM CUDA tiles");
  return result;
}

template <typename InT, typename OutT>
DedispersedResult<OutT> dedisperse_subband_cuda_host_tiled_impl(
    HostSampleView<InT> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  validate_multi_plan(samples, frequency_mhz, plan, cuda_options);
  validate_subband_options(subband_options);
  check_cuda(cudaSetDevice(cuda_options.device_id), "cudaSetDevice");

  CudaDedispersionWorkspace<InT, OutT> workspace(cuda_options.device_id);
  SubbandDeviceMetadata metadata = prepare_subband_metadata(
      frequency_mhz, plan, subband_options, cuda_options.device_id);
  const std::size_t output_nsamples = internal::valid_output_nsamples(
      samples.shape.nsamples, static_cast<std::int32_t>(metadata.max_delay));

  DedispersedResult<OutT> result;
  result.shape = DedispersedShape{plan.ndm, output_nsamples};
  result.data.resize(
      checked_multiply(plan.ndm, output_nsamples, "dedispersed output"));

  const auto estimate = [&](std::size_t tile_nsamples) {
    return subband_tile_workspace_bytes<InT, OutT>(
        tile_nsamples, samples.shape.nsamples, metadata, subband_options);
  };
  const std::size_t tile_nsamples = choose_time_tile_samples(
      output_nsamples, cuda_options.time_tile_samples, cuda_options, estimate);
  const std::size_t max_tile1_len = std::min(
      checked_add(tile_nsamples, metadata.max_residual_delay,
                  "subband intermediate tile"),
      samples.shape.nsamples);
  const std::size_t max_input_nsamples = std::min(
      checked_add(max_tile1_len, metadata.max_coarse_delay,
                  "subband input tile"),
      samples.shape.nsamples);
  workspace.input_tile = CudaDeviceBuffer<InT>(checked_multiply(
      max_input_nsamples, samples.shape.nchans, "subband input tile"));
  workspace.output_tile = CudaDeviceBuffer<OutT>(checked_multiply(
      plan.ndm, tile_nsamples, "subband output tile"));
  workspace.intermediate_tile = CudaDeviceBuffer<OutT>(checked_multiply(
      checked_multiply(metadata.nominal_dm_count, metadata.subband_count,
                       "subband intermediate tile"),
      max_tile1_len, "subband intermediate tile"));

  for (std::size_t output_begin = 0; output_begin < output_nsamples;
       output_begin += tile_nsamples) {
    const std::size_t output_count =
        std::min(tile_nsamples, output_nsamples - output_begin);
    const std::size_t tile1_len = std::min(
        checked_add(output_count, metadata.max_residual_delay,
                    "subband intermediate tile"),
        samples.shape.nsamples - output_begin);
    const std::size_t input_count = std::min(
        checked_add(tile1_len, metadata.max_coarse_delay,
                    "subband input tile"),
        samples.shape.nsamples - output_begin);
    copy_input_tile_async(samples, output_begin, input_count,
                          workspace.input_tile, workspace.stream.stream);
    const auto input_view = make_input_tile_view(
        workspace.input_tile, input_count, samples.shape.nchans,
        cuda_options.device_id);
    const auto output_view = make_output_tile_view(
        workspace.output_tile, plan.ndm, output_count,
        cuda_options.device_id);
    launch_subband_tile_cuda<InT, OutT>(
        input_view, output_view, metadata, subband_options, plan.chan_begin,
        plan.chan_end, samples.shape.nchans, tile1_len,
        cuda_options.threads_per_block, workspace.stream.stream,
        workspace.intermediate_tile);
    copy_output_tile_async(workspace.output_tile, plan.ndm, output_count,
                           output_nsamples, output_begin, result.data,
                           workspace.stream.stream);
  }
  synchronize_stream(workspace.stream.stream, "subband CUDA tiles");
  return result;
}

template <typename T>
CudaDedispersedResult<DedispersedValueT<T>> dedisperse_single_dm_cuda_device_impl(
    HostSampleView<T> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  validate_single_plan(samples, frequency_mhz, plan, options);
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");

  using OutT = DedispersedValueT<T>;
  const std::size_t input_size = samples.size();
  const std::size_t channel_count = plan.chan_end - plan.chan_begin;
  const std::size_t output_nsamples =
      internal::valid_output_nsamples(
          samples.shape.nsamples, internal::single_dm_max_delay(frequency_mhz,
                                                                plan));
  const int threads = static_cast<int>(options.threads_per_block);
  const std::vector<double> frequency_copy = copy_span_to_vector(frequency_mhz);
  const CudaDeviceBuffer<T> device_input =
      copy_to_device<T>(samples.data.first(input_size), "copy samples");
  const CudaDeviceBuffer<double> device_frequency =
      copy_to_device<double>(std::span<const double>(frequency_copy),
                             "copy frequency table");
  CudaDeviceBuffer<std::int32_t> device_delays(channel_count);
  CudaDeviceBuffer<OutT> device_output(output_nsamples);

  compute_single_dm_delay_kernel<<<block_count(channel_count,
                                                options.threads_per_block),
                                   threads>>>(
      device_delays.get(), device_frequency.get(), plan.dm,
      plan.ref_frequency_mhz, plan.tsamp, plan.chan_begin, channel_count);
  launch_barrier("compute_single_dm_delay_kernel");

  single_dm_kernel<T, OutT>
      <<<block_count(output_nsamples, options.threads_per_block),
         threads>>>(device_output.get(), device_input.get(),
                    device_delays.get(), samples.shape.nsamples,
                    output_nsamples, samples.shape.nchans, plan.chan_begin,
                    channel_count);
  launch_barrier("single_dm_kernel");

  CudaDedispersedResult<OutT> result;
  result.shape = DedispersedShape{1, output_nsamples};
  result.data = std::move(device_output);
  result.device_id = options.device_id;
  return result;
}

template <typename T>
CudaDedispersedSpectrum<T> dedisperse_spectrum_cuda_device_impl(
    HostSampleView<T> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  validate_single_plan(samples, frequency_mhz, plan, options);
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");

  const std::size_t input_size = samples.size();
  const std::size_t channel_count = plan.chan_end - plan.chan_begin;
  const std::size_t output_nsamples =
      internal::valid_output_nsamples(
          samples.shape.nsamples, internal::single_dm_max_delay(frequency_mhz,
                                                                plan));
  const std::size_t output_size =
      checked_multiply(output_nsamples, channel_count, "dedispersed spectrum");
  const int threads = static_cast<int>(options.threads_per_block);
  const std::vector<double> frequency_copy = copy_span_to_vector(frequency_mhz);
  const CudaDeviceBuffer<T> device_input =
      copy_to_device<T>(samples.data.first(input_size), "copy samples");
  const CudaDeviceBuffer<double> device_frequency =
      copy_to_device<double>(std::span<const double>(frequency_copy),
                             "copy frequency table");
  CudaDeviceBuffer<std::int32_t> device_delays(channel_count);
  CudaDeviceBuffer<T> device_output(output_size);

  compute_single_dm_delay_kernel<<<block_count(channel_count,
                                                options.threads_per_block),
                                   threads>>>(
      device_delays.get(), device_frequency.get(), plan.dm,
      plan.ref_frequency_mhz, plan.tsamp, plan.chan_begin, channel_count);
  launch_barrier("compute_single_dm_delay_kernel");

  spectrum_single_dm_kernel<T>
      <<<block_count(output_size, options.threads_per_block), threads>>>(
          device_output.get(), device_input.get(), device_delays.get(),
          samples.shape.nsamples, output_nsamples, samples.shape.nchans,
          plan.chan_begin, channel_count);
  launch_barrier("spectrum_single_dm_kernel");

  CudaDedispersedSpectrum<T> result;
  result.shape = SampleShape{output_nsamples, 1, channel_count};
  result.data = std::move(device_output);
  result.dm = plan.dm;
  result.tsamp = plan.tsamp;
  result.chan_begin = plan.chan_begin;
  result.chan_end = plan.chan_end;
  result.device_id = options.device_id;
  return result;
}

template <typename T>
CudaDedispersedResult<DedispersedValueT<T>> dedisperse_multi_dm_cuda_device_impl(
    HostSampleView<T> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  validate_multi_plan(samples, frequency_mhz, plan, options);
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");

  using OutT = DedispersedValueT<T>;
  const std::size_t input_size = samples.size();
  const std::size_t channel_count = plan.chan_end - plan.chan_begin;
  const std::size_t delay_size =
      checked_multiply(plan.ndm, channel_count, "multi-DM delay table");
  const std::size_t output_nsamples =
      internal::valid_output_nsamples(
          samples.shape.nsamples, internal::multi_dm_max_delay(frequency_mhz,
                                                               plan));
  const std::size_t output_size =
      checked_multiply(plan.ndm, output_nsamples, "dedispersed output");
  const int threads = static_cast<int>(options.threads_per_block);
  const std::vector<double> frequency_copy = copy_span_to_vector(frequency_mhz);
  const CudaDeviceBuffer<T> device_input =
      copy_to_device<T>(samples.data.first(input_size), "copy samples");
  const CudaDeviceBuffer<double> device_frequency =
      copy_to_device<double>(std::span<const double>(frequency_copy),
                             "copy frequency table");
  CudaDeviceBuffer<std::int32_t> device_delays(delay_size);
  CudaDeviceBuffer<OutT> device_output(output_size);

  compute_multi_dm_delay_kernel<<<block_count(delay_size,
                                               options.threads_per_block),
                                  threads>>>(
      device_delays.get(), device_frequency.get(), plan.dm_low, plan.dm_step,
      plan.ndm, plan.ref_frequency_mhz, plan.tsamp, plan.chan_begin,
      channel_count);
  launch_barrier("compute_multi_dm_delay_kernel");

  multi_dm_kernel<T, OutT>
      <<<block_count(output_size, options.threads_per_block), threads>>>(
          device_output.get(), device_input.get(), device_delays.get(),
          plan.ndm, samples.shape.nsamples, output_nsamples,
          samples.shape.nchans, plan.chan_begin, channel_count);
  launch_barrier("multi_dm_kernel");

  CudaDedispersedResult<OutT> result;
  result.shape = DedispersedShape{plan.ndm, output_nsamples};
  result.data = std::move(device_output);
  result.device_id = options.device_id;
  return result;
}

template <typename T>
CudaDedispersedResult<DedispersedValueT<T>> dedisperse_subband_cuda_device_impl(
    HostSampleView<T> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  validate_multi_plan(samples, frequency_mhz, plan, cuda_options);
  validate_subband_options(subband_options);
  check_cuda(cudaSetDevice(cuda_options.device_id), "cudaSetDevice");

  using OutT = DedispersedValueT<T>;
  const std::size_t input_size = samples.size();
  const std::size_t channel_count = plan.chan_end - plan.chan_begin;
  const std::size_t subband_count =
      ceil_div(channel_count, subband_options.subband_channels);
  const std::size_t nominal_dm_count =
      ceil_div(plan.ndm, subband_options.ndm_per_nominal);
  const std::size_t coarse_delay_size = checked_multiply(
      nominal_dm_count, channel_count, "subband coarse delay table");
  const std::size_t residual_delay_size =
      checked_multiply(plan.ndm, subband_count, "subband residual delay table");
  const std::size_t output_nsamples =
      internal::valid_output_nsamples(
          samples.shape.nsamples, internal::multi_dm_max_delay(frequency_mhz,
                                                               plan));
  const std::size_t output_size =
      checked_multiply(plan.ndm, output_nsamples, "dedispersed output");
  const int threads = static_cast<int>(cuda_options.threads_per_block);

  const std::vector<double> frequency_copy = copy_span_to_vector(frequency_mhz);
  const std::vector<double> subband_frequency = make_subband_frequency(
      frequency_mhz, plan, subband_options, subband_count);
  const std::vector<std::int32_t> coarse_delays =
      internal::make_subband_coarse_delay_table(
          frequency_mhz, plan, subband_options, nominal_dm_count);
  const std::vector<std::int32_t> residual_delays =
      internal::make_subband_residual_delay_table(
          std::span<const double>(subband_frequency), plan, subband_options);
  const std::size_t max_coarse_delay = static_cast<std::size_t>(
      internal::max_nonnegative_delay(std::span<const std::int32_t>(
          coarse_delays)));
  const std::size_t max_residual_delay = static_cast<std::size_t>(
      internal::max_nonnegative_delay(std::span<const std::int32_t>(
          residual_delays)));

  const CudaDeviceBuffer<T> device_input =
      copy_to_device<T>(samples.data.first(input_size), "copy samples");
  const CudaDeviceBuffer<std::int32_t> device_coarse_delays =
      copy_to_device<std::int32_t>(
          std::span<const std::int32_t>(coarse_delays),
          "copy subband coarse delay table");
  const CudaDeviceBuffer<std::int32_t> device_residual_delays =
      copy_to_device<std::int32_t>(
          std::span<const std::int32_t>(residual_delays),
          "copy subband residual delay table");
  CudaDeviceBuffer<OutT> device_output(output_size);
  const std::size_t tile1_len = std::min(
      checked_add(output_nsamples, max_residual_delay,
                  "subband intermediate"),
      samples.shape.nsamples);
  const std::size_t intermediate_size =
      checked_multiply(checked_multiply(nominal_dm_count, subband_count,
                                        "subband intermediate"),
                       tile1_len, "subband intermediate");
  CudaDeviceBuffer<OutT> device_intermediate(intermediate_size);
  const dim3 stage1_grid(
      static_cast<unsigned int>(
          block_count(tile1_len, cuda_options.threads_per_block)),
      static_cast<unsigned int>(nominal_dm_count),
      static_cast<unsigned int>(subband_count));
  subband_stage1_kernel<T, OutT>
      <<<stage1_grid, threads>>>(
          device_intermediate.get(), device_input.get(),
          device_coarse_delays.get(),
          nominal_dm_count, subband_count, subband_options.subband_channels,
          samples.shape.nsamples, samples.shape.nchans, plan.chan_begin,
          plan.chan_end, tile1_len, channel_count);
  launch_barrier("subband_stage1_kernel");

  const dim3 stage2_grid(
      static_cast<unsigned int>(block_count(output_nsamples,
                                             cuda_options.threads_per_block)),
      static_cast<unsigned int>(plan.ndm));
  subband_stage2_kernel<OutT>
      <<<stage2_grid, threads>>>(device_output.get(), device_intermediate.get(),
                                  device_residual_delays.get(), plan.ndm,
                                  subband_options.ndm_per_nominal,
                                  subband_count, output_nsamples, tile1_len);
  launch_barrier("subband_stage2_kernel");

  CudaDedispersedResult<OutT> result;
  result.shape = DedispersedShape{plan.ndm, output_nsamples};
  result.data = std::move(device_output);
  result.device_id = cuda_options.device_id;
  return result;
}

}  // namespace

template <typename T>
DedispersedResult<T> copy_to_host_impl(const CudaDedispersedResult<T>& result) {
  DedispersedResult<T> host;
  host.shape = result.shape;
  host.data.resize(result.size());
  check_cuda(cudaSetDevice(result.device_id), "cudaSetDevice");
  check_cuda(cudaMemcpy(host.data.data(), result.data.data(), result.bytes(),
                        cudaMemcpyDeviceToHost),
             "copy dedispersed output to host");
  return host;
}

DedispersedResult<std::uint32_t> copy_to_host(
    const CudaDedispersedResult<std::uint32_t>& result) {
  return copy_to_host_impl(result);
}

DedispersedResult<float> copy_to_host(
    const CudaDedispersedResult<float>& result) {
  return copy_to_host_impl(result);
}

template <typename T>
DedispersedSpectrum<T> copy_spectrum_to_host_impl(
    const CudaDedispersedSpectrum<T>& result) {
  DedispersedSpectrum<T> host;
  host.shape = result.shape;
  host.data.resize(result.size());
  host.dm = result.dm;
  host.tsamp = result.tsamp;
  host.chan_begin = result.chan_begin;
  host.chan_end = result.chan_end;
  check_cuda(cudaSetDevice(result.device_id), "cudaSetDevice");
  check_cuda(cudaMemcpy(host.data.data(), result.data.data(), result.bytes(),
                        cudaMemcpyDeviceToHost),
             "copy dedispersed spectrum output to host");
  return host;
}

DedispersedSpectrum<std::uint8_t> copy_to_host(
    const CudaDedispersedSpectrum<std::uint8_t>& result) {
  return copy_spectrum_to_host_impl(result);
}

DedispersedSpectrum<std::uint16_t> copy_to_host(
    const CudaDedispersedSpectrum<std::uint16_t>& result) {
  return copy_spectrum_to_host_impl(result);
}

DedispersedSpectrum<float> copy_to_host(
    const CudaDedispersedSpectrum<float>& result) {
  return copy_spectrum_to_host_impl(result);
}

DedispersedSpectrum<std::uint8_t> dedisperse_spectrum_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return copy_to_host(
      dedisperse_spectrum_cuda_device(samples, frequency_mhz, plan, options));
}

DedispersedSpectrum<std::uint16_t> dedisperse_spectrum_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return copy_to_host(
      dedisperse_spectrum_cuda_device(samples, frequency_mhz, plan, options));
}

DedispersedSpectrum<float> dedisperse_spectrum_cuda(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return copy_to_host(
      dedisperse_spectrum_cuda_device(samples, frequency_mhz, plan, options));
}

DedispersedResult<std::uint32_t> dedisperse_single_dm_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_single_dm_cuda_host_tiled_impl<std::uint8_t, std::uint32_t>(
      samples, frequency_mhz, plan, options);
}

DedispersedResult<std::uint32_t> dedisperse_single_dm_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_single_dm_cuda_host_tiled_impl<std::uint16_t,
                                                   std::uint32_t>(
      samples, frequency_mhz, plan, options);
}

DedispersedResult<float> dedisperse_single_dm_cuda(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_single_dm_cuda_host_tiled_impl<float, float>(
      samples, frequency_mhz, plan, options);
}

DedispersedResult<std::uint32_t> dedisperse_multi_dm_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_multi_dm_cuda_host_tiled_impl<std::uint8_t, std::uint32_t>(
      samples, frequency_mhz, plan, options);
}

DedispersedResult<std::uint32_t> dedisperse_multi_dm_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_multi_dm_cuda_host_tiled_impl<std::uint16_t,
                                                  std::uint32_t>(
      samples, frequency_mhz, plan, options);
}

DedispersedResult<float> dedisperse_multi_dm_cuda(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_multi_dm_cuda_host_tiled_impl<float, float>(
      samples, frequency_mhz, plan, options);
}

DedispersedResult<std::uint32_t> dedisperse_subband_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  return dedisperse_subband_cuda_host_tiled_impl<std::uint8_t, std::uint32_t>(
      samples, frequency_mhz, plan, subband_options, cuda_options);
}

DedispersedResult<std::uint32_t> dedisperse_subband_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  return dedisperse_subband_cuda_host_tiled_impl<std::uint16_t,
                                                 std::uint32_t>(
      samples, frequency_mhz, plan, subband_options, cuda_options);
}

DedispersedResult<float> dedisperse_subband_cuda(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  return dedisperse_subband_cuda_host_tiled_impl<float, float>(
      samples, frequency_mhz, plan, subband_options, cuda_options);
}

CudaDedispersedResult<std::uint32_t> dedisperse_single_dm_cuda_device(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_single_dm_cuda_device_impl(samples, frequency_mhz, plan,
                                               options);
}

CudaDedispersedResult<std::uint32_t> dedisperse_single_dm_cuda_device(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_single_dm_cuda_device_impl(samples, frequency_mhz, plan,
                                               options);
}

CudaDedispersedResult<float> dedisperse_single_dm_cuda_device(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_single_dm_cuda_device_impl(samples, frequency_mhz, plan,
                                               options);
}

CudaDedispersedSpectrum<std::uint8_t> dedisperse_spectrum_cuda_device(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_spectrum_cuda_device_impl(samples, frequency_mhz, plan,
                                              options);
}

CudaDedispersedSpectrum<std::uint16_t> dedisperse_spectrum_cuda_device(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_spectrum_cuda_device_impl(samples, frequency_mhz, plan,
                                              options);
}

CudaDedispersedSpectrum<float> dedisperse_spectrum_cuda_device(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_spectrum_cuda_device_impl(samples, frequency_mhz, plan,
                                              options);
}

CudaDedispersedResult<std::uint32_t> dedisperse_multi_dm_cuda_device(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_multi_dm_cuda_device_impl(samples, frequency_mhz, plan,
                                              options);
}

CudaDedispersedResult<std::uint32_t> dedisperse_multi_dm_cuda_device(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_multi_dm_cuda_device_impl(samples, frequency_mhz, plan,
                                              options);
}

CudaDedispersedResult<float> dedisperse_multi_dm_cuda_device(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return dedisperse_multi_dm_cuda_device_impl(samples, frequency_mhz, plan,
                                              options);
}

CudaDedispersedResult<std::uint32_t> dedisperse_subband_cuda_device(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  return dedisperse_subband_cuda_device_impl(samples, frequency_mhz, plan,
                                             subband_options, cuda_options);
}

CudaDedispersedResult<std::uint32_t> dedisperse_subband_cuda_device(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  return dedisperse_subband_cuda_device_impl(samples, frequency_mhz, plan,
                                             subband_options, cuda_options);
}

CudaDedispersedResult<float> dedisperse_subband_cuda_device(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  return dedisperse_subband_cuda_device_impl(samples, frequency_mhz, plan,
                                             subband_options, cuda_options);
}

}  // namespace gaffa
