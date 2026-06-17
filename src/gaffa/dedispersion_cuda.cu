#include "gaffa/dedispersion_cuda.h"

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

inline constexpr double dispersion_delay_ms = 4148.808;

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

void validate_options(const CudaDedispersionOptions& options) {
  if (options.device_id < 0) {
    throw std::invalid_argument("CUDA device_id must be non-negative");
  }
  if (options.threads_per_block == 0 ||
      options.threads_per_block >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("CUDA threads_per_block is invalid");
  }
  if (options.time_tile_samples == 0) {
    throw std::invalid_argument("CUDA time_tile_samples must be positive");
  }
}

template <typename T>
void validate_samples(HostSampleView<T> samples,
                      std::span<const double> frequency_mhz) {
  if (samples.data == nullptr) {
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

void validate_nonnegative_delay_range(std::span<const double> frequency_mhz,
                                      double ref_frequency_mhz,
                                      std::size_t chan_begin,
                                      std::size_t chan_end) {
  for (std::size_t channel = chan_begin; channel < chan_end; ++channel) {
    if (frequency_mhz[channel] > ref_frequency_mhz) {
      throw std::invalid_argument(
          "CUDA dedispersion requires selected frequencies <= reference "
          "frequency");
    }
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
  validate_nonnegative_delay_range(frequency_mhz, plan.ref_frequency_mhz,
                                   plan.chan_begin, plan.chan_end);
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
  validate_nonnegative_delay_range(frequency_mhz, plan.ref_frequency_mhz,
                                   plan.chan_begin, plan.chan_end);
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
      dispersion_delay_ms * dm * (1.0 / frequency2 - 1.0 / ref2);
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

__global__ void compute_subband_coarse_delay_kernel(
    std::int32_t* delays, const double* frequency_mhz, double dm_low,
    double dm_step, std::size_t ndm_per_nominal,
    std::size_t nominal_dm_count, double ref_frequency_mhz, double tsamp,
    std::size_t chan_begin, std::size_t channel_count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = nominal_dm_count * channel_count;
  if (index >= total) {
    return;
  }
  const std::size_t nominal_index = index / channel_count;
  const std::size_t offset = index % channel_count;
  const std::size_t channel = chan_begin + offset;
  const double nominal_dm =
      dm_low + static_cast<double>(nominal_index * ndm_per_nominal) * dm_step;
  delays[index] = device_delay_bins(nominal_dm, frequency_mhz[channel],
                                    ref_frequency_mhz, tsamp);
}

__global__ void compute_subband_residual_delay_kernel(
    std::int32_t* delays, const double* subband_frequency, double dm_low,
    double dm_step, std::size_t ndm, std::size_t ndm_per_nominal,
    std::size_t subband_count, double ref_frequency_mhz, double tsamp) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = ndm * subband_count;
  if (index >= total) {
    return;
  }
  const std::size_t dm_index = index / subband_count;
  const std::size_t subband = index % subband_count;
  const std::size_t nominal_index = dm_index / ndm_per_nominal;
  const double nominal_dm =
      dm_low + static_cast<double>(nominal_index * ndm_per_nominal) * dm_step;
  const double final_dm = dm_low + static_cast<double>(dm_index) * dm_step;
  delays[index] = device_delay_bins(final_dm - nominal_dm,
                                    subband_frequency[subband],
                                    ref_frequency_mhz, tsamp);
}

template <typename InT, typename OutT>
__global__ void single_dm_kernel(OutT* output, const InT* input,
                                 const std::int32_t* delays,
                                 std::size_t nsamples, std::size_t nchans,
                                 std::size_t chan_begin,
                                 std::size_t channel_count) {
  const std::size_t time = blockIdx.x * blockDim.x + threadIdx.x;
  if (time >= nsamples) {
    return;
  }
  OutT sum = 0;
  for (std::size_t offset = 0; offset < channel_count; ++offset) {
    const std::size_t channel = chan_begin + offset;
    const std::int64_t shifted_time =
        static_cast<std::int64_t>(time) + delays[offset];
    if (shifted_time >= 0 &&
        shifted_time < static_cast<std::int64_t>(nsamples)) {
      sum += static_cast<OutT>(
          input[static_cast<std::size_t>(shifted_time) * nchans + channel]);
    }
  }
  output[time] = sum;
}

template <typename InT, typename OutT>
__global__ void multi_dm_kernel(OutT* output, const InT* input,
                                const std::int32_t* delays,
                                std::size_t ndm, std::size_t nsamples,
                                std::size_t nchans, std::size_t chan_begin,
                                std::size_t channel_count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = ndm * nsamples;
  if (index >= total) {
    return;
  }
  const std::size_t dm_index = index / nsamples;
  const std::size_t time = index % nsamples;
  const std::int32_t* dm_delays = delays + dm_index * channel_count;

  OutT sum = 0;
  for (std::size_t offset = 0; offset < channel_count; ++offset) {
    const std::size_t channel = chan_begin + offset;
    const std::int64_t shifted_time =
        static_cast<std::int64_t>(time) + dm_delays[offset];
    if (shifted_time >= 0 &&
        shifted_time < static_cast<std::int64_t>(nsamples)) {
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
    std::size_t subband_channels, std::size_t nsamples, std::size_t nchans,
    std::size_t chan_begin, std::size_t chan_end, std::size_t tile_offset,
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
  const std::size_t time = tile_offset + time_in_tile;

  OutT sum = 0;
  for (std::size_t channel = subband_begin; channel < subband_end; ++channel) {
    const std::int64_t shifted_time =
        static_cast<std::int64_t>(time) +
        nominal_delays[channel - chan_begin];
    if (shifted_time >= 0 &&
        shifted_time < static_cast<std::int64_t>(nsamples)) {
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
    std::size_t nsamples, std::size_t tile_offset, std::size_t tile_len,
    std::size_t tile1_len) {
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
  output[dm_index * nsamples + tile_offset + time_in_tile] = sum;
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
  const int threads = static_cast<int>(options.threads_per_block);
  const std::vector<double> frequency_copy = copy_span_to_vector(frequency_mhz);
  const CudaDeviceBuffer<T> device_input = copy_to_device<T>(
      std::span<const T>(samples.data, input_size), "copy samples");
  const CudaDeviceBuffer<double> device_frequency =
      copy_to_device<double>(std::span<const double>(frequency_copy),
                             "copy frequency table");
  CudaDeviceBuffer<std::int32_t> device_delays(channel_count);
  CudaDeviceBuffer<OutT> device_output(samples.shape.nsamples);

  compute_single_dm_delay_kernel<<<block_count(channel_count,
                                                options.threads_per_block),
                                   threads>>>(
      device_delays.get(), device_frequency.get(), plan.dm,
      plan.ref_frequency_mhz, plan.tsamp, plan.chan_begin, channel_count);
  launch_barrier("compute_single_dm_delay_kernel");

  single_dm_kernel<T, OutT>
      <<<block_count(samples.shape.nsamples, options.threads_per_block),
         threads>>>(device_output.get(), device_input.get(),
                    device_delays.get(), samples.shape.nsamples,
                    samples.shape.nchans, plan.chan_begin, channel_count);
  launch_barrier("single_dm_kernel");

  CudaDedispersedResult<OutT> result;
  result.shape = DedispersedShape{1, samples.shape.nsamples};
  result.data = std::move(device_output);
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
  const std::size_t output_size =
      checked_multiply(plan.ndm, samples.shape.nsamples, "dedispersed output");
  const int threads = static_cast<int>(options.threads_per_block);
  const std::vector<double> frequency_copy = copy_span_to_vector(frequency_mhz);
  const CudaDeviceBuffer<T> device_input = copy_to_device<T>(
      std::span<const T>(samples.data, input_size), "copy samples");
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
          plan.ndm, samples.shape.nsamples, samples.shape.nchans,
          plan.chan_begin, channel_count);
  launch_barrier("multi_dm_kernel");

  CudaDedispersedResult<OutT> result;
  result.shape = DedispersedShape{plan.ndm, samples.shape.nsamples};
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
  const std::size_t output_size =
      checked_multiply(plan.ndm, samples.shape.nsamples, "dedispersed output");
  const int threads = static_cast<int>(cuda_options.threads_per_block);

  const std::vector<double> frequency_copy = copy_span_to_vector(frequency_mhz);
  const std::vector<double> subband_frequency = make_subband_frequency(
      frequency_mhz, plan, subband_options, subband_count);
  const CudaDeviceBuffer<T> device_input = copy_to_device<T>(
      std::span<const T>(samples.data, input_size), "copy samples");
  const CudaDeviceBuffer<double> device_frequency =
      copy_to_device<double>(std::span<const double>(frequency_copy),
                             "copy frequency table");
  const CudaDeviceBuffer<double> device_subband_frequency =
      copy_to_device<double>(std::span<const double>(subband_frequency),
                             "copy subband frequency table");
  CudaDeviceBuffer<std::int32_t> device_coarse_delays(coarse_delay_size);
  CudaDeviceBuffer<std::int32_t> device_residual_delays(residual_delay_size);

  compute_subband_coarse_delay_kernel<<<
      block_count(coarse_delay_size, cuda_options.threads_per_block),
      threads>>>(device_coarse_delays.get(), device_frequency.get(),
                 plan.dm_low, plan.dm_step, subband_options.ndm_per_nominal,
                 nominal_dm_count, plan.ref_frequency_mhz, plan.tsamp,
                 plan.chan_begin, channel_count);
  launch_barrier("compute_subband_coarse_delay_kernel");

  compute_subband_residual_delay_kernel<<<
      block_count(residual_delay_size, cuda_options.threads_per_block),
      threads>>>(device_residual_delays.get(), device_subband_frequency.get(),
                 plan.dm_low, plan.dm_step, plan.ndm,
                 subband_options.ndm_per_nominal, subband_count,
                 plan.ref_frequency_mhz, plan.tsamp);
  launch_barrier("compute_subband_residual_delay_kernel");

  std::vector<std::int32_t> residual_delays(residual_delay_size);
  check_cuda(cudaMemcpy(residual_delays.data(), device_residual_delays.get(),
                        residual_delays.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "copy residual delays");
  const std::int32_t max_residual =
      *std::max_element(residual_delays.begin(), residual_delays.end());
  if (max_residual < 0) {
    throw std::invalid_argument(
        "CUDA subband dedispersion requires non-negative residual delays");
  }

  CudaDeviceBuffer<OutT> device_output(output_size);
  const std::size_t max_tile1_len =
      std::min(cuda_options.time_tile_samples +
                   static_cast<std::size_t>(max_residual),
               samples.shape.nsamples);
  const std::size_t intermediate_size =
      checked_multiply(checked_multiply(nominal_dm_count, subband_count,
                                        "subband intermediate"),
                       max_tile1_len, "subband intermediate");
  CudaDeviceBuffer<OutT> device_intermediate(intermediate_size);

  for (std::size_t tile_offset = 0; tile_offset < samples.shape.nsamples;
       tile_offset += cuda_options.time_tile_samples) {
    const std::size_t tile_len =
        std::min(cuda_options.time_tile_samples,
                 samples.shape.nsamples - tile_offset);
    const std::size_t tile1_len =
        std::min(tile_len + static_cast<std::size_t>(max_residual),
                 samples.shape.nsamples - tile_offset);

    const dim3 stage1_grid(
        static_cast<unsigned int>(
            block_count(tile1_len, cuda_options.threads_per_block)),
        static_cast<unsigned int>(nominal_dm_count),
        static_cast<unsigned int>(subband_count));
    subband_stage1_kernel<T, OutT>
        <<<stage1_grid, threads>>>(device_intermediate.get(),
                                   device_input.get(),
                                   device_coarse_delays.get(),
                                   nominal_dm_count, subband_count,
                                   subband_options.subband_channels,
                                   samples.shape.nsamples,
                                   samples.shape.nchans, plan.chan_begin,
                                   plan.chan_end, tile_offset, tile1_len,
                                   channel_count);
    launch_barrier("subband_stage1_kernel");

    const dim3 stage2_grid(
        static_cast<unsigned int>(
            block_count(tile_len, cuda_options.threads_per_block)),
        static_cast<unsigned int>(plan.ndm));
    subband_stage2_kernel<OutT>
        <<<stage2_grid, threads>>>(device_output.get(),
                                   device_intermediate.get(),
                                   device_residual_delays.get(), plan.ndm,
                                   subband_options.ndm_per_nominal,
                                   subband_count, samples.shape.nsamples,
                                   tile_offset, tile_len, tile1_len);
    launch_barrier("subband_stage2_kernel");
  }

  CudaDedispersedResult<OutT> result;
  result.shape = DedispersedShape{plan.ndm, samples.shape.nsamples};
  result.data = std::move(device_output);
  result.device_id = cuda_options.device_id;
  return result;
}

}  // namespace

CudaDeviceMemory::CudaDeviceMemory(std::size_t bytes) : bytes_(bytes) {
  if (bytes_ != 0) {
    check_cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
  }
}

CudaDeviceMemory::CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_) {
  other.data_ = nullptr;
  other.bytes_ = 0;
}

CudaDeviceMemory& CudaDeviceMemory::operator=(
    CudaDeviceMemory&& other) noexcept {
  if (this != &other) {
    release();
    data_ = other.data_;
    bytes_ = other.bytes_;
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

CudaDeviceMemory::~CudaDeviceMemory() {
  release();
}

void* CudaDeviceMemory::data() noexcept {
  return data_;
}

const void* CudaDeviceMemory::data() const noexcept {
  return data_;
}

std::size_t CudaDeviceMemory::bytes() const noexcept {
  return bytes_;
}

void CudaDeviceMemory::release() noexcept {
  if (data_ != nullptr) {
    (void)cudaFree(data_);
    data_ = nullptr;
    bytes_ = 0;
  }
}

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

DedispersedResult<std::uint32_t> dedisperse_single_dm_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return copy_to_host(
      dedisperse_single_dm_cuda_device(samples, frequency_mhz, plan, options));
}

DedispersedResult<std::uint32_t> dedisperse_single_dm_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return copy_to_host(
      dedisperse_single_dm_cuda_device(samples, frequency_mhz, plan, options));
}

DedispersedResult<float> dedisperse_single_dm_cuda(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return copy_to_host(
      dedisperse_single_dm_cuda_device(samples, frequency_mhz, plan, options));
}

DedispersedResult<std::uint32_t> dedisperse_multi_dm_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return copy_to_host(
      dedisperse_multi_dm_cuda_device(samples, frequency_mhz, plan, options));
}

DedispersedResult<std::uint32_t> dedisperse_multi_dm_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return copy_to_host(
      dedisperse_multi_dm_cuda_device(samples, frequency_mhz, plan, options));
}

DedispersedResult<float> dedisperse_multi_dm_cuda(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options) {
  return copy_to_host(
      dedisperse_multi_dm_cuda_device(samples, frequency_mhz, plan, options));
}

DedispersedResult<std::uint32_t> dedisperse_subband_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  return copy_to_host(dedisperse_subband_cuda_device(
      samples, frequency_mhz, plan, subband_options, cuda_options));
}

DedispersedResult<std::uint32_t> dedisperse_subband_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  return copy_to_host(dedisperse_subband_cuda_device(
      samples, frequency_mhz, plan, subband_options, cuda_options));
}

DedispersedResult<float> dedisperse_subband_cuda(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const CudaDedispersionOptions& cuda_options) {
  return copy_to_host(dedisperse_subband_cuda_device(
      samples, frequency_mhz, plan, subband_options, cuda_options));
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
