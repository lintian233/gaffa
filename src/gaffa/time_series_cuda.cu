#include "gaffa/time_series_cuda.h"

#include "gaffa/time_series.h"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace gaffa {
namespace {

__global__ void downsample_weighted_sum_float_kernel(
    const float* input,
    std::size_t input_nsamples,
    float* output,
    std::size_t output_nsamples,
    double factor) {
  const std::size_t output_index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t series = blockIdx.y;
  if (output_index >= output_nsamples) {
    return;
  }

  const double start = static_cast<double>(output_index) * factor;
  const double end = start + factor;
  const auto first_input = static_cast<std::size_t>(floor(start));
  const auto last_input = static_cast<std::size_t>(
      fmin(floor(end), static_cast<double>(input_nsamples - 1)));

  const auto first_weight =
      static_cast<float>(static_cast<double>(first_input + 1) - start);
  const auto last_weight =
      static_cast<float>(end - static_cast<double>(last_input));

  const float* series_input = input + series * input_nsamples;
  float sum = first_weight * series_input[first_input];
  for (std::size_t input_index = first_input + 1; input_index < last_input;
       ++input_index) {
    sum += series_input[input_index];
  }
  sum += last_weight * series_input[last_input];

  output[series * output_nsamples + output_index] = sum;
}

std::size_t checked_multiply(std::size_t lhs,
                             std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void validate_launch_options(const CudaLaunchOptions& options) {
  if (options.device_id < 0) {
    throw std::invalid_argument("CUDA launch device_id must be >= 0");
  }
  if (options.threads_per_block == 0) {
    throw std::invalid_argument("CUDA launch threads_per_block must be > 0");
  }
}

unsigned int checked_grid_dim(std::size_t value, const char* message) {
  if (value == 0 ||
      value > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::overflow_error(message);
  }
  return static_cast<unsigned int>(value);
}

void validate_downsample_arguments(CudaTimeSeriesBatchView input,
                                   double factor,
                                   CudaSpan<float> output,
                                   const CudaLaunchOptions& options) {
  if (input.data == nullptr) {
    throw std::invalid_argument(
        "CUDA weighted downsample input data must not be null");
  }
  if (input.nseries == 0) {
    throw std::invalid_argument(
        "CUDA weighted downsample input nseries must be > 0");
  }
  if (input.nsamples == 0) {
    throw std::invalid_argument(
        "CUDA weighted downsample input nsamples must be > 0");
  }
  validate_launch_options(options);
  if (input.device_id != options.device_id ||
      output.device_id != options.device_id) {
    throw std::invalid_argument(
        "CUDA weighted downsample device ids must match launch device_id");
  }
  if (output.data == nullptr) {
    throw std::invalid_argument(
        "CUDA weighted downsample output data must not be null");
  }
  const std::size_t output_nsamples = downsampled_size(input.nsamples, factor);
  const std::size_t expected_output = checked_multiply(
      input.nseries, output_nsamples,
      "CUDA weighted downsample output element count overflow");
  if (output.count != expected_output) {
    throw std::invalid_argument(
        "CUDA weighted downsample output size must match nseries * "
        "downsampled_size");
  }
}

}  // namespace

void downsample_weighted_sum_cuda(CudaTimeSeriesBatchView input,
                                  double factor,
                                  CudaSpan<float> output,
                                  const CudaLaunchOptions& options) {
  validate_downsample_arguments(input, factor, output, options);
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");

  const std::size_t output_nsamples = downsampled_size(input.nsamples, factor);
  const auto threads = static_cast<unsigned int>(options.threads_per_block);
  const std::size_t x_blocks = (output_nsamples + threads - 1) / threads;
  const dim3 block(threads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA weighted downsample grid x overflow"),
      checked_grid_dim(input.nseries,
                       "CUDA weighted downsample grid y overflow"));
  downsample_weighted_sum_float_kernel<<<grid, block, 0, options.stream>>>(
      input.data, input.nsamples, output.data, output_nsamples, factor);
  check_cuda(cudaGetLastError(), "downsample_weighted_sum_cuda kernel launch");
  if (options.synchronize_after_call) {
    check_cuda(cudaStreamSynchronize(options.stream),
               "downsample_weighted_sum_cuda synchronize");
  }
}

}  // namespace gaffa
