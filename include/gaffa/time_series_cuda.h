#pragma once

#include "gaffa/cuda_memory.h"
#include "gaffa/launch_cuda.h"

#include <cstddef>
#include <cstdint>

namespace gaffa {

struct CudaTimeSeriesBatchView {
  const float* data = nullptr;
  std::size_t nseries = 0;
  std::size_t nsamples = 0;
  int device_id = 0;

  [[nodiscard]] std::size_t size() const noexcept {
    return nseries * nsamples;
  }
};

// Mutable counterpart to CudaTimeSeriesBatchView. CUDA preprocessing uses this
// view to make its in-place data ownership contract explicit.
struct MutableCudaTimeSeriesBatchView {
  float* data = nullptr;
  std::size_t nseries = 0;
  std::size_t nsamples = 0;
  int device_id = 0;

  [[nodiscard]] std::size_t size() const noexcept {
    return nseries * nsamples;
  }

  [[nodiscard]] CudaTimeSeriesBatchView as_const() const noexcept {
    return CudaTimeSeriesBatchView{
        .data = data,
        .nseries = nseries,
        .nsamples = nsamples,
        .device_id = device_id,
    };
  }
};

// CUDA batch equivalent of downsample_weighted_sum_cpu(). Each input series is
// downsampled independently. The output size must be
// input.nseries * downsampled_size(input.nsamples, factor).
void downsample_weighted_sum_cuda(CudaTimeSeriesBatchView input,
                                  double factor,
                                  CudaSpan<float> output,
                                  const CudaLaunchOptions& options = {});

// Converts a dense device-resident [series][sample] uint32 batch to float.
// Input and output must have identical element counts and share the launch
// device. This is the explicit handoff between integer-valued algorithms such
// as incoherent dedispersion and floating-point time-series processing.
void convert_time_series_batch_to_float_cuda(
    CudaSpan<const std::uint32_t> input,
    std::size_t nseries,
    std::size_t nsamples,
    CudaSpan<float> output,
    const CudaLaunchOptions& options = {});

}  // namespace gaffa
