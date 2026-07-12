#pragma once

#include "gaffa/cuda_memory.h"
#include "gaffa/launch_cuda.h"

#include <cstddef>

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

// CUDA batch equivalent of downsample_weighted_sum_cpu(). Each input series is
// downsampled independently. The output size must be
// input.nseries * downsampled_size(input.nsamples, factor).
void downsample_weighted_sum_cuda(CudaTimeSeriesBatchView input,
                                  double factor,
                                  CudaSpan<float> output,
                                  const CudaLaunchOptions& options = {});

}  // namespace gaffa
