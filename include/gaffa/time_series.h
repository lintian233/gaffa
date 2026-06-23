#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

// Returns floor(nsamples / factor) for riptide-compatible downsampling.
// The factor must satisfy 1 < factor <= nsamples. A factor of 1 means no
// downsampling and should be handled by the caller by reusing the input span.
std::size_t downsampled_size(std::size_t nsamples, double factor);

// Returns the riptide noise variance model for real-valued weighted-sum
// downsampling. This is used for FFA periodogram/SNR normalization and is not
// the same as the simple sum of squared per-output weights for every sample.
// The factor constraint is the same as downsampled_size().
double downsampled_variance(std::size_t nsamples, double factor);

// Downsamples a 1D time series by integrating each output interval
// [k * factor, (k + 1) * factor) over the input sample grid. Boundary samples
// are weighted by fractional overlap and fully covered samples by 1. The result
// is a weighted sum, not a mean.
void downsample_weighted_sum_cpu(std::span<const float> input,
                                 double factor,
                                 std::span<float> output);

std::vector<float> downsample_weighted_sum_cpu(std::span<const float> input,
                                               double factor);

}  // namespace gaffa
