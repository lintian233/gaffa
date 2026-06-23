#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

std::size_t downsampled_size(std::size_t nsamples, double factor);

double downsampled_variance(std::size_t nsamples, double factor);

void downsample_weighted_sum_cpu(std::span<const float> input,
                                 double factor,
                                 std::span<float> output);

std::vector<float> downsample_weighted_sum_cpu(std::span<const float> input,
                                               double factor);

}  // namespace gaffa
