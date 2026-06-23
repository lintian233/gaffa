#include "gaffa/time_series.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gaffa {
namespace {

void validate_downsample_factor(std::size_t nsamples, double factor) {
  if (nsamples == 0) {
    throw std::invalid_argument("time series must not be empty");
  }
  if (!std::isfinite(factor) || factor <= 1.0 ||
      factor > static_cast<double>(nsamples)) {
    throw std::invalid_argument(
        "downsample factor must satisfy 1 < factor <= nsamples");
  }
}

}  // namespace

std::size_t downsampled_size(std::size_t nsamples, double factor) {
  validate_downsample_factor(nsamples, factor);
  return static_cast<std::size_t>(
      std::floor(static_cast<double>(nsamples) / factor));
}

double downsampled_variance(std::size_t nsamples, double factor) {
  validate_downsample_factor(nsamples, factor);

  const double whole = std::floor(factor);
  const double fractional = factor - whole;
  const double fractional_span =
      static_cast<double>(downsampled_size(nsamples, factor)) * fractional;

  if (fractional_span > 1.0) {
    return factor - 1.0 / 3.0;
  }

  return std::pow(whole - 1.0, 2.0) +
         (2.0 / 3.0) * std::pow(fractional_span, 2.0) - fractional_span + 1.0;
}

void downsample_weighted_sum_cpu(std::span<const float> input,
                                 double factor,
                                 std::span<float> output) {
  const std::size_t expected_size = downsampled_size(input.size(), factor);
  if (output.size() != expected_size) {
    throw std::invalid_argument(
        "downsample output size must match downsampled_size");
  }

  const double last_input_index = static_cast<double>(input.size() - 1);
  for (std::size_t output_index = 0; output_index < output.size();
       ++output_index) {
    const double start = static_cast<double>(output_index) * factor;
    const double end = start + factor;

    const auto first_input =
        static_cast<std::size_t>(std::floor(start));
    const auto last_input = static_cast<std::size_t>(
        std::min(std::floor(end), last_input_index));

    const auto first_weight =
        static_cast<float>(static_cast<double>(first_input + 1) - start);
    const auto last_weight =
        static_cast<float>(end - static_cast<double>(last_input));

    float sum = first_weight * input[first_input];
    for (std::size_t input_index = first_input + 1; input_index < last_input;
         ++input_index) {
      sum += input[input_index];
    }
    sum += last_weight * input[last_input];

    output[output_index] = sum;
  }
}

std::vector<float> downsample_weighted_sum_cpu(std::span<const float> input,
                                               double factor) {
  std::vector<float> output(downsampled_size(input.size(), factor));
  downsample_weighted_sum_cpu(input, factor, output);
  return output;
}

}  // namespace gaffa
