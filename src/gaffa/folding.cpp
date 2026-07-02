#include "gaffa/folding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gaffa {
namespace {

constexpr std::size_t kOpenmpMinFoldSubints = 4;

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs,
                        const char* message) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::overflow_error(message);
  }
  return lhs + rhs;
}

std::size_t checked_cube_size(std::size_t nsubint, std::size_t nchans,
                              std::size_t nbin) {
  return checked_multiply(checked_multiply(nsubint, nchans,
                                           "folded cube size overflow"),
                          nbin, "folded cube size overflow");
}

std::size_t ceil_div(std::size_t value, std::size_t divisor) {
  return value / divisor + static_cast<std::size_t>(value % divisor != 0);
}

void validate_fold_shape(double period, double tsamp, std::size_t nbin) {
  if (!std::isfinite(period) || !(period > 0.0)) {
    throw std::invalid_argument("fold period must be finite and positive");
  }
  if (!std::isfinite(tsamp) || !(tsamp > 0.0)) {
    throw std::invalid_argument("fold tsamp must be finite and positive");
  }
  if (nbin == 0) {
    throw std::invalid_argument("fold nbin must be positive");
  }
  if (nbin > static_cast<std::size_t>(
                 std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("fold nbin exceeds int64 range");
  }
}

void validate_time_series_input(std::span<const float> input,
                                const FoldOptions& options) {
  validate_fold_shape(options.period, options.tsamp, options.nbin);
  if (input.empty()) {
    throw std::invalid_argument("fold input must not be empty");
  }
  if (!std::all_of(input.begin(), input.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("fold input must contain only finite samples");
  }
}

template <typename T>
void validate_spectrum_input(HostSampleView<T> samples,
                             const FoldSpectrumOptions& options) {
  validate_fold_shape(options.period, options.tsamp, options.nbin);
  if (samples.data == nullptr) {
    throw std::invalid_argument("fold spectrum data must not be null");
  }
  if (samples.shape.nifs != 1) {
    throw std::invalid_argument("fold spectrum requires nifs == 1");
  }
  if (samples.shape.nsamples == 0 || samples.shape.nchans == 0) {
    throw std::invalid_argument("fold spectrum shape must be non-empty");
  }
  if (options.subint_samples == 0) {
    throw std::invalid_argument("fold subint_samples must be positive");
  }
  if (options.output_channels > samples.shape.nchans) {
    throw std::invalid_argument(
        "fold output_channels must be <= input channel count");
  }
  if (options.output_channels != 0 &&
      samples.shape.nchans % options.output_channels != 0) {
    throw std::invalid_argument(
        "fold input channel count must be divisible by output_channels");
  }
}

void validate_spectrum_finite_values(HostSampleView<float> samples) {
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (!std::isfinite(samples.data[index])) {
      throw std::invalid_argument(
          "fold spectrum must contain only finite samples");
    }
  }
}

std::size_t positive_mod(std::int64_t value, std::size_t modulus) {
  const auto signed_modulus = static_cast<std::int64_t>(modulus);
  std::int64_t result = value % signed_modulus;
  if (result < 0) {
    result += signed_modulus;
  }
  return static_cast<std::size_t>(result);
}

void accumulate_interval(double value, double low_scaled, double high_scaled,
                         std::span<double> sums,
                         std::span<double> exposure) {
  if (!(high_scaled > low_scaled) || !std::isfinite(low_scaled) ||
      !std::isfinite(high_scaled)) {
    throw std::invalid_argument("fold phase interval is invalid");
  }
  if (sums.size() != exposure.size() || sums.empty()) {
    throw std::invalid_argument("fold accumulator shape is invalid");
  }

  const double first_bin_double = std::floor(low_scaled);
  const double last_bin_double = std::ceil(high_scaled) - 1.0;
  if (first_bin_double < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      first_bin_double > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      last_bin_double < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      last_bin_double > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("fold phase bin index exceeds int64 range");
  }

  const auto first_bin = static_cast<std::int64_t>(first_bin_double);
  const auto last_bin = static_cast<std::int64_t>(last_bin_double);
  for (std::int64_t bin = first_bin; bin <= last_bin; ++bin) {
    const double bin_left = static_cast<double>(bin);
    const double bin_right = bin_left + 1.0;
    const double weight =
        std::min(high_scaled, bin_right) - std::max(low_scaled, bin_left);
    if (!(weight > 0.0)) {
      continue;
    }
    const std::size_t wrapped_bin = positive_mod(bin, sums.size());
    sums[wrapped_bin] += value * weight;
    exposure[wrapped_bin] += weight;
  }
}

void accumulate_sample(double value, std::size_t sample_index, double bin_step,
                       std::span<double> sums,
                       std::span<double> exposure) {
  const double low_scaled = static_cast<double>(sample_index) * bin_step;
  accumulate_interval(value, low_scaled, low_scaled + bin_step, sums, exposure);
}

std::vector<float> normalized_profile(std::span<const double> sums,
                                      std::span<const double> exposure) {
  if (sums.size() != exposure.size()) {
    throw std::invalid_argument("fold normalization shape mismatch");
  }
  std::vector<float> output(sums.size(), 0.0F);
  for (std::size_t index = 0; index < sums.size(); ++index) {
    if (exposure[index] > 0.0) {
      output[index] = static_cast<float>(sums[index] / exposure[index]);
    }
  }
  return output;
}

std::size_t subint_samples_from_seconds(double tsubint, double tsamp) {
  if (!std::isfinite(tsubint) || !(tsubint > 0.0)) {
    throw std::invalid_argument("fold tsubint must be finite and positive");
  }
  if (!std::isfinite(tsamp) || !(tsamp > 0.0)) {
    throw std::invalid_argument(
        "fold spectrum tsamp must be finite and positive");
  }

  const double rounded = std::round(tsubint / tsamp);
  constexpr auto max_size = static_cast<double>(
      std::numeric_limits<std::size_t>::max());
  if (!std::isfinite(rounded) || rounded > max_size) {
    throw std::invalid_argument("fold subint sample count is too large");
  }
  return std::max<std::size_t>(1, static_cast<std::size_t>(rounded));
}

std::size_t folded_cube_index(const FoldedCube& cube, std::size_t subint,
                              std::size_t channel, std::size_t phase) {
  return (subint * cube.nchans + channel) * cube.nbin + phase;
}

bool should_parallelize_fold(std::size_t nsubint) {
  return nsubint >= kOpenmpMinFoldSubints;
}

void validate_phase_index_range(std::size_t start_sample,
                                std::size_t sample_count,
                                double bin_step) {
  if (sample_count == 0) {
    return;
  }
  const std::size_t last_sample = checked_add(
      start_sample, sample_count - 1, "fold sample index overflow");
  const double low_scaled = static_cast<double>(start_sample) * bin_step;
  const double high_scaled =
      static_cast<double>(last_sample) * bin_step + bin_step;
  if (!std::isfinite(low_scaled) || !std::isfinite(high_scaled)) {
    throw std::invalid_argument("fold phase interval is invalid");
  }

  const double first_bin = std::floor(low_scaled);
  const double last_bin = std::ceil(high_scaled) - 1.0;
  if (first_bin <
          static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      first_bin >
          static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      last_bin <
          static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      last_bin >
          static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("fold phase bin index exceeds int64 range");
  }
}

FoldDedispersedSpectrumResult project_folded_cube(
    FoldedCube cube,
    const FoldDedispersedSpectrumOptions& options,
    double tsamp,
    double actual_tsubint) {
  FoldDedispersedSpectrumResult result;
  result.profile.assign(cube.nbin, 0.0F);
  result.freq_phase.assign(cube.nchans * cube.nbin, 0.0F);
  result.time_phase.assign(cube.nsubint * cube.nbin, 0.0F);
  result.nsubint = cube.nsubint;
  result.nchans = cube.nchans;
  result.nbin = cube.nbin;
  result.period = options.period;
  result.tsamp = tsamp;
  result.tsubint = actual_tsubint;

  std::vector<double> profile_sum(cube.nbin, 0.0);
  std::vector<double> profile_exposure(cube.nbin, 0.0);
  std::vector<double> freq_sum(cube.nchans * cube.nbin, 0.0);
  std::vector<double> freq_exposure(cube.nchans * cube.nbin, 0.0);

  for (std::size_t subint = 0; subint < cube.nsubint; ++subint) {
    for (std::size_t phase = 0; phase < cube.nbin; ++phase) {
      const double exposure = cube.exposure[subint * cube.nbin + phase];
      if (!(exposure > 0.0)) {
        continue;
      }

      double time_sum = 0.0;
      for (std::size_t channel = 0; channel < cube.nchans; ++channel) {
        const float value =
            cube.data[folded_cube_index(cube, subint, channel, phase)];
        time_sum += value;
        profile_sum[phase] += static_cast<double>(value) * exposure;
        profile_exposure[phase] += exposure;

        const std::size_t freq_index = channel * cube.nbin + phase;
        freq_sum[freq_index] += static_cast<double>(value) * exposure;
        freq_exposure[freq_index] += exposure;
      }
      result.time_phase[subint * cube.nbin + phase] =
          static_cast<float>(time_sum / static_cast<double>(cube.nchans));
    }
  }

  for (std::size_t phase = 0; phase < cube.nbin; ++phase) {
    if (profile_exposure[phase] > 0.0) {
      result.profile[phase] = static_cast<float>(
          profile_sum[phase] / profile_exposure[phase]);
    }
  }
  for (std::size_t index = 0; index < freq_sum.size(); ++index) {
    if (freq_exposure[index] > 0.0) {
      result.freq_phase[index] =
          static_cast<float>(freq_sum[index] / freq_exposure[index]);
    }
  }

  result.cube = std::move(cube);
  return result;
}

template <typename T>
FoldedCube fold_spectrum_impl(HostSampleView<T> samples,
                              const FoldSpectrumOptions& options) {
  validate_spectrum_input(samples, options);
  if constexpr (std::is_same_v<T, float>) {
    validate_spectrum_finite_values(samples);
  }

  const std::size_t nsubint =
      ceil_div(samples.shape.nsamples, options.subint_samples);
  const std::size_t nbin = options.nbin;
  const std::size_t input_channels = samples.shape.nchans;
  const std::size_t output_channels =
      options.output_channels == 0 ? input_channels : options.output_channels;
  const std::size_t channel_factor = input_channels / output_channels;
  const std::size_t output_size =
      checked_cube_size(nsubint, output_channels, nbin);
  const std::size_t exposure_size =
      checked_multiply(nsubint, nbin, "folded exposure size overflow");
  if (nsubint > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("fold subint count exceeds int64 range");
  }

  std::vector<double> sums(output_size, 0.0);
  std::vector<double> exposure(exposure_size, 0.0);
  const double bin_step =
      options.tsamp / options.period * static_cast<double>(nbin);
  validate_phase_index_range(options.start_sample, samples.shape.nsamples,
                             bin_step);

#pragma omp parallel for if(should_parallelize_fold(nsubint)) schedule(static)
  for (std::int64_t subint_index = 0;
       subint_index < static_cast<std::int64_t>(nsubint); ++subint_index) {
    const auto subint = static_cast<std::size_t>(subint_index);
    const std::size_t time_begin = subint * options.subint_samples;
    const std::size_t time_end = std::min(samples.shape.nsamples,
                                         time_begin + options.subint_samples);
    std::vector<double> row_values;
    if (channel_factor != 1) {
      row_values.resize(output_channels);
    }

    for (std::size_t time = time_begin; time < time_end; ++time) {
      const std::size_t sample_index = options.start_sample + time;
      const double low_scaled = static_cast<double>(sample_index) * bin_step;
      const double high_scaled = low_scaled + bin_step;
      const auto first_bin = static_cast<std::int64_t>(std::floor(low_scaled));
      const auto last_bin =
          static_cast<std::int64_t>(std::ceil(high_scaled) - 1.0);
      const T* const row = samples.data + time * input_channels;

      if (channel_factor != 1) {
        for (std::size_t output_channel = 0; output_channel < output_channels;
             ++output_channel) {
          double value = 0.0;
          const std::size_t input_begin = output_channel * channel_factor;
          for (std::size_t offset = 0; offset < channel_factor; ++offset) {
            value += static_cast<double>(row[input_begin + offset]);
          }
          row_values[output_channel] =
              value / static_cast<double>(channel_factor);
        }
      }

      for (std::int64_t bin = first_bin; bin <= last_bin; ++bin) {
        const double bin_left = static_cast<double>(bin);
        const double bin_right = bin_left + 1.0;
        const double weight =
            std::min(high_scaled, bin_right) - std::max(low_scaled, bin_left);
        if (!(weight > 0.0)) {
          continue;
        }
        const std::size_t phase = positive_mod(bin, nbin);
        exposure[subint * nbin + phase] += weight;

        for (std::size_t output_channel = 0; output_channel < output_channels;
             ++output_channel) {
          const double value =
              channel_factor == 1
                  ? static_cast<double>(row[output_channel])
                  : row_values[output_channel];
          const std::size_t output_index =
              (subint * output_channels + output_channel) * nbin + phase;
          sums[output_index] += value * weight;
        }
      }
    }
  }

  FoldedCube result;
  result.data.resize(output_size, 0.0F);
  result.exposure = std::move(exposure);
  result.nsubint = nsubint;
  result.nchans = output_channels;
  result.nbin = nbin;

#pragma omp parallel for if(should_parallelize_fold(nsubint)) collapse(2) \
    schedule(static)
  for (std::int64_t subint_index = 0;
       subint_index < static_cast<std::int64_t>(nsubint); ++subint_index) {
    for (std::int64_t channel_index = 0;
         channel_index < static_cast<std::int64_t>(output_channels);
         ++channel_index) {
      const auto subint = static_cast<std::size_t>(subint_index);
      const auto channel = static_cast<std::size_t>(channel_index);
      for (std::size_t phase = 0; phase < nbin; ++phase) {
        const std::size_t exposure_index = subint * nbin + phase;
        const std::size_t output_index =
            (subint * output_channels + channel) * nbin + phase;
        if (result.exposure[exposure_index] > 0.0) {
          result.data[output_index] = static_cast<float>(
              sums[output_index] / result.exposure[exposure_index]);
        }
      }
    }
  }

  return result;
}

template <typename T>
FoldDedispersedSpectrumResult fold_dedispersed_spectrum_view_impl(
    HostSampleView<T> samples,
    double tsamp,
    const FoldDedispersedSpectrumOptions& options) {
  const std::size_t subint_samples =
      subint_samples_from_seconds(options.tsubint, tsamp);
  FoldedCube cube = fold_spectrum_impl<T>(
      samples,
      FoldSpectrumOptions{
          .period = options.period,
          .tsamp = tsamp,
          .nbin = options.nbin,
          .subint_samples = subint_samples,
          .output_channels = options.output_channels,
      });
  return project_folded_cube(std::move(cube), options, tsamp,
                             static_cast<double>(subint_samples) * tsamp);
}

template <typename T>
FoldDedispersedSpectrumResult fold_dedispersed_spectrum_impl(
    const DedispersedSpectrum<T>& spectrum,
    const FoldDedispersedSpectrumOptions& options) {
  if (spectrum.data.size() != sample_element_count(spectrum.shape)) {
    throw std::invalid_argument("fold spectrum data size does not match shape");
  }
  return fold_dedispersed_spectrum_view_impl(spectrum.view(), spectrum.tsamp,
                                             options);
}

}  // namespace

FoldedProfile fold_time_series_cpu(std::span<const float> input,
                                   const FoldOptions& options) {
  validate_time_series_input(input, options);

  std::vector<double> sums(options.nbin, 0.0);
  std::vector<double> exposure(options.nbin, 0.0);
  const double bin_step = options.tsamp / options.period *
                          static_cast<double>(options.nbin);

  for (std::size_t index = 0; index < input.size(); ++index) {
    accumulate_sample(static_cast<double>(input[index]),
                      checked_add(options.start_sample, index,
                                  "fold sample index overflow"),
                      bin_step, sums, exposure);
  }

  return FoldedProfile{
      .profile = normalized_profile(sums, exposure),
      .exposure = std::move(exposure),
      .nbin = options.nbin,
  };
}

FoldedCube fold_spectrum_cpu(HostSampleView<std::uint8_t> samples,
                             const FoldSpectrumOptions& options) {
  return fold_spectrum_impl(samples, options);
}

FoldedCube fold_spectrum_cpu(HostSampleView<std::uint16_t> samples,
                             const FoldSpectrumOptions& options) {
  return fold_spectrum_impl(samples, options);
}

FoldedCube fold_spectrum_cpu(HostSampleView<float> samples,
                             const FoldSpectrumOptions& options) {
  return fold_spectrum_impl(samples, options);
}

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_view_cpu(
    HostSampleView<std::uint8_t> samples,
    double tsamp,
    const FoldDedispersedSpectrumOptions& options) {
  return fold_dedispersed_spectrum_view_impl(samples, tsamp, options);
}

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_view_cpu(
    HostSampleView<std::uint16_t> samples,
    double tsamp,
    const FoldDedispersedSpectrumOptions& options) {
  return fold_dedispersed_spectrum_view_impl(samples, tsamp, options);
}

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_view_cpu(
    HostSampleView<float> samples,
    double tsamp,
    const FoldDedispersedSpectrumOptions& options) {
  return fold_dedispersed_spectrum_view_impl(samples, tsamp, options);
}

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_cpu(
    const DedispersedSpectrum<std::uint8_t>& spectrum,
    const FoldDedispersedSpectrumOptions& options) {
  return fold_dedispersed_spectrum_impl(spectrum, options);
}

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_cpu(
    const DedispersedSpectrum<std::uint16_t>& spectrum,
    const FoldDedispersedSpectrumOptions& options) {
  return fold_dedispersed_spectrum_impl(spectrum, options);
}

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_cpu(
    const DedispersedSpectrum<float>& spectrum,
    const FoldDedispersedSpectrumOptions& options) {
  return fold_dedispersed_spectrum_impl(spectrum, options);
}

}  // namespace gaffa
