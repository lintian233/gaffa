#pragma once

#include "gaffa/dedispersion.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace gaffa::internal {

inline constexpr double dispersion_delay_ms = 4148.808;

inline std::size_t checked_delay_table_size(std::size_t ndm,
                                            std::size_t nchans) {
  if (ndm != 0 && nchans > std::numeric_limits<std::size_t>::max() / ndm) {
    throw std::overflow_error("dedispersion delay table size overflow");
  }
  return ndm * nchans;
}

inline void validate_nonnegative_delay_range(
    std::span<const double> frequency_mhz, double ref_frequency_mhz,
    std::size_t chan_begin, std::size_t chan_end) {
  for (std::size_t channel = chan_begin; channel < chan_end; ++channel) {
    if (frequency_mhz[channel] > ref_frequency_mhz) {
      throw std::invalid_argument(
          "dedispersion requires selected frequencies <= reference frequency");
    }
  }
}

inline std::int32_t dedispersion_delay_bins(double dm, double frequency_mhz,
                                            double ref_frequency_mhz,
                                            double tsamp) {
  const double frequency2 = frequency_mhz * frequency_mhz;
  const double ref2 = ref_frequency_mhz * ref_frequency_mhz;
  const double delay =
      dispersion_delay_ms * dm * (1.0 / frequency2 - 1.0 / ref2);
  const double bins = std::round(delay / tsamp);
  if (bins < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
      bins > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("dedispersion delay exceeds int32 range");
  }
  return static_cast<std::int32_t>(bins);
}

inline std::vector<std::int32_t> make_single_dm_delay_table(
    std::span<const double> frequency_mhz, double dm,
    double ref_frequency_mhz, double tsamp, std::size_t chan_begin,
    std::size_t chan_end) {
  std::vector<std::int32_t> delays(chan_end - chan_begin);
  for (std::size_t channel = chan_begin; channel < chan_end; ++channel) {
    delays[channel - chan_begin] = dedispersion_delay_bins(
        dm, frequency_mhz[channel], ref_frequency_mhz, tsamp);
  }
  return delays;
}

inline std::vector<std::int32_t> make_multi_dm_delay_table(
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan) {
  const std::size_t channel_count = plan.chan_end - plan.chan_begin;
  std::vector<std::int32_t> delays(
      checked_delay_table_size(plan.ndm, channel_count));
  for (std::size_t dm_index = 0; dm_index < plan.ndm; ++dm_index) {
    const double dm = plan.dm_low + static_cast<double>(dm_index) * plan.dm_step;
    for (std::size_t offset = 0; offset < channel_count; ++offset) {
      const std::size_t channel = plan.chan_begin + offset;
      delays[dm_index * channel_count + offset] =
          dedispersion_delay_bins(dm, frequency_mhz[channel],
                                  plan.ref_frequency_mhz, plan.tsamp);
    }
  }
  return delays;
}

inline std::int32_t max_nonnegative_delay(
    std::span<const std::int32_t> delays) {
  if (delays.empty()) {
    throw std::invalid_argument("dedispersion delay table must not be empty");
  }
  const auto [min_delay, max_delay] =
      std::minmax_element(delays.begin(), delays.end());
  if (*min_delay < 0) {
    throw std::invalid_argument("dedispersion requires non-negative delays");
  }
  return *max_delay;
}

inline std::size_t valid_output_nsamples(std::size_t input_nsamples,
                                         std::int32_t max_delay) {
  if (max_delay < 0) {
    throw std::invalid_argument("dedispersion requires non-negative delays");
  }
  const auto delay = static_cast<std::size_t>(max_delay);
  if (delay >= input_nsamples) {
    throw std::invalid_argument("dedispersion valid output range is empty");
  }
  return input_nsamples - delay;
}

inline std::size_t valid_output_nsamples_from_delays(
    std::size_t input_nsamples, std::span<const std::int32_t> delays) {
  return valid_output_nsamples(input_nsamples, max_nonnegative_delay(delays));
}

inline double min_selected_frequency(std::span<const double> frequency_mhz,
                                     std::size_t chan_begin,
                                     std::size_t chan_end) {
  return *std::min_element(frequency_mhz.begin() +
                               static_cast<std::ptrdiff_t>(chan_begin),
                           frequency_mhz.begin() +
                               static_cast<std::ptrdiff_t>(chan_end));
}

inline std::int32_t single_dm_max_delay(
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan) {
  const double min_frequency =
      min_selected_frequency(frequency_mhz, plan.chan_begin, plan.chan_end);
  return dedispersion_delay_bins(plan.dm, min_frequency,
                                 plan.ref_frequency_mhz, plan.tsamp);
}

inline std::int32_t multi_dm_max_delay(
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan) {
  const double max_dm =
      plan.dm_low + static_cast<double>(plan.ndm - 1) * plan.dm_step;
  const double min_frequency =
      min_selected_frequency(frequency_mhz, plan.chan_begin, plan.chan_end);
  return dedispersion_delay_bins(max_dm, min_frequency,
                                 plan.ref_frequency_mhz, plan.tsamp);
}

}  // namespace gaffa::internal
