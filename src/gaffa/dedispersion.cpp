#include "gaffa/dedispersion.h"
#include "gaffa/internal/dedispersion_delay.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
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

std::size_t checked_output_size(std::size_t ndm, std::size_t nsamples) {
  if (ndm != 0 && nsamples > std::numeric_limits<std::size_t>::max() / ndm) {
    throw std::overflow_error("dedispersed output size overflow");
  }
  return ndm * nsamples;
}

std::size_t ceil_div(std::size_t value, std::size_t divisor) {
  return value / divisor + static_cast<std::size_t>(value % divisor != 0);
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

template <typename T>
void validate_single_plan(HostSampleView<T> samples,
                          std::span<const double> frequency_mhz,
                          const SingleDmDedispersionPlan& plan) {
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
                         const MultiDmDedispersionPlan& plan) {
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

std::vector<std::int32_t> make_subband_coarse_delay_table(
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options,
    std::size_t nominal_dm_count) {
  const std::size_t channel_count = plan.chan_end - plan.chan_begin;
  std::vector<std::int32_t> delays(
      checked_output_size(nominal_dm_count, channel_count));
  for (std::size_t nominal_index = 0; nominal_index < nominal_dm_count;
       ++nominal_index) {
    const double nominal_dm =
        plan.dm_low +
        static_cast<double>(nominal_index * options.ndm_per_nominal) *
            plan.dm_step;
    for (std::size_t offset = 0; offset < channel_count; ++offset) {
      const std::size_t channel = plan.chan_begin + offset;
      delays[nominal_index * channel_count + offset] =
          internal::dedispersion_delay_bins(
              nominal_dm, frequency_mhz[channel], plan.ref_frequency_mhz,
              plan.tsamp);
    }
  }
  return delays;
}

std::vector<std::int32_t> make_subband_residual_delay_table(
    std::span<const double> subband_frequency,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options) {
  std::vector<std::int32_t> delays(
      checked_output_size(plan.ndm, subband_frequency.size()));
  for (std::size_t dm_index = 0; dm_index < plan.ndm; ++dm_index) {
    const std::size_t nominal_index = dm_index / options.ndm_per_nominal;
    const double nominal_dm =
        plan.dm_low +
        static_cast<double>(nominal_index * options.ndm_per_nominal) *
            plan.dm_step;
    const double final_dm =
        plan.dm_low + static_cast<double>(dm_index) * plan.dm_step;
    for (std::size_t subband = 0; subband < subband_frequency.size();
         ++subband) {
      delays[dm_index * subband_frequency.size() + subband] =
          internal::dedispersion_delay_bins(
              final_dm - nominal_dm, subband_frequency[subband],
              plan.ref_frequency_mhz, plan.tsamp);
    }
  }
  return delays;
}

template <typename T>
const T& sample_at(HostSampleView<T> samples, std::size_t time,
                   std::size_t channel) {
  return samples.data[time * samples.shape.nchans + channel];
}

template <typename T>
DedispersedValueT<T> direct_sum_at(HostSampleView<T> samples,
                                   std::span<const std::int32_t> delays,
                                   std::size_t chan_begin,
                                   std::size_t time) {
  using OutT = DedispersedValueT<T>;
  OutT sum = 0;
  for (std::size_t offset = 0; offset < delays.size(); ++offset) {
    const std::size_t channel = chan_begin + offset;
    const std::int64_t shifted_time =
        static_cast<std::int64_t>(time) + delays[offset];
    if (shifted_time >= 0 &&
        shifted_time < static_cast<std::int64_t>(samples.shape.nsamples)) {
      sum += static_cast<OutT>(
          sample_at(samples, static_cast<std::size_t>(shifted_time), channel));
    }
  }
  return sum;
}

template <typename T>
DedispersedResult<DedispersedValueT<T>> dedisperse_single_dm_cpu_impl(
    HostSampleView<T> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan) {
  validate_single_plan(samples, frequency_mhz, plan);

  using OutT = DedispersedValueT<T>;
  const std::vector<std::int32_t> delays = internal::make_single_dm_delay_table(
      frequency_mhz, plan.dm, plan.ref_frequency_mhz, plan.tsamp,
      plan.chan_begin, plan.chan_end);
  const std::size_t output_nsamples =
      internal::valid_output_nsamples(
          samples.shape.nsamples, internal::single_dm_max_delay(frequency_mhz,
                                                                plan));
  DedispersedResult<OutT> result;
  result.shape = DedispersedShape{1, output_nsamples};
  result.data.resize(output_nsamples);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t time = 0;
       time < static_cast<std::int64_t>(output_nsamples); ++time) {
    result.data[static_cast<std::size_t>(time)] =
        direct_sum_at(samples, std::span<const std::int32_t>(delays),
                      plan.chan_begin,
                      static_cast<std::size_t>(time));
  }

  return result;
}

template <typename T>
DedispersedResult<DedispersedValueT<T>> dedisperse_multi_dm_cpu_impl(
    HostSampleView<T> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan) {
  validate_multi_plan(samples, frequency_mhz, plan);

  using OutT = DedispersedValueT<T>;
  const std::size_t channel_count = plan.chan_end - plan.chan_begin;
  const std::vector<std::int32_t> delays =
      internal::make_multi_dm_delay_table(frequency_mhz, plan);
  const std::size_t output_nsamples =
      internal::valid_output_nsamples(
          samples.shape.nsamples, internal::multi_dm_max_delay(frequency_mhz,
                                                               plan));
  DedispersedResult<OutT> result;
  result.shape = DedispersedShape{plan.ndm, output_nsamples};
  result.data.resize(checked_output_size(plan.ndm, output_nsamples));

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::int64_t dm_index = 0;
       dm_index < static_cast<std::int64_t>(plan.ndm); ++dm_index) {
    for (std::int64_t time = 0;
         time < static_cast<std::int64_t>(output_nsamples); ++time) {
      const auto delay_offset =
          static_cast<std::size_t>(dm_index) * channel_count;
      const std::span<const std::int32_t> dm_delays(delays.data() + delay_offset,
                                                    channel_count);
      result.data[static_cast<std::size_t>(dm_index) *
                      output_nsamples +
                  static_cast<std::size_t>(time)] =
          direct_sum_at(samples, dm_delays, plan.chan_begin,
                        static_cast<std::size_t>(time));
    }
  }

  return result;
}

template <typename T>
DedispersedValueT<T> subband_stage1_sum(
    HostSampleView<T> samples, std::span<const std::int32_t> delays,
    const MultiDmDedispersionPlan& plan,
    std::size_t subband_begin, std::size_t subband_end, std::size_t time) {
  using OutT = DedispersedValueT<T>;
  OutT sum = 0;
  for (std::size_t channel = subband_begin; channel < subband_end; ++channel) {
    const std::int64_t shifted_time =
        static_cast<std::int64_t>(time) +
        delays[channel - plan.chan_begin];
    if (shifted_time >= 0 &&
        shifted_time < static_cast<std::int64_t>(samples.shape.nsamples)) {
      sum += static_cast<OutT>(
          sample_at(samples, static_cast<std::size_t>(shifted_time), channel));
    }
  }
  return sum;
}

template <typename T>
DedispersedResult<DedispersedValueT<T>> dedisperse_subband_cpu_impl(
    HostSampleView<T> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options) {
  validate_multi_plan(samples, frequency_mhz, plan);
  validate_subband_options(options);

  using OutT = DedispersedValueT<T>;
  const std::size_t channel_count = plan.chan_end - plan.chan_begin;
  const std::size_t subband_count =
      ceil_div(channel_count, options.subband_channels);
  const std::size_t nominal_dm_count =
      ceil_div(plan.ndm, options.ndm_per_nominal);

  std::vector<std::size_t> subband_begin(subband_count);
  std::vector<std::size_t> subband_end(subband_count);
  std::vector<double> subband_frequency(subband_count);
  for (std::size_t subband = 0; subband < subband_count; ++subband) {
    const std::size_t begin =
        plan.chan_begin + subband * options.subband_channels;
    const std::size_t end =
        std::min(begin + options.subband_channels, plan.chan_end);
    subband_begin[subband] = begin;
    subband_end[subband] = end;
    subband_frequency[subband] = frequency_mhz[(begin + end - 1) / 2];
  }

  const std::vector<std::int32_t> coarse_delays =
      make_subband_coarse_delay_table(frequency_mhz, plan, options,
                                      nominal_dm_count);
  const std::vector<std::int32_t> residual_delays =
      make_subband_residual_delay_table(
          std::span<const double>(subband_frequency), plan, options);
  const std::size_t output_nsamples =
      internal::valid_output_nsamples(
          samples.shape.nsamples, internal::multi_dm_max_delay(frequency_mhz,
                                                               plan));
  const std::size_t inter_stride =
      checked_output_size(subband_count, samples.shape.nsamples);
  std::vector<OutT> intermediate(
      checked_output_size(nominal_dm_count, inter_stride));

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (std::int64_t nominal_index = 0;
       nominal_index < static_cast<std::int64_t>(nominal_dm_count);
       ++nominal_index) {
    for (std::int64_t subband = 0;
         subband < static_cast<std::int64_t>(subband_count); ++subband) {
      for (std::int64_t time = 0;
           time < static_cast<std::int64_t>(samples.shape.nsamples); ++time) {
        const auto coarse_delay_offset =
            static_cast<std::size_t>(nominal_index) * channel_count;
        const std::span<const std::int32_t> nominal_delays(
            coarse_delays.data() + coarse_delay_offset, channel_count);
        intermediate[static_cast<std::size_t>(nominal_index) * inter_stride +
                     static_cast<std::size_t>(subband) *
                         samples.shape.nsamples +
                     static_cast<std::size_t>(time)] =
            subband_stage1_sum(samples, nominal_delays, plan,
                               subband_begin[static_cast<std::size_t>(subband)],
                               subband_end[static_cast<std::size_t>(subband)],
                               static_cast<std::size_t>(time));
      }
    }
  }

  DedispersedResult<OutT> result;
  result.shape = DedispersedShape{plan.ndm, output_nsamples};
  result.data.resize(checked_output_size(plan.ndm, output_nsamples));

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::int64_t dm_index = 0;
       dm_index < static_cast<std::int64_t>(plan.ndm); ++dm_index) {
    for (std::int64_t time = 0;
         time < static_cast<std::int64_t>(output_nsamples); ++time) {
      const auto dm_index_size = static_cast<std::size_t>(dm_index);
      const auto nominal_index = dm_index_size / options.ndm_per_nominal;
      const auto residual_offset = dm_index_size * subband_count;

      OutT sum = 0;
      for (std::size_t subband = 0; subband < subband_count; ++subband) {
        const std::int64_t shifted_time =
            static_cast<std::int64_t>(time) +
            residual_delays[residual_offset + subband];
        if (shifted_time >= 0 &&
            shifted_time < static_cast<std::int64_t>(samples.shape.nsamples)) {
          sum += intermediate[nominal_index * inter_stride +
                              subband * samples.shape.nsamples +
                              static_cast<std::size_t>(shifted_time)];
        }
      }
      result.data[dm_index_size * output_nsamples +
                  static_cast<std::size_t>(time)] = sum;
    }
  }

  return result;
}

}  // namespace

DedispersedResult<std::uint32_t> dedisperse_single_dm_cpu(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan) {
  return dedisperse_single_dm_cpu_impl(samples, frequency_mhz, plan);
}

DedispersedResult<std::uint32_t> dedisperse_single_dm_cpu(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan) {
  return dedisperse_single_dm_cpu_impl(samples, frequency_mhz, plan);
}

DedispersedResult<float> dedisperse_single_dm_cpu(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan) {
  return dedisperse_single_dm_cpu_impl(samples, frequency_mhz, plan);
}

DedispersedResult<std::uint32_t> dedisperse_multi_dm_cpu(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan) {
  return dedisperse_multi_dm_cpu_impl(samples, frequency_mhz, plan);
}

DedispersedResult<std::uint32_t> dedisperse_multi_dm_cpu(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan) {
  return dedisperse_multi_dm_cpu_impl(samples, frequency_mhz, plan);
}

DedispersedResult<float> dedisperse_multi_dm_cpu(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan) {
  return dedisperse_multi_dm_cpu_impl(samples, frequency_mhz, plan);
}

DedispersedResult<std::uint32_t> dedisperse_subband_cpu(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options) {
  return dedisperse_subband_cpu_impl(samples, frequency_mhz, plan, options);
}

DedispersedResult<std::uint32_t> dedisperse_subband_cpu(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options) {
  return dedisperse_subband_cpu_impl(samples, frequency_mhz, plan, options);
}

DedispersedResult<float> dedisperse_subband_cpu(
    HostSampleView<float> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options) {
  return dedisperse_subband_cpu_impl(samples, frequency_mhz, plan, options);
}

}  // namespace gaffa
