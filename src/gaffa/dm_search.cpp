#include "gaffa/dm_search.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gaffa {
namespace {

std::size_t checked_size(DedispersedShape shape) {
  if (shape.ndm == 0) {
    throw std::invalid_argument("DM search dedispersed ndm must be > 0");
  }
  if (shape.nsamples == 0) {
    throw std::invalid_argument("DM search dedispersed nsamples must be > 0");
  }
  if (shape.ndm > std::numeric_limits<std::size_t>::max() / shape.nsamples) {
    throw std::overflow_error("DM search dedispersed shape size overflow");
  }
  return shape.ndm * shape.nsamples;
}

void validate_tsamp(double tsamp) {
  if (!(tsamp > 0.0) || !std::isfinite(tsamp)) {
    throw std::invalid_argument("DM search tsamp must be finite and > 0");
  }
}

template <typename T>
void validate_dedispersed_result(const DedispersedResult<T>& input) {
  const std::size_t expected_size = checked_size(input.shape);
  if (input.data.size() != expected_size) {
    throw std::invalid_argument(
        "DM search dedispersed data size does not match shape");
  }
}

template <typename T>
std::span<const T> dm_row(const DedispersedResult<T>& input,
                          std::size_t dm_index) {
  validate_dedispersed_result(input);
  if (dm_index >= input.shape.ndm) {
    throw std::out_of_range("DM search dm_index is out of range");
  }
  const std::size_t offset = dm_index * input.shape.nsamples;
  return std::span<const T>(input.data).subspan(offset, input.shape.nsamples);
}

template <typename T>
TimeSeries dm_time_series_impl(const DedispersedResult<T>& input,
                               std::size_t dm_index,
                               double tsamp) {
  validate_tsamp(tsamp);

  const auto row = dm_row(input, dm_index);
  std::vector<float> data(row.size());
  for (std::size_t index = 0; index < row.size(); ++index) {
    data[index] = static_cast<float>(row[index]);
  }
  return TimeSeries{
      .data = std::move(data),
      .tsamp = tsamp,
  };
}

void validate_dm_search_inputs(DedispersedShape shape,
                               std::span<const double> dms,
                               double tsamp,
                               const DmSearchOptions& options) {
  (void)checked_size(shape);
  validate_tsamp(tsamp);
  if (dms.size() != shape.ndm) {
    throw std::invalid_argument("DM search dms size must match ndm");
  }
  for (const double dm : dms) {
    if (!std::isfinite(dm)) {
      throw std::invalid_argument("DM search dms must be finite");
    }
  }
  if (!std::isfinite(options.snr_threshold)) {
    throw std::invalid_argument("DM search S/N threshold must be finite");
  }
  if (options.frequency_cluster_radius < 0.0 ||
      !std::isfinite(options.frequency_cluster_radius)) {
    throw std::invalid_argument(
        "DM search frequency_cluster_radius must be finite and >= 0");
  }
}

bool is_better_dm_peak(const DmPeak& lhs, const DmPeak& rhs) {
  if (is_better_ffa_peak(lhs.peak, rhs.peak)) {
    return true;
  }
  if (is_better_ffa_peak(rhs.peak, lhs.peak)) {
    return false;
  }
  if (lhs.dm_index != rhs.dm_index) {
    return lhs.dm_index < rhs.dm_index;
  }
  return lhs.dm < rhs.dm;
}

TimeSeries maybe_preprocess(TimeSeries input, const PreprocessPlan& plan) {
  if (plan.steps.empty()) {
    return input;
  }
  return preprocess_time_series_cpu(input, plan);
}

void validate_preprocessed_time_series(const TimeSeries& time_series,
                                       DedispersedShape shape,
                                       double tsamp) {
  if (time_series.data.size() != shape.nsamples ||
      time_series.tsamp != tsamp) {
    throw std::logic_error(
        "DM search preprocessing must preserve time-series shape and tsamp");
  }
}

template <typename T>
void collect_dm_peaks(const DedispersedResult<T>& input,
                      std::span<const double> dms,
                      std::size_t dm_index,
                      double tsamp,
                      const PreprocessPlan& preprocess,
                      const FfaSearchPlan& ffa_plan,
                      const FfaSearchOptions& ffa_options,
                      std::vector<DmPeak>& peaks) {
  TimeSeries time_series =
      maybe_preprocess(dm_time_series_impl(input, dm_index, tsamp), preprocess);
  validate_preprocessed_time_series(time_series, input.shape, tsamp);
  const FfaSearchResult search =
      search_ffa_cpu(time_series.data, ffa_plan, ffa_options);
  for (const auto& peak : search.peaks) {
    peaks.push_back(DmPeak{
        .dm = dms[dm_index],
        .dm_index = dm_index,
        .peak = peak,
    });
  }
}

template <typename T>
DmSearchResult find_dm_peaks_impl(const DedispersedResult<T>& input,
                                  std::span<const double> dms,
                                  double tsamp,
                                  const DmSearchOptions& options) {
  validate_dedispersed_result(input);
  validate_dm_search_inputs(input.shape, dms, tsamp, options);

  const FfaSearchOptions ffa_options{
      .snr_threshold = options.snr_threshold,
      .frequency_cluster_radius = options.frequency_cluster_radius,
      .max_peaks = options.max_peaks,
  };
  const FfaSearchPlan ffa_plan =
      make_riptide_ffa_plan(input.shape.nsamples, tsamp, options.plan);

  std::vector<DmPeak> global_peaks;
  std::exception_ptr error;
  std::atomic_bool has_error = false;
  const bool parallel = input.shape.ndm > 4;

#pragma omp parallel if(parallel)
  {
    std::vector<DmPeak> local_peaks;

#pragma omp for schedule(dynamic, 1)
    for (std::size_t dm_index = 0; dm_index < input.shape.ndm; ++dm_index) {
      if (has_error.load(std::memory_order_relaxed)) {
        continue;
      }
      try {
        collect_dm_peaks(input, dms, dm_index, tsamp, options.preprocess,
                         ffa_plan, ffa_options, local_peaks);
      } catch (...) {
        bool expected = false;
        if (has_error.compare_exchange_strong(expected, true,
                                              std::memory_order_relaxed)) {
#pragma omp critical(dm_search_error)
          { error = std::current_exception(); }
        }
      }
    }

#pragma omp critical(dm_search_peaks)
    {
      global_peaks.insert(global_peaks.end(), local_peaks.begin(),
                          local_peaks.end());
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  std::sort(global_peaks.begin(), global_peaks.end(), is_better_dm_peak);
  return DmSearchResult{
      .peaks = std::move(global_peaks),
  };
}

}  // namespace

TimeSeries dm_time_series_cpu(const DedispersedResult<std::uint32_t>& input,
                              std::size_t dm_index,
                              double tsamp) {
  return dm_time_series_impl(input, dm_index, tsamp);
}

TimeSeries dm_time_series_cpu(const DedispersedResult<float>& input,
                              std::size_t dm_index,
                              double tsamp) {
  return dm_time_series_impl(input, dm_index, tsamp);
}

DmSearchResult find_dm_peaks_cpu(const DedispersedResult<std::uint32_t>& input,
                                 std::span<const double> dms,
                                 double tsamp,
                                 const DmSearchOptions& options) {
  return find_dm_peaks_impl(input, dms, tsamp, options);
}

DmSearchResult find_dm_peaks_cpu(const DedispersedResult<float>& input,
                                 std::span<const double> dms,
                                 double tsamp,
                                 const DmSearchOptions& options) {
  return find_dm_peaks_impl(input, dms, tsamp, options);
}

}  // namespace gaffa
