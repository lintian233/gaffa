#include "gaffa/dm_search.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
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

template <typename T>
void validate_dedispersed_result(DedispersedResultView<T> input) {
  const std::size_t expected_size = checked_size(input.shape);
  if (input.data.size() != expected_size) {
    throw std::invalid_argument(
        "DM search dedispersed data size does not match shape");
  }
}

template <typename T>
std::span<const T> dm_row(DedispersedResultView<T> input,
                          std::size_t dm_index) {
  const std::size_t offset = dm_index * input.shape.nsamples;
  return input.data.subspan(offset, input.shape.nsamples);
}

void validate_dm_search_inputs(DedispersedShape shape,
                               DmTrialView trials,
                               const FfaSearchPlan& plan,
                               const DmFfaOptions& options) {
  (void)checked_size(shape);
  validate_ffa_search_plan(plan);
  if (plan.observation.nsamples != shape.nsamples) {
    throw std::invalid_argument(
        "DM search plan observation must match dedispersed nsamples");
  }
  if (trials.values.size() != shape.ndm) {
    throw std::invalid_argument("DM trial count must match dedispersed ndm");
  }
  validate_dm_trials(trials);
  if (!std::isfinite(options.search.snr_threshold)) {
    throw std::invalid_argument("DM search S/N threshold must be finite");
  }
}

template <typename T>
DmPeaks search_ffa_peaks_for_dm(
    DedispersedResultView<T> input,
    DmTrialView trials,
    std::size_t dm_index,
    const PreprocessPlan& preprocess,
    const FfaSearchPlan& ffa_plan,
    const FfaSearchOptions& ffa_options,
    std::vector<float>& scratch) {
  const std::span<const T> row = dm_row(input, dm_index);
  std::span<const float> time_series;
  if constexpr (std::is_same_v<T, float>) {
    if (preprocess.steps.empty()) {
      time_series = row;
    } else {
      scratch.assign(row.begin(), row.end());
      preprocess_time_series_inplace_cpu(scratch, preprocess);
      time_series = scratch;
    }
  } else {
    scratch.resize(row.size());
    for (std::size_t index = 0; index < row.size(); ++index) {
      scratch[index] = static_cast<float>(row[index]);
    }
    preprocess_time_series_inplace_cpu(scratch, preprocess);
    time_series = scratch;
  }

  const std::vector<PeriodicPeak> peaks = search_ffa_cpu(
      time_series, ffa_plan, ffa_options);
  return attach_dm_peaks(peaks, trials.values[dm_index],
                         trials.index_offset + dm_index);
}

template <typename T>
DmPeaks search_dm_ffa_impl(DedispersedResultView<T> input,
                           DmTrialView trials,
                           const FfaSearchPlan& ffa_plan,
                           const DmFfaOptions& options) {
  validate_dedispersed_result(input);
  validate_dm_search_inputs(input.shape, trials, ffa_plan, options);

  DmPeaks global_peaks;
  std::exception_ptr error;
  std::atomic_bool has_error = false;
  const bool parallel = input.shape.ndm > 4;

#pragma omp parallel if(parallel)
  {
    DmPeaks local_peaks;
    std::vector<float> scratch;

#pragma omp for schedule(dynamic, 1)
    for (std::size_t dm_index = 0; dm_index < input.shape.ndm; ++dm_index) {
      if (has_error.load(std::memory_order_relaxed)) {
        continue;
      }
      try {
        DmPeaks peaks = search_ffa_peaks_for_dm(
            input, trials, dm_index, options.preprocess, ffa_plan,
            options.search, scratch);
        if (!peaks.empty()) {
          local_peaks.insert(local_peaks.end(),
                             std::make_move_iterator(peaks.begin()),
                             std::make_move_iterator(peaks.end()));
        }
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
      global_peaks.insert(global_peaks.end(),
                          std::make_move_iterator(local_peaks.begin()),
                          std::make_move_iterator(local_peaks.end()));
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  std::stable_sort(global_peaks.begin(), global_peaks.end(),
                   [](const DmPeak& lhs, const DmPeak& rhs) {
                     return lhs.dm_index < rhs.dm_index;
                   });
  return global_peaks;
}

}  // namespace

DmPeaks search_dm_ffa_cpu(
    DedispersedResultView<std::uint32_t> input,
    DmTrialView trials,
    const FfaSearchPlan& plan,
    const DmFfaOptions& options) {
  return search_dm_ffa_impl(input, trials, plan, options);
}

DmPeaks search_dm_ffa_cpu(
    DedispersedResultView<float> input,
    DmTrialView trials,
    const FfaSearchPlan& plan,
    const DmFfaOptions& options) {
  return search_dm_ffa_impl(input, trials, plan, options);
}

DmPeaks search_dm_ffa_cpu(
    const DedispersedResult<std::uint32_t>& input,
    DmTrialView trials,
    const FfaSearchPlan& plan,
    const DmFfaOptions& options) {
  return search_dm_ffa_cpu(input.view(), trials, plan, options);
}

DmPeaks search_dm_ffa_cpu(
    const DedispersedResult<float>& input,
    DmTrialView trials,
    const FfaSearchPlan& plan,
    const DmFfaOptions& options) {
  return search_dm_ffa_cpu(input.view(), trials, plan, options);
}

}  // namespace gaffa
