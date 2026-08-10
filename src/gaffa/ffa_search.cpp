#include "gaffa/ffa_search.h"

#include "gaffa/ffa_detection.h"
#include "gaffa/ffa_executor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace gaffa {
namespace {

void validate_search_inputs(const FfaSearchPlan& plan,
                            const FfaSearchOptions& options) {
  if (!std::isfinite(options.snr_threshold)) {
    throw std::invalid_argument(
        "FFA search S/N threshold must be finite");
  }
  validate_ffa_search_plan(plan);
  if (plan.width_trials.empty()) {
    throw std::invalid_argument(
        "FFA search plan width_trials must not be empty");
  }
}

}  // namespace

FfaSearchResult search_ffa_raw_cpu(std::span<const float> time_series,
                                   const FfaSearchPlan& plan,
                                   const FfaSearchOptions& options) {
  validate_search_inputs(plan, options);

  std::vector<FfaPeak> peaks;
  FfaPeakCollector collector{
      .peaks = &peaks,
      .max_peaks = options.max_peaks,
  };
  const FfaDetectionOptions detection_options{
      .snr_threshold = options.snr_threshold,
      .max_peaks = options.max_peaks,
  };
  const FfaSearchTask* current_task = nullptr;
  FfaDetectionPlan detection_plan;
  std::vector<float> circular_prefix;

  for_each_ffa_row_cpu(time_series, plan, [&](const FfaRowView& row) {
    if (row.task != current_task) {
      current_task = row.task;
      detection_plan =
          make_ffa_detection_plan(plan.width_trials, current_task->bins);
      circular_prefix.assign(
          current_task->bins + detection_plan.max_width + 1, 0.0F);
    }
    detect_ffa_row_cpu(row.profile, row.shift, *row.task, detection_plan,
                       row.stdnoise, detection_options, circular_prefix,
                       collector);
  });

  sort_ffa_peaks(peaks);
  return FfaSearchResult{
      .peaks = std::move(peaks),
  };
}

std::vector<PeriodicPeak> search_ffa_cpu(
    std::span<const float> preprocessed_time_series,
    const FfaSearchPlan& plan,
    const FfaSearchOptions& options) {
  const FfaSearchResult raw =
      search_ffa_raw_cpu(preprocessed_time_series, plan, options);
  return periodic_peaks_from_ffa(raw.peaks, plan.observation);
}

FfaSearchResult reduce_ffa_peaks_cpu(
    std::vector<FfaPeak> peaks, const FfaObservation& observation,
    const PeakReductionOptions& options) {
  if (!options.enabled()) {
    return FfaSearchResult{.peaks = std::move(peaks)};
  }
  if (options.max_groups_per_series == 0) {
    throw std::invalid_argument(
        "FFA peak reduction requires max_groups_per_series > 0");
  }
  if (observation.nsamples == 0 ||
      !(observation.tsamp_seconds > 0.0) ||
      !std::isfinite(observation.tsamp_seconds)) {
    throw std::invalid_argument(
        "FFA peak reduction observation must be valid");
  }
  const double duration_seconds = observation.duration_seconds();
  if (!(duration_seconds > 0.0) || !std::isfinite(duration_seconds)) {
    throw std::invalid_argument(
        "FFA peak reduction observation duration must be finite and > 0");
  }

  double tolerance_hz = options.frequency_tolerance_hz;
  if (tolerance_hz == 0.0) {
    tolerance_hz = 1.0 / duration_seconds;
  }
  if (!(tolerance_hz > 0.0) || !std::isfinite(tolerance_hz)) {
    throw std::invalid_argument(
        "FFA frequency reduction tolerance must be finite and > 0");
  }

  std::unordered_map<std::uint64_t, std::vector<FfaPeak>> groups;
  groups.reserve(std::min(peaks.size(), options.max_groups_per_series));
  for (FfaPeak& peak : peaks) {
    if (!std::isfinite(peak.frequency) || !(peak.frequency > 0.0)) {
      throw std::invalid_argument(
          "FFA peak reduction requires finite positive peak frequency");
    }
    const long double bucket_value = std::floor(
        static_cast<long double>(peak.frequency) /
        static_cast<long double>(tolerance_hz));
    if (!(bucket_value >= 0.0L) ||
        bucket_value >
            static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
      throw std::overflow_error(
          "FFA frequency reduction bucket index overflow");
    }
    const std::uint64_t bucket = static_cast<std::uint64_t>(bucket_value);
    auto& group = groups[bucket];
    group.push_back(std::move(peak));
  }

  for (auto& entry : groups) {
    auto& group = entry.second;
    std::sort(group.begin(), group.end(), is_better_ffa_peak);
    if (group.size() > options.top_k_per_group) {
      group.resize(options.top_k_per_group);
    }
  }

  std::vector<std::uint64_t> group_keys;
  group_keys.reserve(groups.size());
  for (const auto& entry : groups) {
    if (!entry.second.empty()) {
      group_keys.push_back(entry.first);
    }
  }
  std::sort(group_keys.begin(), group_keys.end(),
            [&groups](std::uint64_t lhs, std::uint64_t rhs) {
              return is_better_ffa_peak(groups.at(lhs).front(),
                                        groups.at(rhs).front());
            });

  FfaSearchResult result;
  if (group_keys.size() > options.max_groups_per_series) {
    group_keys.resize(options.max_groups_per_series);
    result.complete = false;
    result.warnings.push_back(
        "Native CPU reduction discarded frequency groups because "
        "max_groups_per_series was exceeded");
  }
  std::size_t retained_peak_count = 0;
  for (const std::uint64_t key : group_keys) {
    retained_peak_count += groups.at(key).size();
  }
  result.peaks.reserve(retained_peak_count);
  for (const std::uint64_t key : group_keys) {
    auto& group = groups.at(key);
    result.peaks.insert(result.peaks.end(),
                        std::make_move_iterator(group.begin()),
                        std::make_move_iterator(group.end()));
  }
  sort_ffa_peaks(result.peaks);
  return result;
}

}  // namespace gaffa
