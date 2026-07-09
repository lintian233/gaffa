#include "gaffa/ffa_search.h"

#include "gaffa/ffa_detection.h"
#include "gaffa/ffa_executor.h"

#include <cmath>
#include <stdexcept>

namespace gaffa {
namespace {

void validate_search_inputs(const FfaSearchPlan& plan,
                            const FfaSearchOptions& options) {
  if (!std::isfinite(options.snr_threshold)) {
    throw std::invalid_argument(
        "FFA search S/N threshold must be finite");
  }
  if (plan.width_trials.empty()) {
    throw std::invalid_argument(
        "FFA search plan width_trials must not be empty");
  }
}

}  // namespace

FfaSearchResult search_ffa_cpu(std::span<const float> time_series,
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

}  // namespace gaffa
