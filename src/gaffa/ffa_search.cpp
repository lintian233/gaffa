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
  if (options.frequency_cluster_radius < 0.0 ||
      !std::isfinite(options.frequency_cluster_radius)) {
    throw std::invalid_argument(
        "FFA search frequency_cluster_radius must be finite and >= 0");
  }
  if (plan.width_trials.empty()) {
    throw std::invalid_argument(
        "FFA search plan width_trials must not be empty");
  }
}

void collect_block_peaks(const FfaBlockView& block,
                         std::span<const std::size_t> width_trials,
                         const FfaDetectionOptions& detection_options,
                         std::vector<FfaPeak>& peaks) {
  auto block_peaks =
      find_ffa_peaks_cpu(block.transform, block.shape, *block.task,
                         width_trials, block.stdnoise, detection_options);
  peaks.insert(peaks.end(), block_peaks.begin(), block_peaks.end());
}

}  // namespace

FfaSearchResult search_ffa_cpu(std::span<const float> time_series,
                               const FfaSearchPlan& plan,
                               const FfaSearchOptions& options) {
  validate_search_inputs(plan, options);

  std::vector<FfaPeak> peaks;
  const FfaDetectionOptions detection_options{
      .snr_threshold = options.snr_threshold,
      .frequency_cluster_radius = options.frequency_cluster_radius,
      .max_peaks = options.max_peaks,
  };

  for_each_ffa_block_cpu(time_series, plan, [&](const FfaBlockView& block) {
    collect_block_peaks(block, plan.width_trials, detection_options, peaks);
  });

  sort_ffa_peaks(peaks);
  return FfaSearchResult{
      .peaks = std::move(peaks),
  };
}

}  // namespace gaffa
