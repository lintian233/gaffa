#include "gaffa/ffa_search.h"

#include "gaffa/ffa_detection.h"
#include "gaffa/ffa_executor.h"

#include <cmath>
#include <stdexcept>

namespace gaffa {
namespace {

void validate_search_inputs(const FfaSearchPlan& plan,
                            const FfaSearchOptions& options) {
  if (options.max_candidates == 0) {
    throw std::invalid_argument("FFA search max_candidates must be > 0");
  }
  if (!std::isfinite(options.snr_threshold)) {
    throw std::invalid_argument(
        "FFA search S/N threshold must be finite");
  }
  if (plan.width_trials.empty()) {
    throw std::invalid_argument(
        "FFA search plan width_trials must not be empty");
  }
}

void collect_block_candidates(const FfaBlockView& block,
                              std::span<const std::size_t> width_trials,
                              const FfaDetectionOptions& detection_options,
                              FfaCandidateTopK& global_candidates) {
  const auto local_candidates =
      detect_ffa_block_cpu(block.transform, block.shape, *block.task,
                           width_trials, block.stdnoise, detection_options);
  for (const auto& candidate : local_candidates) {
    global_candidates.consider(candidate);
  }
}

}  // namespace

FfaSearchResult search_ffa_cpu(std::span<const float> time_series,
                               const FfaSearchPlan& plan,
                               const FfaSearchOptions& options) {
  validate_search_inputs(plan, options);

  FfaCandidateTopK global_candidates(options.max_candidates);
  const FfaDetectionOptions detection_options{
      .snr_threshold = options.snr_threshold,
      .max_candidates = options.max_candidates,
  };

  for_each_ffa_block_cpu(time_series, plan, [&](const FfaBlockView& block) {
    collect_block_candidates(block, plan.width_trials, detection_options,
                             global_candidates);
  });

  return FfaSearchResult{
      .candidates = std::move(global_candidates).sorted(),
  };
}

}  // namespace gaffa
