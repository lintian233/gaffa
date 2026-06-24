#include "gaffa/ffa_search.h"

#include "gaffa/ffa_executor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace gaffa {
namespace {

bool is_better_candidate(const FfaCandidate& lhs, const FfaCandidate& rhs) {
  if (lhs.snr != rhs.snr) {
    return lhs.snr > rhs.snr;
  }
  if (lhs.period != rhs.period) {
    return lhs.period < rhs.period;
  }
  if (lhs.width != rhs.width) {
    return lhs.width < rhs.width;
  }
  return lhs.phase < rhs.phase;
}

struct WorstCandidateFirst {
  bool operator()(const FfaCandidate& lhs, const FfaCandidate& rhs) const {
    return is_better_candidate(lhs, rhs);
  }
};

class CandidateTopK {
 public:
  explicit CandidateTopK(std::size_t max_candidates)
      : max_candidates_(max_candidates) {
    candidates_.reserve(max_candidates);
  }

  void consider(const FfaCandidate& candidate) {
    if (candidates_.size() < max_candidates_) {
      candidates_.push_back(candidate);
      std::push_heap(candidates_.begin(), candidates_.end(),
                     WorstCandidateFirst{});
      return;
    }
    if (is_better_candidate(candidate, candidates_.front())) {
      std::pop_heap(candidates_.begin(), candidates_.end(),
                    WorstCandidateFirst{});
      candidates_.back() = candidate;
      std::push_heap(candidates_.begin(), candidates_.end(),
                     WorstCandidateFirst{});
    }
  }

  std::vector<FfaCandidate> sorted() && {
    std::sort(candidates_.begin(), candidates_.end(), is_better_candidate);
    return std::move(candidates_);
  }

 private:
  std::size_t max_candidates_ = 0;
  std::vector<FfaCandidate> candidates_;
};

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
                              CandidateTopK& global_candidates) {
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

  CandidateTopK global_candidates(options.max_candidates);
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
