#pragma once

#include "gaffa/ffa_candidate.h"
#include "gaffa/ffa_plan.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct FfaSearchOptions {
  float snr_threshold = 6.0F;
  std::size_t max_candidates = 1024;
};

struct FfaSearchResult {
  std::vector<FfaCandidate> candidates;
};

// Executes an already-built FFA search plan on a 1D float time series. This is
// a thin composition layer: it streams blocks through the CPU executor, runs
// per-block detection, and returns the global top candidates.
FfaSearchResult search_ffa_cpu(std::span<const float> time_series,
                               const FfaSearchPlan& plan,
                               const FfaSearchOptions& options = {});

}  // namespace gaffa
