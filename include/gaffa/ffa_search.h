#pragma once

#include "gaffa/ffa_peak.h"
#include "gaffa/ffa_plan.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct FfaSearchOptions {
  float snr_threshold = 6.0F;
  // 0 means unbounded. Applied independently to each FFA block's raw
  // detection. Final candidate limiting belongs to candidate filtering.
  std::size_t max_peaks = 0;
};

struct FfaSearchResult {
  std::vector<FfaPeak> peaks;
};

// Executes an already-built FFA search plan on a 1D float time series. This is
// a thin composition layer: it streams blocks through the CPU executor, runs
// per-block peak detection, and returns all significant peaks.
FfaSearchResult search_ffa_cpu(std::span<const float> time_series,
                               const FfaSearchPlan& plan,
                               const FfaSearchOptions& options = {});

}  // namespace gaffa
