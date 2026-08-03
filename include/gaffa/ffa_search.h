#pragma once

#include "gaffa/ffa_peak.h"
#include "gaffa/ffa_plan.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct FfaSearchOptions {
  float snr_threshold = 6.0F;
  // 0 means unbounded. This is a per-time-series raw-peak safety guard across
  // every task in that series' FFA plan; reaching it raises rather than
  // silently truncating output. In CUDA batch search the guard applies
  // independently to each series. Final candidate limiting belongs to
  // candidate filtering.
  std::size_t max_peaks = 0;
};

struct FfaSearchResult {
  std::vector<FfaPeak> peaks;
};

// Executes an already-built FFA search plan on a finite, preprocessed 1D float
// time series. The input must be approximately zero-mean with unit sample
// variance. This raw FFA API does not detrend, normalise, or attach DM
// metadata; task-local weighted-sum downsampling is part of FFA execution.
FfaSearchResult search_ffa_cpu(std::span<const float> preprocessed_time_series,
                               const FfaSearchPlan& plan,
                               const FfaSearchOptions& options = {});

// Convenience composition of raw FFA search and projection to the
// backend-neutral periodic model. Every result uses the plan observation's
// midpoint as its reference epoch.
std::vector<PeriodicPeak> search_ffa_periodic_cpu(
    std::span<const float> preprocessed_time_series,
    const FfaSearchPlan& plan,
    const FfaSearchOptions& options = {});

}  // namespace gaffa
