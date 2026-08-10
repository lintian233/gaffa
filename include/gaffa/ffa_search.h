#pragma once

#include "gaffa/ffa_peak.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/peak_reduction.h"

#include <cstddef>
#include <span>
#include <string>
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
  // CUDA bounded reduction may return a partial result with diagnostics. CPU
  // search always leaves this true and the warning list empty.
  bool complete = true;
  std::vector<std::string> warnings;
};

// Executes an already-built FFA search plan on a finite, preprocessed 1D float
// time series and returns native FFA responses. The input must be
// approximately zero-mean with unit sample variance. This raw API does not
// detrend, normalise, or attach DM metadata; task-local weighted-sum
// downsampling is part of FFA execution.
FfaSearchResult search_ffa_raw_cpu(
    std::span<const float> preprocessed_time_series,
    const FfaSearchPlan& plan,
    const FfaSearchOptions& options = {});

// Executes an already-built FFA search plan on a preprocessed 1D float time
// series and returns backend-neutral periodic peaks. Every result uses the
// plan observation's midpoint as its reference epoch.
std::vector<PeriodicPeak> search_ffa_cpu(
    std::span<const float> preprocessed_time_series,
    const FfaSearchPlan& plan,
    const FfaSearchOptions& options = {});

// Applies the Native FFA bounded reduction to one series of raw FFA peaks.
// Peaks are grouped by the same frequency bucket used by Native CUDA, ranked
// by S/N, and limited by top_k_per_group and max_groups_per_series. The
// observation is required because a zero frequency tolerance means one
// Fourier bin (1 / duration_seconds). The input is consumed by value so the
// normal DM wrapper can reduce in-place without a second raw-peak copy.
FfaSearchResult reduce_ffa_peaks_cpu(
    std::vector<FfaPeak> peaks,
    const FfaObservation& observation,
    const PeakReductionOptions& options);

}  // namespace gaffa
