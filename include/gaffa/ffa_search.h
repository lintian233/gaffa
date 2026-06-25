#pragma once

#include "gaffa/ffa_peak.h"
#include "gaffa/ffa_plan.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct FfaSearchOptions {
  float snr_threshold = 6.0F;
  // Frequency clustering radius in Fourier-bin units. This is converted to Hz
  // per FFA task as frequency_cluster_radius / observation_seconds.
  double frequency_cluster_radius = 0.1;
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
