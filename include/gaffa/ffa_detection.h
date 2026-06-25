#pragma once

#include "gaffa/ffa.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/ffa_peak.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct FfaDetectionOptions {
  float snr_threshold = 6.0F;
  // Frequency clustering radius in Fourier-bin units. The detector converts
  // this to Hz as frequency_cluster_radius / observation_seconds, so 0.1 means
  // peaks within 0.1 independent frequency bins are merged for each width.
  double frequency_cluster_radius = 0.1;
  std::size_t max_peaks = 0;
};

std::vector<FfaPeak> find_ffa_peaks_cpu(
    std::span<const float> transform,
    FfaTransformShape shape,
    const FfaSearchTask& task,
    std::span<const std::size_t> width_trials,
    float stdnoise,
    const FfaDetectionOptions& options = {});

}  // namespace gaffa
