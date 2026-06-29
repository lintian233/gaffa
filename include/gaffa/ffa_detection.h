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
  // 0 means unbounded. This is a per-call raw peak safety guard before
  // candidate clustering, not a final candidate limit.
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
