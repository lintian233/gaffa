#pragma once

#include "gaffa/ffa.h"
#include "gaffa/ffa_plan.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct FfaCandidate {
  double period = 0.0;
  std::size_t width = 0;
  std::size_t phase = 0;
  std::size_t shift = 0;
  std::size_t bins = 0;
  float snr = 0.0F;
};

struct FfaDetectionOptions {
  float snr_threshold = 6.0F;
  std::size_t max_candidates = 1024;
};

std::vector<FfaCandidate> detect_ffa_block_cpu(
    std::span<const float> transform,
    FfaTransformShape shape,
    const FfaSearchTask& task,
    std::span<const std::size_t> width_trials,
    float stdnoise,
    const FfaDetectionOptions& options = {});

}  // namespace gaffa
