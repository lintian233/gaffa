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

struct FfaBoxcarTrial {
  std::size_t width = 0;
  std::size_t width_index = 0;
  float height = 0.0F;
  float baseline = 0.0F;
  double duty_cycle = 0.0;
};

struct FfaDetectionPlan {
  std::size_t bins = 0;
  std::vector<FfaBoxcarTrial> boxcar_trials;
  std::size_t max_width = 0;
};

struct FfaPeakCollector {
  std::vector<FfaPeak>* peaks = nullptr;
  std::size_t max_peaks = 0;

  void add(const FfaPeak& peak);
};

FfaDetectionPlan make_ffa_detection_plan(
    std::span<const std::size_t> width_trials,
    std::size_t bins);

void detect_ffa_row_cpu(
    std::span<const float> profile,
    std::size_t shift,
    const FfaSearchTask& task,
    const FfaDetectionPlan& plan,
    float stdnoise,
    const FfaDetectionOptions& options,
    std::span<float> circular_prefix,
    FfaPeakCollector& collector);

std::vector<FfaPeak> find_ffa_peaks_cpu(
    std::span<const float> transform,
    FfaTransformShape shape,
    const FfaSearchTask& task,
    std::span<const std::size_t> width_trials,
    float stdnoise,
    const FfaDetectionOptions& options = {});

}  // namespace gaffa
