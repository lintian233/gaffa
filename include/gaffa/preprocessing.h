#pragma once

#include "gaffa/time_series.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct DetrendRunningMedianOptions {
  // Centered running median requires an odd sample count so the window has a
  // single center sample. Use running_median_window_seconds() when matching a
  // Riptide-style seconds-based window.
  std::size_t window_samples = 101;
  // Match Riptide's fast_running_median: large windows are evaluated on a
  // mean-scrunched copy so at least this many low-resolution points span the
  // requested window. Must be odd.
  std::size_t min_points = 101;
};

enum class PreprocessStepKind {
  DetrendRunningMedian,
  Normalise,
};

struct PreprocessStep {
  PreprocessStepKind kind = PreprocessStepKind::Normalise;
  DetrendRunningMedianOptions detrend_running_median{};
  NormaliseOptions normalise{};
};

struct PreprocessPlan {
  std::vector<PreprocessStep> steps;
};

struct RiptidePreprocessOptions {
  double running_median_width_seconds = 5.0;
  std::size_t running_median_min_points = 101;
  bool normalise = true;
  NormaliseOptions normalise_options{};
};

DetrendRunningMedianOptions running_median_window_seconds(
    double window_seconds,
    double tsamp);

PreprocessPlan make_riptide_preprocess_plan(
    double tsamp,
    const RiptidePreprocessOptions& options = {});

// Subtracts a Riptide-style approximate running median from each sample. Large
// windows are mean-scrunched, median filtered at low resolution, then linearly
// interpolated back to the input length. The exact median path uses edge-value
// padding at the boundaries.
void detrend_running_median_cpu(std::span<const float> input,
                                DetrendRunningMedianOptions options,
                                std::span<float> output);

std::vector<float> detrend_running_median_cpu(
    std::span<const float> input,
    DetrendRunningMedianOptions options);

TimeSeries preprocess_time_series_cpu(const TimeSeries& input,
                                      const PreprocessPlan& plan);

// Applies the plan and writes the final result back into input.data. Steps such
// as running-median detrending may still allocate temporary buffers internally.
void preprocess_time_series_inplace_cpu(TimeSeries& input,
                                        const PreprocessPlan& plan);

}  // namespace gaffa
