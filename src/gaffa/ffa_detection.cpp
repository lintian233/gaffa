// The boxcar S/N calculation in this file follows riptide
// (https://github.com/v-morello/riptide), licensed under the MIT License.

#include "gaffa/ffa_detection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gaffa {
namespace {

struct BoxcarPeak {
  float snr = -std::numeric_limits<float>::infinity();
  std::size_t phase = 0;
};

std::size_t checked_block_size(FfaTransformShape shape) {
  if (shape.rows == 0) {
    throw std::invalid_argument("FFA detection rows must be > 0");
  }
  if (shape.bins <= 1) {
    throw std::invalid_argument("FFA detection bins must be > 1");
  }
  if (shape.rows > std::numeric_limits<std::size_t>::max() / shape.bins) {
    throw std::overflow_error("FFA detection shape size overflow");
  }
  return shape.rows * shape.bins;
}

void validate_width_trials(std::span<const std::size_t> width_trials,
                           std::size_t bins) {
  if (width_trials.empty()) {
    throw std::invalid_argument("FFA detection width trials must not be empty");
  }
  for (const std::size_t width : width_trials) {
    if (width == 0 || width >= bins) {
      throw std::invalid_argument(
          "FFA detection width trials must satisfy 0 < width < bins");
    }
  }
}

void validate_detection_arguments(std::span<const float> transform,
                                  FfaTransformShape shape,
                                  const FfaSearchTask& task,
                                  std::span<const std::size_t> width_trials,
                                  float stdnoise,
                                  const FfaDetectionOptions& options) {
  const std::size_t expected_size = checked_block_size(shape);
  if (transform.size() != expected_size) {
    throw std::invalid_argument(
        "FFA detection transform size does not match shape");
  }
  if (task.bins != shape.bins) {
    throw std::invalid_argument("FFA detection task bins must match shape");
  }
  if (task.rows_eval != shape.rows) {
    throw std::invalid_argument("FFA detection task rows_eval must match shape");
  }
  if (task.rows < task.rows_eval || task.rows == 0) {
    throw std::invalid_argument(
        "FFA detection task rows must be >= rows_eval and > 0");
  }
  if (!(task.effective_tsamp > 0.0) || !std::isfinite(task.effective_tsamp)) {
    throw std::invalid_argument(
        "FFA detection task effective_tsamp must be finite and > 0");
  }
  if (!(stdnoise > 0.0F) || !std::isfinite(stdnoise)) {
    throw std::invalid_argument("FFA detection stdnoise must be finite and > 0");
  }
  if (!std::isfinite(options.snr_threshold)) {
    throw std::invalid_argument(
        "FFA detection S/N threshold must be finite");
  }
  if (options.max_candidates == 0) {
    throw std::invalid_argument("FFA detection max_candidates must be > 0");
  }
  validate_width_trials(width_trials, shape.bins);
}

double candidate_period(const FfaSearchTask& task, std::size_t shift) {
  if (task.rows <= 1) {
    return task.effective_tsamp * static_cast<double>(task.bins);
  }

  const double bins = static_cast<double>(task.bins);
  return task.effective_tsamp * bins * bins /
         (bins - static_cast<double>(shift) /
                     static_cast<double>(task.rows - 1));
}

BoxcarPeak boxcar_peak_snr(std::span<const float> profile,
                           std::size_t width,
                           float stdnoise,
                           std::span<const float> circular_prefix,
                           float profile_sum) {
  const std::size_t bins = profile.size();
  float max_boxcar_sum = -std::numeric_limits<float>::infinity();
  std::size_t best_phase = 0;
  for (std::size_t phase = 0; phase < bins; ++phase) {
    const float boxcar_sum =
        circular_prefix[phase + width] - circular_prefix[phase];
    if (boxcar_sum > max_boxcar_sum) {
      max_boxcar_sum = boxcar_sum;
      best_phase = phase;
    }
  }

  const auto width_f = static_cast<float>(width);
  const auto bins_f = static_cast<float>(bins);
  const float height = std::sqrt((bins_f - width_f) / (bins_f * width_f));
  const float baseline = width_f / (bins_f - width_f) * height;
  return BoxcarPeak{
      .snr = ((height + baseline) * max_boxcar_sum -
              baseline * profile_sum) /
             stdnoise,
      .phase = best_phase,
  };
}

float fill_circular_prefix(std::span<const float> profile,
                           std::size_t max_width,
                           std::span<float> circular_prefix) {
  const std::size_t bins = profile.size();
  circular_prefix[0] = 0.0F;
  for (std::size_t index = 0; index < bins + max_width; ++index) {
    circular_prefix[index + 1] =
        circular_prefix[index] + profile[index % bins];
  }
  return circular_prefix[bins];
}

}  // namespace

std::vector<FfaCandidate> detect_ffa_block_cpu(
    std::span<const float> transform,
    FfaTransformShape shape,
    const FfaSearchTask& task,
    std::span<const std::size_t> width_trials,
    float stdnoise,
    const FfaDetectionOptions& options) {
  validate_detection_arguments(transform, shape, task, width_trials, stdnoise,
                               options);

  FfaCandidateTopK candidates(options.max_candidates);
  const std::size_t max_width =
      *std::max_element(width_trials.begin(), width_trials.end());
  std::vector<float> circular_prefix(shape.bins + max_width + 1, 0.0F);

  for (std::size_t shift = 0; shift < shape.rows; ++shift) {
    const auto row = transform.subspan(shift * shape.bins, shape.bins);
    const float profile_sum =
        fill_circular_prefix(row, max_width, circular_prefix);
    for (const std::size_t width : width_trials) {
      const BoxcarPeak peak =
          boxcar_peak_snr(row, width, stdnoise, circular_prefix, profile_sum);
      if (peak.snr >= options.snr_threshold) {
        candidates.consider(FfaCandidate{
            .period = candidate_period(task, shift),
            .width = width,
            .phase = peak.phase,
            .shift = shift,
            .bins = shape.bins,
            .snr = peak.snr,
        });
      }
    }
  }

  return std::move(candidates).sorted();
}

}  // namespace gaffa
