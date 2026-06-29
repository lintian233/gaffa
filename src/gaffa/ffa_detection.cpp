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

struct BoxcarTrial {
  std::size_t width = 0;
  std::size_t width_index = 0;
  float height = 0.0F;
  float baseline = 0.0F;
  double duty_cycle = 0.0;
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

std::vector<BoxcarTrial> make_boxcar_trials(
    std::span<const std::size_t> width_trials,
    std::size_t bins) {
  const auto bins_f = static_cast<float>(bins);
  std::vector<BoxcarTrial> trials;
  trials.reserve(width_trials.size());
  for (std::size_t width_index = 0; width_index < width_trials.size();
       ++width_index) {
    const std::size_t width = width_trials[width_index];
    const auto width_f = static_cast<float>(width);
    const float height = std::sqrt((bins_f - width_f) / (bins_f * width_f));
    trials.push_back(BoxcarTrial{
        .width = width,
        .width_index = width_index,
        .height = height,
        .baseline = width_f / (bins_f - width_f) * height,
        .duty_cycle = static_cast<double>(width) / static_cast<double>(bins),
    });
  }
  return trials;
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
  validate_width_trials(width_trials, shape.bins);
}

double trial_period(const FfaSearchTask& task, std::size_t shift) {
  if (task.rows <= 1) {
    return task.effective_tsamp * static_cast<double>(task.bins);
  }

  const double bins = static_cast<double>(task.bins);
  return task.effective_tsamp * bins * bins /
         (bins - static_cast<double>(shift) /
                     static_cast<double>(task.rows - 1));
}

void append_peak_with_guard(std::vector<FfaPeak>& peaks,
                            const FfaPeak& peak,
                            const FfaDetectionOptions& options) {
  if (options.max_peaks != 0 && peaks.size() >= options.max_peaks) {
    throw std::runtime_error(
        "FFA detection peak count exceeded max_peaks safety guard");
  }
  peaks.push_back(peak);
}

BoxcarPeak scan_boxcar_peak(const BoxcarTrial& trial,
                           std::span<const float> circular_prefix,
                           std::size_t bins,
                           float profile_sum,
                           float inv_stdnoise) {
  float max_boxcar_sum = -std::numeric_limits<float>::infinity();
  std::size_t best_phase = 0;
  for (std::size_t phase = 0; phase < bins; ++phase) {
    const float boxcar_sum =
        circular_prefix[phase + trial.width] - circular_prefix[phase];
    if (boxcar_sum > max_boxcar_sum) {
      max_boxcar_sum = boxcar_sum;
      best_phase = phase;
    }
  }

  return BoxcarPeak{
      .snr = ((trial.height + trial.baseline) * max_boxcar_sum -
              trial.baseline * profile_sum) *
             inv_stdnoise,
      .phase = best_phase,
  };
}

float fill_circular_prefix(std::span<const float> profile,
                           std::size_t max_width,
                           std::span<float> circular_prefix) {
  const std::size_t bins = profile.size();
  circular_prefix[0] = 0.0F;
  for (std::size_t index = 0; index < bins; ++index) {
    circular_prefix[index + 1] = circular_prefix[index] + profile[index];
  }
  for (std::size_t index = 0; index < max_width; ++index) {
    circular_prefix[bins + index + 1] =
        circular_prefix[bins + index] + profile[index];
  }
  return circular_prefix[bins];
}

FfaPeak make_peak(const BoxcarTrial& trial,
                  const BoxcarPeak& boxcar,
                  double period,
                  double frequency,
                  std::size_t shift,
                  std::size_t bins) {
  return FfaPeak{
      .period = period,
      .frequency = frequency,
      .width = trial.width,
      .duty_cycle = trial.duty_cycle,
      .width_index = trial.width_index,
      .period_index = shift,
      .phase = boxcar.phase,
      .shift = shift,
      .bins = bins,
      .snr = boxcar.snr,
  };
}

}  // namespace

std::vector<FfaPeak> find_ffa_peaks_cpu(
    std::span<const float> transform,
    FfaTransformShape shape,
    const FfaSearchTask& task,
    std::span<const std::size_t> width_trials,
    float stdnoise,
    const FfaDetectionOptions& options) {
  validate_detection_arguments(transform, shape, task, width_trials, stdnoise,
                               options);

  std::vector<FfaPeak> peaks;
  if (options.max_peaks != 0) {
    peaks.reserve(options.max_peaks);
  }
  const std::vector<BoxcarTrial> boxcar_trials =
      make_boxcar_trials(width_trials, shape.bins);
  const std::size_t max_width = std::max_element(
      boxcar_trials.begin(), boxcar_trials.end(),
      [](const BoxcarTrial& lhs, const BoxcarTrial& rhs) {
        return lhs.width < rhs.width;
      })->width;
  std::vector<float> circular_prefix(shape.bins + max_width + 1, 0.0F);
  const float inv_stdnoise = 1.0F / stdnoise;

  for (std::size_t shift = 0; shift < shape.rows; ++shift) {
    const auto row = transform.subspan(shift * shape.bins, shape.bins);
    const float profile_sum =
        fill_circular_prefix(row, max_width, circular_prefix);
    const double period = trial_period(task, shift);
    const double frequency = 1.0 / period;
    for (const BoxcarTrial& trial : boxcar_trials) {
      const BoxcarPeak boxcar = scan_boxcar_peak(
          trial, circular_prefix, shape.bins, profile_sum, inv_stdnoise);
      if (boxcar.snr >= options.snr_threshold) {
        append_peak_with_guard(
            peaks, make_peak(trial, boxcar, period, frequency, shift,
                             shape.bins),
            options);
      }
    }
  }

  sort_ffa_peaks(peaks);
  return peaks;
}

}  // namespace gaffa
