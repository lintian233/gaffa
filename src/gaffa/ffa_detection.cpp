// The boxcar S/N calculation in this file follows riptide
// (https://github.com/v-morello/riptide), licensed under the MIT License.

#include "gaffa/ffa_detection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#include <immintrin.h>
#define GAFFA_HAS_X86_AVX2_DISPATCH 1
#endif

namespace gaffa {
namespace {

struct BoxcarPeak {
  float snr = -std::numeric_limits<float>::infinity();
  std::size_t phase = 0;
};

using ScanBoxcarPeakFn = BoxcarPeak (*)(const FfaBoxcarTrial&,
                                        std::span<const float>,
                                        std::size_t,
                                        float,
                                        float);

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

std::vector<FfaBoxcarTrial> make_boxcar_trials(
    std::span<const std::size_t> width_trials,
    std::size_t bins) {
  const auto bins_f = static_cast<float>(bins);
  std::vector<FfaBoxcarTrial> trials;
  trials.reserve(width_trials.size());
  for (std::size_t width_index = 0; width_index < width_trials.size();
       ++width_index) {
    const std::size_t width = width_trials[width_index];
    const auto width_f = static_cast<float>(width);
    const float height = std::sqrt((bins_f - width_f) / (bins_f * width_f));
    trials.push_back(FfaBoxcarTrial{
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

BoxcarPeak scan_boxcar_peak_portable(const FfaBoxcarTrial& trial,
                                     std::span<const float> circular_prefix,
                                     std::size_t bins,
                                     float profile_sum,
                                     float inv_stdnoise) {
  float max_boxcar_sum = -std::numeric_limits<float>::infinity();
#pragma omp simd reduction(max : max_boxcar_sum)
  for (std::size_t phase = 0; phase < bins; ++phase) {
    const float boxcar_sum =
        circular_prefix[phase + trial.width] - circular_prefix[phase];
    max_boxcar_sum = std::max(max_boxcar_sum, boxcar_sum);
  }

  std::size_t best_phase = 0;
  for (std::size_t phase = 0; phase < bins; ++phase) {
    const float boxcar_sum =
        circular_prefix[phase + trial.width] - circular_prefix[phase];
    if (boxcar_sum == max_boxcar_sum) {
      best_phase = phase;
      break;
    }
  }

  return BoxcarPeak{
      .snr = ((trial.height + trial.baseline) * max_boxcar_sum -
              trial.baseline * profile_sum) *
             inv_stdnoise,
      .phase = best_phase,
  };
}

#if defined(GAFFA_HAS_X86_AVX2_DISPATCH)
__attribute__((target("avx2"))) BoxcarPeak scan_boxcar_peak_avx2(
    const FfaBoxcarTrial& trial,
    std::span<const float> circular_prefix,
    std::size_t bins,
    float profile_sum,
    float inv_stdnoise) {
  if (bins > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return scan_boxcar_peak_portable(trial, circular_prefix, bins, profile_sum,
                                     inv_stdnoise);
  }

  const float* prefix = circular_prefix.data();
  const auto width = static_cast<int>(trial.width);
  const __m256i lane_offsets =
      _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  __m256 best_values =
      _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  __m256i best_phases = lane_offsets;

  std::size_t phase = 0;
  for (; phase + 8 <= bins; phase += 8) {
    const auto phase_i = static_cast<int>(phase);
    const __m256 left = _mm256_loadu_ps(prefix + phase);
    const __m256 right = _mm256_loadu_ps(prefix + phase + trial.width);
    const __m256 sums = _mm256_sub_ps(right, left);
    const __m256 update_mask = _mm256_cmp_ps(sums, best_values, _CMP_GT_OQ);
    best_values = _mm256_blendv_ps(best_values, sums, update_mask);
    const __m256i phases =
        _mm256_add_epi32(_mm256_set1_epi32(phase_i), lane_offsets);
    best_phases = _mm256_blendv_epi8(
        best_phases, phases, _mm256_castps_si256(update_mask));
  }

  alignas(32) std::array<float, 8> lane_values{};
  alignas(32) std::array<int, 8> lane_phases{};
  _mm256_store_ps(lane_values.data(), best_values);
  _mm256_store_si256(reinterpret_cast<__m256i*>(lane_phases.data()),
                     best_phases);

  float max_boxcar_sum = -std::numeric_limits<float>::infinity();
  std::size_t best_phase = 0;
  for (std::size_t lane = 0; lane < lane_values.size(); ++lane) {
    const float value = lane_values[lane];
    const auto candidate_phase = static_cast<std::size_t>(lane_phases[lane]);
    if (value > max_boxcar_sum ||
        (value == max_boxcar_sum && candidate_phase < best_phase)) {
      max_boxcar_sum = value;
      best_phase = candidate_phase;
    }
  }

  for (; phase < bins; ++phase) {
    const float boxcar_sum =
        prefix[phase + static_cast<std::size_t>(width)] - prefix[phase];
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

ScanBoxcarPeakFn select_scan_boxcar_peak_kernel() {
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2")) {
    return scan_boxcar_peak_avx2;
  }
  return scan_boxcar_peak_portable;
}
#else
ScanBoxcarPeakFn select_scan_boxcar_peak_kernel() {
  return scan_boxcar_peak_portable;
}
#endif

FfaPeak make_peak(const FfaBoxcarTrial& trial,
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

void FfaPeakCollector::add(const FfaPeak& peak) {
  if (peaks == nullptr) {
    throw std::invalid_argument("FFA peak collector output must not be null");
  }
  if (max_peaks != 0 && peaks->size() >= max_peaks) {
    throw std::runtime_error(
        "FFA detection peak count exceeded max_peaks safety guard");
  }
  peaks->push_back(peak);
}

FfaDetectionPlan make_ffa_detection_plan(
    std::span<const std::size_t> width_trials,
    std::size_t bins) {
  validate_width_trials(width_trials, bins);
  FfaDetectionPlan plan{
      .bins = bins,
      .boxcar_trials = make_boxcar_trials(width_trials, bins),
  };
  plan.max_width = std::max_element(
                       plan.boxcar_trials.begin(), plan.boxcar_trials.end(),
                       [](const FfaBoxcarTrial& lhs,
                          const FfaBoxcarTrial& rhs) {
                         return lhs.width < rhs.width;
                       })
                       ->width;
  return plan;
}

void detect_ffa_row_cpu(
    std::span<const float> profile,
    std::size_t shift,
    const FfaSearchTask& task,
    const FfaDetectionPlan& plan,
    float stdnoise,
    const FfaDetectionOptions& options,
    std::span<float> circular_prefix,
    FfaPeakCollector& collector) {
  if (profile.size() != plan.bins) {
    throw std::invalid_argument(
        "FFA row detection profile size must match detection plan bins");
  }
  if (task.bins != plan.bins) {
    throw std::invalid_argument(
        "FFA row detection task bins must match detection plan bins");
  }
  if (shift >= task.rows_eval) {
    throw std::invalid_argument(
        "FFA row detection shift must be within rows_eval");
  }
  if (task.rows < task.rows_eval || task.rows == 0) {
    throw std::invalid_argument(
        "FFA row detection task rows must be >= rows_eval and > 0");
  }
  if (!(task.effective_tsamp > 0.0) || !std::isfinite(task.effective_tsamp)) {
    throw std::invalid_argument(
        "FFA row detection task effective_tsamp must be finite and > 0");
  }
  if (!(stdnoise > 0.0F) || !std::isfinite(stdnoise)) {
    throw std::invalid_argument(
        "FFA row detection stdnoise must be finite and > 0");
  }
  if (!std::isfinite(options.snr_threshold)) {
    throw std::invalid_argument(
        "FFA row detection S/N threshold must be finite");
  }
  if (plan.boxcar_trials.empty()) {
    throw std::invalid_argument(
        "FFA row detection plan must contain at least one boxcar trial");
  }
  if (circular_prefix.size() < plan.bins + plan.max_width + 1) {
    throw std::invalid_argument("FFA row detection circular prefix is too small");
  }

  const float profile_sum =
      fill_circular_prefix(profile, plan.max_width, circular_prefix);
  const float inv_stdnoise = 1.0F / stdnoise;
  const ScanBoxcarPeakFn scan_boxcar_peak = select_scan_boxcar_peak_kernel();
  bool period_ready = false;
  double period = 0.0;
  double frequency = 0.0;
  for (const FfaBoxcarTrial& trial : plan.boxcar_trials) {
    const BoxcarPeak boxcar = scan_boxcar_peak(
        trial, circular_prefix, plan.bins, profile_sum, inv_stdnoise);
    if (boxcar.snr >= options.snr_threshold) {
      if (!period_ready) {
        period = trial_period(task, shift);
        frequency = 1.0 / period;
        period_ready = true;
      }
      collector.add(
          make_peak(trial, boxcar, period, frequency, shift, plan.bins));
    }
  }
}

std::vector<FfaPeak> find_ffa_peaks_cpu(
    std::span<const float> transform,
    FfaTransformShape shape,
    const FfaSearchTask& task,
    std::span<const std::size_t> width_trials,
    float stdnoise,
    const FfaDetectionOptions& options) {
  validate_detection_arguments(transform, shape, task, stdnoise, options);

  std::vector<FfaPeak> peaks;
  if (options.max_peaks != 0) {
    peaks.reserve(options.max_peaks);
  }
  const FfaDetectionPlan detection_plan =
      make_ffa_detection_plan(width_trials, shape.bins);
  std::vector<float> circular_prefix(
      shape.bins + detection_plan.max_width + 1, 0.0F);
  FfaPeakCollector collector{
      .peaks = &peaks,
      .max_peaks = options.max_peaks,
  };

  for (std::size_t shift = 0; shift < shape.rows; ++shift) {
    const auto row = transform.subspan(shift * shape.bins, shape.bins);
    detect_ffa_row_cpu(row, shift, task, detection_plan, stdnoise, options,
                       circular_prefix, collector);
  }

  sort_ffa_peaks(peaks);
  return peaks;
}

}  // namespace gaffa
