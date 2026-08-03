#include "gaffa/ffa_peak.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gaffa {
namespace {

bool is_finite_peak(const FfaPeak& peak) {
  return std::isfinite(peak.snr) && std::isfinite(peak.period) &&
         std::isfinite(peak.frequency);
}

void validate_projection_observation(const FfaObservation& observation) {
  if (observation.nsamples == 0 ||
      !(observation.tsamp_seconds > 0.0) ||
      !std::isfinite(observation.tsamp_seconds) ||
      !(observation.duration_seconds() > 0.0) ||
      !std::isfinite(observation.duration_seconds())) {
    throw std::invalid_argument(
        "FFA periodic projection observation must have finite positive "
        "nsamples and tsamp_seconds");
  }
}

PeriodicPeak project_ffa_peak(const FfaPeak& peak,
                              double reference_time_seconds) {
  return PeriodicPeak{
      .motion = {
          .order = MotionOrder::Frequency,
          .reference_time_seconds = reference_time_seconds,
          .frequency_hz = peak.frequency,
      },
      .phase_bin = peak.phase,
      .phase_bins = peak.bins,
      .boxcar_width_bins = peak.width,
      .duty_cycle = peak.duty_cycle,
      .snr = peak.snr,
  };
}

}  // namespace

bool is_better_ffa_peak(const FfaPeak& lhs, const FfaPeak& rhs) {
  const bool lhs_finite = is_finite_peak(lhs);
  const bool rhs_finite = is_finite_peak(rhs);
  if (lhs_finite != rhs_finite) {
    return lhs_finite;
  }
  if (lhs.snr != rhs.snr) {
    return lhs.snr > rhs.snr;
  }
  if (lhs.period != rhs.period) {
    return lhs.period < rhs.period;
  }
  if (lhs.frequency != rhs.frequency) {
    return lhs.frequency < rhs.frequency;
  }
  if (lhs.width != rhs.width) {
    return lhs.width < rhs.width;
  }
  if (lhs.phase != rhs.phase) {
    return lhs.phase < rhs.phase;
  }
  if (lhs.shift != rhs.shift) {
    return lhs.shift < rhs.shift;
  }
  return lhs.bins < rhs.bins;
}

void sort_ffa_peaks(std::vector<FfaPeak>& peaks) {
  std::sort(peaks.begin(), peaks.end(), is_better_ffa_peak);
}

PeriodicPeak periodic_peak_from_ffa(const FfaPeak& peak,
                                    const FfaObservation& observation) {
  validate_projection_observation(observation);
  return project_ffa_peak(peak, observation.reference_time_seconds());
}

std::vector<PeriodicPeak> periodic_peaks_from_ffa(
    std::span<const FfaPeak> peaks,
    const FfaObservation& observation) {
  validate_projection_observation(observation);
  const double reference_time_seconds = observation.reference_time_seconds();
  std::vector<PeriodicPeak> result;
  result.reserve(peaks.size());
  for (const FfaPeak& peak : peaks) {
    result.push_back(project_ffa_peak(peak, reference_time_seconds));
  }
  return result;
}

}  // namespace gaffa
