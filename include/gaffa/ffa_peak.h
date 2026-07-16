#pragma once

#include <cstddef>
#include <vector>

#include "gaffa/periodic_peak.h"

namespace gaffa {

struct FfaPeak {
  double period = 0.0;
  double frequency = 0.0;
  std::size_t width = 0;
  double duty_cycle = 0.0;
  std::size_t width_index = 0;
  std::size_t period_index = 0;
  std::size_t phase = 0;
  std::size_t shift = 0;
  std::size_t bins = 0;
  float snr = 0.0F;
};

bool is_better_ffa_peak(const FfaPeak& lhs, const FfaPeak& rhs);

void sort_ffa_peaks(std::vector<FfaPeak>& peaks);

// Converts an FFA-specific peak to the backend-neutral periodic peak model.
// FfaPeak has no observation-time context, so the result keeps the default
// reference_time_seconds of zero. A wrapper that owns the searched time series
// assigns its physical reference epoch.
PeriodicPeak periodic_peak_from_ffa(const FfaPeak& peak);

}  // namespace gaffa
