#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "gaffa/ffa_plan.h"
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

// Converts an FFA-specific peak to the backend-neutral periodic peak model at
// the observation midpoint. FfaPeak intentionally carries no observation
// context, so callers must supply the plan's observation explicitly.
PeriodicPeak periodic_peak_from_ffa(const FfaPeak& peak,
                                    const FfaObservation& observation);

std::vector<PeriodicPeak> periodic_peaks_from_ffa(
    std::span<const FfaPeak> peaks,
    const FfaObservation& observation);

}  // namespace gaffa
