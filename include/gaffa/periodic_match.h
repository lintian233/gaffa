#pragma once

#include "gaffa/periodic_peak.h"

namespace gaffa {

// Maximum absolute phase separation over one searched observation interval.
struct PhaseDrift {
  double maximum_cycles = 0.0;
  double time_seconds = 0.0;
};

struct FrequencyDrift {
  double maximum_hz = 0.0;
  double time_seconds = 0.0;
};

// Compares two periodic trajectories after anchoring both phases at the start
// of the searched interval. The maximum is solved over the full interval, not
// approximated from a fixed set of sample times.
[[nodiscard]] PhaseDrift periodic_phase_drift(
    const PeriodicMotion& lhs,
    const PeriodicMotion& rhs,
    double observation_seconds);

// Compares child phase with ratio * parent phase over the searched interval.
// ratio is child_frequency / parent_frequency for the tested harmonic.
[[nodiscard]] PhaseDrift harmonic_phase_drift(
    const PeriodicMotion& parent,
    const PeriodicMotion& child,
    double ratio,
    double observation_seconds);

// Maximum instantaneous child - ratio * parent frequency mismatch over the
// same anchored observation interval.
[[nodiscard]] FrequencyDrift harmonic_frequency_drift(
    const PeriodicMotion& parent,
    const PeriodicMotion& child,
    double ratio,
    double observation_seconds);

}  // namespace gaffa
