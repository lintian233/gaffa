#pragma once

#include "gaffa/periodic_match.h"

#include <array>

namespace gaffa::detail {

using PhasePolynomial = std::array<long double, 5>;

PhasePolynomial make_phase_polynomial(
    const PeriodicMotion& motion,
    double observation_seconds) noexcept;

PhaseDrift phase_drift(const PhasePolynomial& lhs,
                       long double lhs_scale,
                       const PhasePolynomial& rhs,
                       long double rhs_scale,
                       double observation_seconds);

bool sampled_phase_within(const PhasePolynomial& lhs,
                          const PhasePolynomial& rhs,
                          double maximum_cycles) noexcept;

FrequencyDrift frequency_drift(const PhasePolynomial& lhs,
                               long double lhs_scale,
                               const PhasePolynomial& rhs,
                               long double rhs_scale,
                               double observation_seconds);

}  // namespace gaffa::detail
