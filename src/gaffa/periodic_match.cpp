#include "gaffa/periodic_match.h"

#include "detail/phase_match.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gaffa {
namespace {

constexpr long double kSpeedOfLightMPerS = 299792458.0L;
constexpr std::size_t kPolynomialOrder = 4;
constexpr std::size_t kRootCapacity = kPolynomialOrder + 1;
using Polynomial = detail::PhasePolynomial;

struct RootSet {
  std::array<long double, kRootCapacity> values{};
  std::size_t count = 0;
};

long double evaluate(const Polynomial& coefficients,
                     std::size_t degree,
                     long double x) {
  long double value = coefficients[degree];
  while (degree != 0) {
    --degree;
    value = value * x + coefficients[degree];
  }
  return value;
}

std::size_t effective_degree(const Polynomial& coefficients,
                             std::size_t degree) {
  long double scale = 0.0L;
  for (std::size_t index = 0; index <= degree; ++index) {
    scale = std::max(scale, std::abs(coefficients[index]));
  }
  const long double tolerance =
      64.0L * std::numeric_limits<long double>::epsilon() *
      std::max(1.0L, scale);
  while (degree != 0 && std::abs(coefficients[degree]) <= tolerance) {
    --degree;
  }
  return degree;
}

void append_unique_root(RootSet& roots, long double root) {
  root = std::clamp(root, 0.0L, 1.0L);
  constexpr long double kRootTolerance =
      256.0L * std::numeric_limits<long double>::epsilon();
  if (roots.count != 0 &&
      std::abs(roots.values[roots.count - 1] - root) <= kRootTolerance) {
    return;
  }
  if (roots.count >= roots.values.size()) {
    throw std::logic_error("Polynomial root capacity exceeded");
  }
  roots.values[roots.count++] = root;
}

RootSet roots_on_unit_interval(const Polynomial& input,
                               std::size_t input_degree) {
  const std::size_t degree = effective_degree(input, input_degree);
  RootSet roots;
  if (degree == 0) {
    return roots;
  }
  if (degree == 1) {
    const long double root = -input[0] / input[1];
    if (root >= 0.0L && root <= 1.0L) {
      append_unique_root(roots, root);
    }
    return roots;
  }

  Polynomial derivative{};
  for (std::size_t index = 1; index <= degree; ++index) {
    derivative[index - 1] =
        static_cast<long double>(index) * input[index];
  }
  const RootSet critical = roots_on_unit_interval(derivative, degree - 1);

  std::array<long double, kRootCapacity + 2> boundaries{};
  std::size_t boundary_count = 0;
  boundaries[boundary_count++] = 0.0L;
  for (std::size_t index = 0; index < critical.count; ++index) {
    boundaries[boundary_count++] = critical.values[index];
  }
  boundaries[boundary_count++] = 1.0L;

  long double scale = 0.0L;
  for (std::size_t index = 0; index <= degree; ++index) {
    scale = std::max(scale, std::abs(input[index]));
  }
  const long double value_tolerance =
      128.0L * std::numeric_limits<long double>::epsilon() *
      std::max(1.0L, scale);

  for (std::size_t index = 0; index < boundary_count; ++index) {
    const long double point = boundaries[index];
    if (std::abs(evaluate(input, degree, point)) <= value_tolerance) {
      append_unique_root(roots, point);
    }
    if (index + 1 == boundary_count) {
      continue;
    }

    long double left = point;
    long double right = boundaries[index + 1];
    long double left_value = evaluate(input, degree, left);
    const long double right_value = evaluate(input, degree, right);
    if (left_value == 0.0L || right_value == 0.0L ||
        std::signbit(left_value) == std::signbit(right_value)) {
      continue;
    }

    for (int iteration = 0; iteration < 96; ++iteration) {
      const long double middle = std::midpoint(left, right);
      const long double middle_value = evaluate(input, degree, middle);
      if (middle_value == 0.0L) {
        left = middle;
        right = middle;
        break;
      }
      if (std::signbit(left_value) != std::signbit(middle_value)) {
        right = middle;
      } else {
        left = middle;
        left_value = middle_value;
      }
    }
    append_unique_root(roots, std::midpoint(left, right));
  }
  std::sort(roots.values.begin(), roots.values.begin() + roots.count);
  return roots;
}

PhaseDrift maximum_phase_drift(const Polynomial& residual,
                               double observation_seconds) {
  Polynomial derivative{};
  for (std::size_t index = 1; index <= kPolynomialOrder; ++index) {
    derivative[index - 1] =
        static_cast<long double>(index) * residual[index];
  }
  const RootSet stationary =
      roots_on_unit_interval(derivative, kPolynomialOrder - 1);

  long double best_x = 0.0L;
  long double best_value = std::abs(evaluate(residual, kPolynomialOrder, 0.0L));
  const auto consider = [&](long double x) {
    const long double value =
        std::abs(evaluate(residual, kPolynomialOrder, x));
    if (value > best_value) {
      best_value = value;
      best_x = x;
    }
  };
  for (std::size_t index = 0; index < stationary.count; ++index) {
    consider(stationary.values[index]);
  }
  consider(1.0L);
  return PhaseDrift{
      .maximum_cycles = static_cast<double>(best_value),
      .time_seconds = static_cast<double>(
          best_x * static_cast<long double>(observation_seconds)),
  };
}

void validate_match_arguments(const PeriodicMotion& lhs,
                              const PeriodicMotion& rhs,
                              double observation_seconds) {
  validate_periodic_motion(lhs);
  validate_periodic_motion(rhs);
  if (!(observation_seconds > 0.0) || !std::isfinite(observation_seconds)) {
    throw std::invalid_argument(
        "Periodic match observation_seconds must be finite and > 0");
  }
}

}  // namespace

namespace detail {

PhasePolynomial make_phase_polynomial(
    const PeriodicMotion& motion,
    double observation_seconds) noexcept {
  const long double frequency = motion.frequency_hz;
  const long double acceleration = motion.acceleration_m_per_s2;
  const long double jerk = motion.jerk_m_per_s3;
  const long double snap = motion.snap_m_per_s4;
  const long double epoch_offset = -motion.reference_time_seconds;
  const long double t = observation_seconds;

  PhasePolynomial result{};
  result[1] = frequency * t *
              (1.0L - acceleration * epoch_offset / kSpeedOfLightMPerS -
               jerk * epoch_offset * epoch_offset /
                   (2.0L * kSpeedOfLightMPerS) -
               snap * epoch_offset * epoch_offset * epoch_offset /
                   (6.0L * kSpeedOfLightMPerS));
  result[2] = -frequency * t * t *
              (acceleration / 2.0L + jerk * epoch_offset / 2.0L +
               snap * epoch_offset * epoch_offset / 4.0L) /
              kSpeedOfLightMPerS;
  result[3] = -frequency * t * t * t *
              (jerk / 6.0L + snap * epoch_offset / 6.0L) /
              kSpeedOfLightMPerS;
  result[4] = -frequency * t * t * t * t * snap /
              (24.0L * kSpeedOfLightMPerS);
  return result;
}

PhaseDrift phase_drift(const PhasePolynomial& lhs,
                       long double lhs_scale,
                       const PhasePolynomial& rhs,
                       long double rhs_scale,
                       double observation_seconds) {
  Polynomial residual{};
  for (std::size_t index = 0; index <= kPolynomialOrder; ++index) {
    residual[index] = lhs_scale * lhs[index] - rhs_scale * rhs[index];
  }
  return maximum_phase_drift(residual, observation_seconds);
}

bool sampled_phase_within(const PhasePolynomial& lhs,
                          const PhasePolynomial& rhs,
                          double maximum_cycles) noexcept {
  Polynomial residual{};
  for (std::size_t index = 0; index <= kPolynomialOrder; ++index) {
    residual[index] = lhs[index] - rhs[index];
  }
  constexpr std::array<long double, 4> kSamples{0.25L, 0.5L, 0.75L, 1.0L};
  for (const long double sample : kSamples) {
    const long double value =
        std::abs(evaluate(residual, kPolynomialOrder, sample));
    const long double tolerance =
        64.0L * std::numeric_limits<long double>::epsilon() *
        std::max(1.0L, value);
    if (value > static_cast<long double>(maximum_cycles) + tolerance) {
      return false;
    }
  }
  return true;
}

FrequencyDrift frequency_drift(const PhasePolynomial& lhs,
                               long double lhs_scale,
                               const PhasePolynomial& rhs,
                               long double rhs_scale,
                               double observation_seconds) {
  Polynomial frequency{};
  for (std::size_t index = 1; index <= kPolynomialOrder; ++index) {
    frequency[index - 1] =
        static_cast<long double>(index) *
        (lhs_scale * lhs[index] - rhs_scale * rhs[index]) /
        static_cast<long double>(observation_seconds);
  }

  Polynomial derivative{};
  for (std::size_t index = 1; index < kPolynomialOrder; ++index) {
    derivative[index - 1] =
        static_cast<long double>(index) * frequency[index];
  }
  const RootSet stationary =
      roots_on_unit_interval(derivative, kPolynomialOrder - 2);
  long double best_x = 0.0L;
  long double best_value = std::abs(evaluate(frequency, 3, 0.0L));
  const auto consider = [&](long double x) {
    const long double value = std::abs(evaluate(frequency, 3, x));
    if (value > best_value) {
      best_value = value;
      best_x = x;
    }
  };
  for (std::size_t index = 0; index < stationary.count; ++index) {
    consider(stationary.values[index]);
  }
  consider(1.0L);
  return FrequencyDrift{
      .maximum_hz = static_cast<double>(best_value),
      .time_seconds = static_cast<double>(
          best_x * static_cast<long double>(observation_seconds)),
  };
}

}  // namespace detail

PhaseDrift periodic_phase_drift(const PeriodicMotion& lhs,
                                const PeriodicMotion& rhs,
                                double observation_seconds) {
  validate_match_arguments(lhs, rhs, observation_seconds);
  return detail::phase_drift(
      detail::make_phase_polynomial(lhs, observation_seconds), 1.0L,
      detail::make_phase_polynomial(rhs, observation_seconds), 1.0L,
      observation_seconds);
}

PhaseDrift harmonic_phase_drift(const PeriodicMotion& parent,
                                const PeriodicMotion& child,
                                double ratio,
                                double observation_seconds) {
  validate_match_arguments(parent, child, observation_seconds);
  if (!(ratio > 0.0) || !std::isfinite(ratio)) {
    throw std::invalid_argument(
        "Harmonic phase ratio must be finite and > 0");
  }
  return detail::phase_drift(
      detail::make_phase_polynomial(child, observation_seconds), 1.0L,
      detail::make_phase_polynomial(parent, observation_seconds),
      static_cast<long double>(ratio), observation_seconds);
}

FrequencyDrift harmonic_frequency_drift(const PeriodicMotion& parent,
                                        const PeriodicMotion& child,
                                        double ratio,
                                        double observation_seconds) {
  validate_match_arguments(parent, child, observation_seconds);
  if (!(ratio > 0.0) || !std::isfinite(ratio)) {
    throw std::invalid_argument(
        "Harmonic frequency ratio must be finite and > 0");
  }
  return detail::frequency_drift(
      detail::make_phase_polynomial(child, observation_seconds), 1.0L,
      detail::make_phase_polynomial(parent, observation_seconds),
      static_cast<long double>(ratio), observation_seconds);
}

}  // namespace gaffa
