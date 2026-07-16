#include "gaffa/harmonic.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace gaffa {
namespace {

constexpr double kDispersionDelayMs = 4.148808e3;

struct HarmonicRatio {
  std::int32_t numerator = 1;
  std::int32_t denominator = 1;
  double value = 1.0;
};

bool is_better_candidate_for_harmonic(const Candidate& lhs,
                                      const Candidate& rhs) {
  if (lhs.best.peak.snr != rhs.best.peak.snr) {
    return lhs.best.peak.snr > rhs.best.peak.snr;
  }
  if (lhs.peak_count != rhs.peak_count) {
    return lhs.peak_count > rhs.peak_count;
  }
  if (lhs.best.peak.motion.frequency_hz !=
      rhs.best.peak.motion.frequency_hz) {
    return lhs.best.peak.motion.frequency_hz <
           rhs.best.peak.motion.frequency_hz;
  }
  return lhs.best.dm_index < rhs.best.dm_index;
}

void validate_context(const HarmonicContext& context) {
  if (!(context.observation_seconds > 0.0) ||
      !std::isfinite(context.observation_seconds)) {
    throw std::invalid_argument(
        "Harmonic context observation_seconds must be finite and > 0");
  }
  if (!(context.frequency_low_mhz > 0.0) ||
      !(context.frequency_high_mhz > 0.0) ||
      !std::isfinite(context.frequency_low_mhz) ||
      !std::isfinite(context.frequency_high_mhz)) {
    throw std::invalid_argument(
        "Harmonic context frequencies must be finite and > 0 MHz");
  }
}

void validate_options(const HarmonicOptions& options) {
  if (options.max_harmonic < 2) {
    throw std::invalid_argument("Harmonic max_harmonic must be >= 2");
  }
  if (options.denominator_max == 0) {
    throw std::invalid_argument("Harmonic denominator_max must be > 0");
  }
  if (options.frequency_tolerance_bins < 0.0 ||
      !std::isfinite(options.frequency_tolerance_bins)) {
    throw std::invalid_argument(
        "Harmonic frequency_tolerance_bins must be finite and >= 0");
  }
  if (options.phase_distance_max < 0.0 ||
      !std::isfinite(options.phase_distance_max)) {
    throw std::invalid_argument(
        "Harmonic phase_distance_max must be finite and >= 0");
  }
  if (options.dm_distance_max < 0.0 ||
      !std::isfinite(options.dm_distance_max)) {
    throw std::invalid_argument(
        "Harmonic dm_distance_max must be finite and >= 0");
  }
  if (options.snr_distance_max < 0.0 ||
      !std::isfinite(options.snr_distance_max)) {
    throw std::invalid_argument(
        "Harmonic snr_distance_max must be finite and >= 0");
  }
}

void validate_candidate(const Candidate& candidate) {
  const DmPeak& best = candidate.best;
  if (!std::isfinite(best.dm) ||
      !std::isfinite(best.peak.motion.reference_time_seconds) ||
      !std::isfinite(best.peak.motion.frequency_hz) ||
      !std::isfinite(best.peak.motion.acceleration_m_per_s2) ||
      !std::isfinite(best.peak.motion.jerk_m_per_s3) ||
      !std::isfinite(best.peak.motion.snap_m_per_s4) ||
      !std::isfinite(best.peak.duty_cycle) ||
      !std::isfinite(best.peak.snr)) {
    throw std::invalid_argument(
        "Harmonic candidates must contain finite scientific values");
  }
  if (best.peak.motion.order != MotionOrder::Frequency) {
    throw std::invalid_argument(
        "Harmonic filtering currently supports frequency-only candidates");
  }
  if (!(best.peak.motion.frequency_hz > 0.0) ||
      !(best.peak.duty_cycle > 0.0) || !(best.peak.snr >= 0.0F)) {
    throw std::invalid_argument(
        "Harmonic candidate frequency, period, duty_cycle, and snr are invalid");
  }
}

std::vector<HarmonicRatio> make_harmonic_ratios(
    const HarmonicOptions& options) {
  std::vector<HarmonicRatio> ratios;
  ratios.reserve(options.max_harmonic * options.denominator_max);
  for (std::size_t numerator = 1; numerator <= options.max_harmonic;
       ++numerator) {
    for (std::size_t denominator = 1; denominator <= options.denominator_max;
         ++denominator) {
      const auto divisor = std::gcd(numerator, denominator);
      const auto reduced_numerator = numerator / divisor;
      const auto reduced_denominator = denominator / divisor;
      if (reduced_numerator == reduced_denominator) {
        continue;
      }
      ratios.push_back(HarmonicRatio{
          .numerator = static_cast<std::int32_t>(reduced_numerator),
          .denominator = static_cast<std::int32_t>(reduced_denominator),
          .value = static_cast<double>(reduced_numerator) /
                   static_cast<double>(reduced_denominator),
      });
    }
  }

  std::sort(ratios.begin(), ratios.end(),
            [](const HarmonicRatio& lhs, const HarmonicRatio& rhs) {
              if (lhs.value != rhs.value) {
                return lhs.value < rhs.value;
              }
              if (lhs.numerator != rhs.numerator) {
                return lhs.numerator < rhs.numerator;
              }
              return lhs.denominator < rhs.denominator;
            });
  ratios.erase(std::unique(ratios.begin(), ratios.end(),
                           [](const HarmonicRatio& lhs,
                              const HarmonicRatio& rhs) {
                             return lhs.numerator == rhs.numerator &&
                                    lhs.denominator == rhs.denominator;
                           }),
               ratios.end());
  return ratios;
}

std::optional<HarmonicRatio> nearest_harmonic_ratio(
    double ratio,
    const std::vector<HarmonicRatio>& ratios) {
  if (!(ratio > 0.0) || ratios.empty()) {
    return std::nullopt;
  }

  const auto lower = std::lower_bound(
      ratios.begin(), ratios.end(), ratio,
      [](const HarmonicRatio& lhs, double value) { return lhs.value < value; });

  std::optional<HarmonicRatio> best;
  auto consider = [&](std::vector<HarmonicRatio>::const_iterator it) {
    if (it == ratios.end()) {
      return;
    }
    if (!best ||
        std::abs(it->value - ratio) < std::abs(best->value - ratio)) {
      best = *it;
    }
  };

  consider(lower);
  if (lower != ratios.begin()) {
    consider(std::prev(lower));
  }
  return best;
}

double pulse_width_seconds(const Candidate& candidate) {
  return candidate.best.peak.period_seconds() * candidate.best.peak.duty_cycle;
}

double dm_distance(const Candidate& parent,
                   const Candidate& child,
                   const HarmonicContext& context) {
  const double low = std::min(context.frequency_low_mhz,
                              context.frequency_high_mhz);
  const double high = std::max(context.frequency_low_mhz,
                               context.frequency_high_mhz);
  const double band_delay_ms =
      std::abs(parent.best.dm - child.best.dm) * kDispersionDelayMs *
      std::abs(1.0 / (low * low) - 1.0 / (high * high));
  const double width_seconds =
      std::min(pulse_width_seconds(parent), pulse_width_seconds(child));
  return (band_delay_ms / 1000.0) / width_seconds;
}

std::optional<HarmonicRelation> test_harmonic_relation(
    const Candidate& parent,
    const Candidate& child,
    std::size_t parent_index,
    const HarmonicContext& context,
    const HarmonicOptions& options,
    const std::vector<HarmonicRatio>& ratios) {
  const auto harmonic_ratio =
      nearest_harmonic_ratio(child.best.peak.motion.frequency_hz /
                                 parent.best.peak.motion.frequency_hz,
                             ratios);
  if (!harmonic_ratio) {
    return std::nullopt;
  }

  const double expected_frequency =
      parent.best.peak.motion.frequency_hz * harmonic_ratio->value;
  const double frequency_error_bins =
      std::abs(child.best.peak.motion.frequency_hz - expected_frequency) *
      context.observation_seconds;
  if (frequency_error_bins > options.frequency_tolerance_bins) {
    return std::nullopt;
  }

  const double fast_duty_cycle =
      child.best.peak.motion.frequency_hz >= parent.best.peak.motion.frequency_hz
          ? child.best.peak.duty_cycle
          : parent.best.peak.duty_cycle;
  const double phase_distance = frequency_error_bins / fast_duty_cycle;
  if (phase_distance > options.phase_distance_max) {
    return std::nullopt;
  }

  const double distance = dm_distance(parent, child, context);
  if (distance > options.dm_distance_max) {
    return std::nullopt;
  }

  const double expected_snr =
      static_cast<double>(parent.best.peak.snr) /
      std::sqrt(static_cast<double>(harmonic_ratio->numerator) *
                static_cast<double>(harmonic_ratio->denominator));
  const double snr_distance =
      std::abs(static_cast<double>(child.best.peak.snr) - expected_snr);
  if (options.use_snr_consistency &&
      snr_distance > options.snr_distance_max) {
    return std::nullopt;
  }

  return HarmonicRelation{
      .is_harmonic = true,
      .parent_index = parent_index,
      .numerator = harmonic_ratio->numerator,
      .denominator = harmonic_ratio->denominator,
      .frequency_error_bins = frequency_error_bins,
      .phase_distance = phase_distance,
      .dm_distance = distance,
      .expected_snr = expected_snr,
      .snr_distance = snr_distance,
  };
}

}  // namespace

std::vector<HarmonicCandidate> flag_harmonics_cpu(
    std::span<const Candidate> candidates,
    const HarmonicContext& context,
    const HarmonicOptions& options) {
  validate_context(context);
  validate_options(options);
  if (candidates.empty()) {
    return {};
  }

  std::vector<HarmonicCandidate> flagged;
  flagged.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    validate_candidate(candidate);
    flagged.push_back(HarmonicCandidate{.candidate = candidate});
  }

  std::sort(flagged.begin(), flagged.end(),
            [](const HarmonicCandidate& lhs,
               const HarmonicCandidate& rhs) {
              return is_better_candidate_for_harmonic(lhs.candidate,
                                                      rhs.candidate);
            });

  const std::vector<HarmonicRatio> ratios = make_harmonic_ratios(options);

  for (std::size_t parent_index = 0; parent_index < flagged.size();
       ++parent_index) {
    if (flagged[parent_index].harmonic.is_harmonic) {
      continue;
    }
    const Candidate& parent = flagged[parent_index].candidate;
    for (std::size_t child_index = parent_index + 1;
         child_index < flagged.size(); ++child_index) {
      if (flagged[child_index].harmonic.is_harmonic) {
        continue;
      }
      const Candidate& child = flagged[child_index].candidate;
      auto relation = test_harmonic_relation(parent, child, parent_index,
                                             context, options, ratios);
      if (relation) {
        flagged[child_index].harmonic = *relation;
      }
    }
  }

  return flagged;
}

std::vector<Candidate> remove_harmonics_cpu(
    std::span<const HarmonicCandidate> candidates,
    std::size_t max_candidates) {
  std::vector<Candidate> selected;
  selected.reserve(candidates.size());
  for (const HarmonicCandidate& candidate : candidates) {
    if (candidate.harmonic.is_harmonic) {
      continue;
    }
    selected.push_back(candidate.candidate);
    if (max_candidates != 0 && selected.size() >= max_candidates) {
      break;
    }
  }
  return selected;
}

}  // namespace gaffa
