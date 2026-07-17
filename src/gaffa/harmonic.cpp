#include "gaffa/harmonic.h"

#include "gaffa/periodic_match.h"

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

bool is_better_candidate(const Candidate& lhs, const Candidate& rhs) {
  if (lhs.best.peak.snr != rhs.best.peak.snr) {
    return lhs.best.peak.snr > rhs.best.peak.snr;
  }
  if (lhs.member_count != rhs.member_count) {
    return lhs.member_count > rhs.member_count;
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
  if (options.frequency_tolerance_bins &&
      (*options.frequency_tolerance_bins < 0.0 ||
       !std::isfinite(*options.frequency_tolerance_bins))) {
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

void validate_candidate_set(const CandidateSet& candidate_set) {
  for (const Candidate& candidate : candidate_set.candidates) {
    const DmPeak& best = candidate.best;
    if (!std::isfinite(best.dm) || !std::isfinite(best.peak.duty_cycle) ||
        !std::isfinite(best.peak.snr)) {
      throw std::invalid_argument(
          "Harmonic candidates must contain finite scientific values");
    }
    validate_periodic_motion(best.peak.motion);
    if (!(best.peak.duty_cycle > 0.0) || !(best.peak.snr >= 0.0F)) {
      throw std::invalid_argument(
          "Harmonic candidate duty_cycle or snr is invalid");
    }
    (void)candidate_set.members_of(candidate);
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
      const std::size_t divisor = std::gcd(numerator, denominator);
      const std::size_t reduced_numerator = numerator / divisor;
      const std::size_t reduced_denominator = denominator / divisor;
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
  auto consider = [&](std::vector<HarmonicRatio>::const_iterator current) {
    if (current == ratios.end()) {
      return;
    }
    if (!best || std::abs(current->value - ratio) <
                     std::abs(best->value - ratio)) {
      best = *current;
    }
  };
  consider(lower);
  if (lower != ratios.begin()) {
    consider(std::prev(lower));
  }
  return best;
}

double frequency_at_observation_start(const Candidate& candidate) {
  const PeriodicMotion& motion = candidate.best.peak.motion;
  return periodic_frequency_hz_at(motion, -motion.reference_time_seconds);
}

double pulse_width_seconds(const Candidate& candidate) {
  return candidate.best.peak.period_seconds() * candidate.best.peak.duty_cycle;
}

double dm_distance(const Candidate& parent,
                   const Candidate& child,
                   const HarmonicContext& context) {
  const double low =
      std::min(context.frequency_low_mhz, context.frequency_high_mhz);
  const double high =
      std::max(context.frequency_low_mhz, context.frequency_high_mhz);
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
  const double parent_frequency = frequency_at_observation_start(parent);
  const double child_frequency = frequency_at_observation_start(child);
  const auto ratio =
      nearest_harmonic_ratio(child_frequency / parent_frequency, ratios);
  if (!ratio) {
    return std::nullopt;
  }

  const PhaseDrift phase_drift = harmonic_phase_drift(
      parent.best.peak.motion, child.best.peak.motion, ratio->value,
      context.observation_seconds);
  const FrequencyDrift frequency_drift = harmonic_frequency_drift(
      parent.best.peak.motion, child.best.peak.motion, ratio->value,
      context.observation_seconds);
  const double frequency_error_bins =
      frequency_drift.maximum_hz * context.observation_seconds;
  if (options.frequency_tolerance_bins &&
      frequency_error_bins > *options.frequency_tolerance_bins) {
    return std::nullopt;
  }

  const double fast_duty_cycle = child_frequency >= parent_frequency
                                     ? child.best.peak.duty_cycle
                                     : parent.best.peak.duty_cycle;
  const double phase_distance =
      phase_drift.maximum_cycles / fast_duty_cycle;
  if (phase_distance > options.phase_distance_max) {
    return std::nullopt;
  }

  const double distance = dm_distance(parent, child, context);
  if (distance > options.dm_distance_max) {
    return std::nullopt;
  }

  const double expected_snr =
      static_cast<double>(parent.best.peak.snr) /
      std::sqrt(static_cast<double>(ratio->numerator) *
                static_cast<double>(ratio->denominator));
  const double snr_distance =
      std::abs(static_cast<double>(child.best.peak.snr) - expected_snr);
  if (options.use_snr_consistency &&
      snr_distance > options.snr_distance_max) {
    return std::nullopt;
  }

  return HarmonicRelation{
      .is_harmonic = true,
      .parent_index = parent_index,
      .numerator = ratio->numerator,
      .denominator = ratio->denominator,
      .maximum_phase_drift_cycles = phase_drift.maximum_cycles,
      .phase_drift_time_seconds = phase_drift.time_seconds,
      .frequency_error_bins = frequency_error_bins,
      .phase_distance = phase_distance,
      .dm_distance = distance,
      .expected_snr = expected_snr,
      .snr_distance = snr_distance,
  };
}

}  // namespace

std::vector<HarmonicRelation> flag_harmonics_cpu(
    const CandidateSet& candidates,
    const HarmonicContext& context,
    const HarmonicOptions& options) {
  validate_context(context);
  validate_options(options);
  validate_candidate_set(candidates);

  std::vector<HarmonicRelation> relations(candidates.candidates.size());
  std::vector<std::size_t> order(candidates.candidates.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    return is_better_candidate(candidates.candidates[lhs],
                               candidates.candidates[rhs]);
  });

  const std::vector<HarmonicRatio> ratios = make_harmonic_ratios(options);
  for (std::size_t parent_rank = 0; parent_rank < order.size(); ++parent_rank) {
    const std::size_t parent_index = order[parent_rank];
    if (relations[parent_index].is_harmonic) {
      continue;
    }
    const Candidate& parent = candidates.candidates[parent_index];
    for (std::size_t child_rank = parent_rank + 1;
         child_rank < order.size(); ++child_rank) {
      const std::size_t child_index = order[child_rank];
      if (relations[child_index].is_harmonic) {
        continue;
      }
      const auto relation = test_harmonic_relation(
          parent, candidates.candidates[child_index], parent_index, context,
          options, ratios);
      if (relation) {
        relations[child_index] = *relation;
      }
    }
  }
  return relations;
}

std::vector<std::size_t> remove_harmonics_cpu(
    const CandidateSet& candidates,
    std::span<const HarmonicRelation> relations,
    std::size_t max_candidates) {
  if (relations.size() != candidates.candidates.size()) {
    throw std::invalid_argument(
        "Harmonic relation count must match candidate count");
  }
  std::vector<std::size_t> selected;
  selected.reserve(candidates.candidates.size());
  for (std::size_t index = 0; index < relations.size(); ++index) {
    if (relations[index].is_harmonic) {
      continue;
    }
    selected.push_back(index);
    if (max_candidates != 0 && selected.size() >= max_candidates) {
      break;
    }
  }
  return selected;
}

}  // namespace gaffa
