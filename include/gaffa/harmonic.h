#pragma once

#include "gaffa/candidate.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace gaffa {

inline constexpr std::size_t kNoHarmonicParent =
    std::numeric_limits<std::size_t>::max();

struct HarmonicContext {
  double observation_seconds = 0.0;
  double frequency_low_mhz = 0.0;
  double frequency_high_mhz = 0.0;
};

struct HarmonicOptions {
  std::size_t max_harmonic = 16;
  // Keep the default conservative: low-order fractional harmonics are useful
  // diagnostics, while high-denominator ratios can turn ordinary nearby
  // candidates into apparent harmonics.
  std::size_t denominator_max = 5;
  double frequency_tolerance_bins = 1.5;
  double phase_distance_max = 1.0;
  double dm_distance_max = 3.0;
  bool use_snr_consistency = false;
  double snr_distance_max = 3.0;
};

struct HarmonicRelation {
  bool is_harmonic = false;
  std::size_t parent_index = kNoHarmonicParent;
  std::int32_t numerator = 1;
  std::int32_t denominator = 1;
  double frequency_error_bins = 0.0;
  double phase_distance = 0.0;
  double dm_distance = 0.0;
  double expected_snr = 0.0;
  double snr_distance = 0.0;
};

struct HarmonicCandidate {
  Candidate candidate;
  HarmonicRelation harmonic;
};

std::vector<HarmonicCandidate> flag_harmonics_cpu(
    std::span<const Candidate> candidates,
    const HarmonicContext& context,
    const HarmonicOptions& options = {});

std::vector<Candidate> remove_harmonics_cpu(
    std::span<const HarmonicCandidate> candidates,
    std::size_t max_candidates = 0);

}  // namespace gaffa
