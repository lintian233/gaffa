#pragma once

#include "gaffa/candidate.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
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
  std::size_t denominator_max = 5;
  // Optional diagnostic guard in Fourier-bin units. Exact phase drift remains
  // the primary trajectory-consistency condition when this is unset.
  std::optional<double> frequency_tolerance_bins{};
  double phase_distance_max = 1.0;
  double dm_distance_max = 3.0;
  bool use_snr_consistency = false;
  double snr_distance_max = 3.0;
};

// One entry per CandidateSet::candidates element. parent_index always refers
// to that same candidate array, never to an internally sorted temporary.
struct HarmonicRelation {
  bool is_harmonic = false;
  std::size_t parent_index = kNoHarmonicParent;
  std::int32_t numerator = 1;
  std::int32_t denominator = 1;
  double maximum_phase_drift_cycles = 0.0;
  double phase_drift_time_seconds = 0.0;
  double frequency_error_bins = 0.0;
  double phase_distance = 0.0;
  double dm_distance = 0.0;
  double expected_snr = 0.0;
  double snr_distance = 0.0;
};

std::vector<HarmonicRelation> flag_harmonics_cpu(
    const CandidateSet& candidates,
    const HarmonicContext& context,
    const HarmonicOptions& options = {});

// Returns ranked indices into CandidateSet::candidates after excluding flagged
// harmonics. No Candidate or raw DmPeak is copied.
std::vector<std::size_t> remove_harmonics_cpu(
    const CandidateSet& candidates,
    std::span<const HarmonicRelation> relations,
    std::size_t max_candidates = 0);

}  // namespace gaffa
