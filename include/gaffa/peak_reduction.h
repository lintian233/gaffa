#pragma once

#include <cstddef>

namespace gaffa {

// Bounded, intentionally lossy reduction applied after backend detection.
// The final CPU candidate stage remains the scientific clustering authority.
struct PeakReductionOptions {
  // Zero disables backend reduction and preserves all detected peaks.
  std::size_t top_k_per_group = 0;

  // Maximum number of non-empty coordinate groups retained for one input
  // series. Zero is invalid when top_k_per_group is non-zero.
  std::size_t max_groups_per_series = 0;

  // Native FFA frequency bucket width. Zero selects one Fourier bin of the
  // active observation (1 / duration). Loki does not use this field.
  double frequency_tolerance_hz = 0.0;

  // Loki phase-cell width in cycles over the searched observation. Zero uses
  // one exact Loki search coordinate per group. A positive value creates a
  // conservative, backend-local phase-cell key; it is a lossy pre-reduction
  // hint, not the final scientific candidate equivalence test. Native FFA
  // does not use this field.
  double phase_tolerance_cycles = 0.0;

  [[nodiscard]] bool enabled() const noexcept {
    return top_k_per_group != 0;
  }
};

}  // namespace gaffa
