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
  // active observation (1 / duration). Loki uses its discrete search
  // coordinates and does not use this field.
  double frequency_tolerance_hz = 0.0;

  [[nodiscard]] bool enabled() const noexcept {
    return top_k_per_group != 0;
  }
};

}  // namespace gaffa
