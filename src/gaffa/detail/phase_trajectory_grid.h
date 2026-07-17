#pragma once

#include "gaffa/periodic_peak.h"

#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

namespace gaffa::detail {

using PhaseTrajectory = std::array<double, 4>;
using PhaseTrajectoryCell = std::array<std::int64_t, 4>;

struct PhaseTrajectoryCellHash {
  std::size_t operator()(const PhaseTrajectoryCell& cell) const noexcept {
    std::size_t result = 0;
    for (const std::int64_t coordinate : cell) {
      const std::size_t value = std::hash<std::int64_t>{}(coordinate);
      result ^= value + 0x9e3779b97f4a7c15ULL + (result << 6U) +
                (result >> 2U);
    }
    return result;
  }
};

inline double anchored_phase(const PeriodicMotion& motion,
                             double time_seconds) {
  const double start_offset = -motion.reference_time_seconds;
  return periodic_phase_offset_cycles(motion, time_seconds + start_offset) -
         periodic_phase_offset_cycles(motion, start_offset);
}

inline PhaseTrajectory make_phase_trajectory(
    const PeriodicMotion& motion,
    double searched_duration_seconds) {
  return {
      anchored_phase(motion, searched_duration_seconds * 0.25),
      anchored_phase(motion, searched_duration_seconds * 0.50),
      anchored_phase(motion, searched_duration_seconds * 0.75),
      anchored_phase(motion, searched_duration_seconds),
  };
}

inline double phase_trajectory_cell_tolerance(double radius) {
  return 64.0 * std::numeric_limits<double>::epsilon() *
         std::max(1.0, radius);
}

inline bool make_phase_trajectory_cell(const PhaseTrajectory& trajectory,
                                       double cell_width,
                                       PhaseTrajectoryCell& cell) {
  const long double lower =
      static_cast<long double>(std::numeric_limits<std::int64_t>::min() + 1);
  const long double upper =
      static_cast<long double>(std::numeric_limits<std::int64_t>::max() - 1);
  for (std::size_t index = 0; index < trajectory.size(); ++index) {
    const long double scaled =
        std::floor(static_cast<long double>(trajectory[index]) / cell_width);
    if (!std::isfinite(scaled) || scaled < lower || scaled > upper) {
      return false;
    }
    cell[index] = static_cast<std::int64_t>(scaled);
  }
  return true;
}

inline bool offset_phase_trajectory_cell(const PhaseTrajectoryCell& cell,
                                         int offset0,
                                         int offset1,
                                         int offset2,
                                         int offset3,
                                         PhaseTrajectoryCell& result) {
  const std::array<int, 4> offsets{offset0, offset1, offset2, offset3};
  for (std::size_t index = 0; index < cell.size(); ++index) {
    const std::int64_t offset = offsets[index];
    if ((offset > 0 && cell[index] >
                           std::numeric_limits<std::int64_t>::max() - offset) ||
        (offset < 0 && cell[index] <
                           std::numeric_limits<std::int64_t>::min() - offset)) {
      return false;
    }
    result[index] = cell[index] + offset;
  }
  return true;
}

// Invokes callback for every cell that can contain a trajectory whose sampled
// phase coordinates differ by at most one cell width from center.
template <typename Callback>
inline void for_each_neighbor_phase_trajectory_cell(
    const PhaseTrajectoryCell& center,
    Callback&& callback) {
  for (int offset0 = -1; offset0 <= 1; ++offset0) {
    for (int offset1 = -1; offset1 <= 1; ++offset1) {
      for (int offset2 = -1; offset2 <= 1; ++offset2) {
        for (int offset3 = -1; offset3 <= 1; ++offset3) {
          PhaseTrajectoryCell neighbor{};
          if (offset_phase_trajectory_cell(center, offset0, offset1, offset2,
                                           offset3, neighbor)) {
            callback(neighbor);
          }
        }
      }
    }
  }
}

}  // namespace gaffa::detail
