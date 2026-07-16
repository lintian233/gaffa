#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace gaffa {

// Inclusive numeric range for a physical search parameter or candidate extent.
struct ValueRange {
  double minimum = 0.0;
  double maximum = 0.0;
};

// The highest-order line-of-sight motion term included in a periodic search.
// Each order includes every lower-order term.
enum class MotionOrder : std::uint8_t {
  Frequency,
  Acceleration,
  Jerk,
  Snap,
};

// Periodic signal parameters at reference_time_seconds, measured relative to
// the start of the searched time interval. Motion values use SI units.
struct PeriodicMotion {
  MotionOrder order = MotionOrder::Frequency;
  double reference_time_seconds = 0.0;
  double frequency_hz = 0.0;
  double acceleration_m_per_s2 = 0.0;
  double jerk_m_per_s3 = 0.0;
  double snap_m_per_s4 = 0.0;
};

// One significant periodic-search detection. phase_bin is absent when a
// backend scores a folded profile but does not report its maximizing phase.
struct PeriodicPeak {
  PeriodicMotion motion{};
  std::optional<std::size_t> phase_bin{};
  std::size_t phase_bins = 0;
  std::size_t boxcar_width_bins = 0;
  double duty_cycle = 0.0;
  float snr = 0.0F;

  [[nodiscard]] double period_seconds() const noexcept {
    return 1.0 / motion.frequency_hz;
  }
};

// A periodic peak associated with one dedispersion trial.
struct DmPeak {
  double dm = 0.0;
  std::size_t dm_index = 0;
  PeriodicPeak peak{};
};

struct MotionRange {
  ValueRange acceleration_m_per_s2{};
  ValueRange jerk_m_per_s3{};
  ValueRange snap_m_per_s4{};
};

struct CandidateExtent {
  std::size_t dm_index_min = 0;
  std::size_t dm_index_max = 0;
  ValueRange frequency_hz{};
  MotionRange motion{};
};

}  // namespace gaffa
