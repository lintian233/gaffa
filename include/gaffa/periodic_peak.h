#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

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
// the start of this backend's searched time interval. The frequency and
// line-of-sight acceleration, jerk, and snap are the direct search
// coordinates. Motion values use SI units.
struct PeriodicMotion {
  MotionOrder order = MotionOrder::Frequency;
  double reference_time_seconds = 0.0;
  double frequency_hz = 0.0;
  double acceleration_m_per_s2 = 0.0;
  double jerk_m_per_s3 = 0.0;
  double snap_m_per_s4 = 0.0;
};

// Validates the canonical representation of a periodic motion model. Every
// value must be finite, frequency_hz must be positive, and coefficients above
// order must be zero.
void validate_periodic_motion(const PeriodicMotion& motion);

// Returns the phase difference in cycles at offset_seconds relative to the
// motion reference epoch. This uses the first-order line-of-sight Doppler
// model shared by the current GAFFA and Loki periodic searches. The caller
// must pass a previously validated motion model.
[[nodiscard]] double periodic_phase_offset_cycles(
    const PeriodicMotion& motion,
    double offset_seconds) noexcept;

// Returns the instantaneous observed frequency at offset_seconds relative to
// the motion reference epoch under the same Doppler model. The caller must
// pass a previously validated motion model.
[[nodiscard]] double periodic_frequency_hz_at(
    const PeriodicMotion& motion,
    double offset_seconds) noexcept;

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
  // Physical dispersion measure in pc cm^-3.
  double dm = 0.0;
  // Stable identity of the source DM trial. This is not a physical distance.
  std::size_t dm_index = 0;
  PeriodicPeak peak{};
};

// Raw periodic-search responses from one backend or a batch of DM trials.
using DmPeaks = std::vector<DmPeak>;

// Identity of one series within a backend batch. The index is local to the
// current batch and carries no DM semantics until attach_dm_trials() is used.
struct SeriesPeak {
  std::size_t series_index = 0;
  PeriodicPeak peak{};
};

using SeriesPeaks = std::vector<SeriesPeak>;

// Non-owning DM coordinates for one series batch. values[i] belongs to local
// series i; index_offset + i is its unique trial identity. The index is not a
// physical DM coordinate or a distance measure.
struct DmTrialView {
  std::span<const double> values;
  std::size_t index_offset = 0;
};

void validate_dm_trials(DmTrialView trials);

// Attaches one dedispersion-trial identity to backend-neutral periodic peaks.
// The source peaks are copied because DmPeaks owns its values independently.
DmPeaks attach_dm_peaks(std::span<const PeriodicPeak> peaks,
                        double dm,
                        std::size_t dm_index);

// Attaches tile-local series peaks to their physical and global DM identities.
DmPeaks attach_dm_trials(std::span<const SeriesPeak> peaks,
                         DmTrialView trials);

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
