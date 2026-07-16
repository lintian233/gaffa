#include "gaffa/periodic_peak.h"

#include <cmath>
#include <stdexcept>

namespace gaffa {

void validate_periodic_motion(const PeriodicMotion& motion) {
  if (!std::isfinite(motion.reference_time_seconds) ||
      !std::isfinite(motion.frequency_hz) ||
      !std::isfinite(motion.acceleration_m_per_s2) ||
      !std::isfinite(motion.jerk_m_per_s3) ||
      !std::isfinite(motion.snap_m_per_s4)) {
    throw std::invalid_argument(
        "Periodic motion values must be finite");
  }
  if (!(motion.frequency_hz > 0.0)) {
    throw std::invalid_argument(
        "Periodic motion frequency_hz must be > 0");
  }

  switch (motion.order) {
    case MotionOrder::Frequency:
      if (motion.acceleration_m_per_s2 != 0.0 ||
          motion.jerk_m_per_s3 != 0.0 || motion.snap_m_per_s4 != 0.0) {
        throw std::invalid_argument(
            "Frequency-only periodic motion must not contain higher-order terms");
      }
      return;
    case MotionOrder::Acceleration:
      if (motion.jerk_m_per_s3 != 0.0 || motion.snap_m_per_s4 != 0.0) {
        throw std::invalid_argument(
            "Acceleration periodic motion must not contain jerk or snap");
      }
      return;
    case MotionOrder::Jerk:
      if (motion.snap_m_per_s4 != 0.0) {
        throw std::invalid_argument(
            "Jerk periodic motion must not contain snap");
      }
      return;
    case MotionOrder::Snap:
      return;
  }

  throw std::invalid_argument("Periodic motion order is invalid");
}

}  // namespace gaffa
