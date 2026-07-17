#include "gaffa/periodic_match.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

TEST(PeriodicMatch, FrequencyOnlyDriftIsExact) {
  const gaffa::PeriodicMotion lhs{.frequency_hz = 10.0};
  const gaffa::PeriodicMotion rhs{.frequency_hz = 10.01};

  const auto drift = gaffa::periodic_phase_drift(lhs, rhs, 100.0);

  EXPECT_NEAR(drift.maximum_cycles, 1.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(drift.time_seconds, 100.0);
}

TEST(PeriodicMatch, AnchoringMakesFrequencyEpochIndependent) {
  const gaffa::PeriodicMotion lhs{
      .reference_time_seconds = 0.0,
      .frequency_hz = 10.0,
  };
  const gaffa::PeriodicMotion rhs{
      .reference_time_seconds = 50.0,
      .frequency_hz = 10.0,
  };

  EXPECT_DOUBLE_EQ(gaffa::periodic_phase_drift(lhs, rhs, 100.0)
                       .maximum_cycles,
                   0.0);
}

TEST(PeriodicMatch, FindsInteriorTaylorMaximum) {
  const gaffa::PeriodicMotion reference{.frequency_hz = 100.0};
  const gaffa::PeriodicMotion accelerated{
      .order = gaffa::MotionOrder::Acceleration,
      .reference_time_seconds = 50.0,
      .frequency_hz = 100.0,
      .acceleration_m_per_s2 = 100000.0,
  };

  const auto drift =
      gaffa::periodic_phase_drift(reference, accelerated, 100.0);

  EXPECT_NEAR(drift.time_seconds, 50.0, 1.0e-8);
  EXPECT_GT(drift.maximum_cycles, 0.4);
}

TEST(PeriodicMatch, HarmonicTrajectorySupportsMotionTerms) {
  const gaffa::PeriodicMotion parent{
      .order = gaffa::MotionOrder::Snap,
      .reference_time_seconds = 50.0,
      .frequency_hz = 10.0,
      .acceleration_m_per_s2 = 12.0,
      .jerk_m_per_s3 = 0.4,
      .snap_m_per_s4 = 0.02,
  };
  auto child = parent;
  child.frequency_hz = 20.0;

  const auto phase = gaffa::harmonic_phase_drift(parent, child, 2.0, 100.0);
  const auto frequency =
      gaffa::harmonic_frequency_drift(parent, child, 2.0, 100.0);

  EXPECT_NEAR(phase.maximum_cycles, 0.0, 1.0e-12);
  EXPECT_NEAR(frequency.maximum_hz, 0.0, 1.0e-12);
}

TEST(PeriodicMatch, SnapMaximumAgreesWithDenseReference) {
  const gaffa::PeriodicMotion lhs{
      .order = gaffa::MotionOrder::Snap,
      .reference_time_seconds = 37.0,
      .frequency_hz = 80.0,
      .acceleration_m_per_s2 = 1200.0,
      .jerk_m_per_s3 = -70.0,
      .snap_m_per_s4 = 3.0,
  };
  const gaffa::PeriodicMotion rhs{
      .order = gaffa::MotionOrder::Snap,
      .reference_time_seconds = 61.0,
      .frequency_hz = 80.0002,
      .acceleration_m_per_s2 = -800.0,
      .jerk_m_per_s3 = 40.0,
      .snap_m_per_s4 = -2.0,
  };
  constexpr double observation_seconds = 100.0;
  const auto exact =
      gaffa::periodic_phase_drift(lhs, rhs, observation_seconds);

  auto anchored_phase = [](const gaffa::PeriodicMotion& motion, double time) {
    const double start = -motion.reference_time_seconds;
    return gaffa::periodic_phase_offset_cycles(motion, start + time) -
           gaffa::periodic_phase_offset_cycles(motion, start);
  };
  double dense_maximum = 0.0;
  for (int index = 0; index <= 100000; ++index) {
    const double time = observation_seconds * index / 100000.0;
    dense_maximum = std::max(
        dense_maximum,
        std::abs(anchored_phase(lhs, time) - anchored_phase(rhs, time)));
  }

  EXPECT_GE(exact.maximum_cycles + 1.0e-12, dense_maximum);
  EXPECT_NEAR(exact.maximum_cycles, dense_maximum, 1.0e-8);
  EXPECT_GE(exact.time_seconds, 0.0);
  EXPECT_LE(exact.time_seconds, observation_seconds);
}

TEST(PeriodicMatch, RejectsInvalidObservationAndRatio) {
  const gaffa::PeriodicMotion motion{.frequency_hz = 10.0};
  EXPECT_THROW((void)gaffa::periodic_phase_drift(motion, motion, 0.0),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::harmonic_phase_drift(motion, motion, 0.0, 1.0),
               std::invalid_argument);
}
