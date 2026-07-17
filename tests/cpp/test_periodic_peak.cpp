#include "gaffa/periodic_peak.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

constexpr double kSpeedOfLightMPerS = 299792458.0;

}  // namespace

TEST(PeriodicMotion, AcceptsCanonicalOrders) {
  EXPECT_NO_THROW(gaffa::validate_periodic_motion({.frequency_hz = 1.0}));
  EXPECT_NO_THROW(gaffa::validate_periodic_motion({
      .order = gaffa::MotionOrder::Acceleration,
      .frequency_hz = 1.0,
      .acceleration_m_per_s2 = 2.0,
  }));
  EXPECT_NO_THROW(gaffa::validate_periodic_motion({
      .order = gaffa::MotionOrder::Jerk,
      .frequency_hz = 1.0,
      .acceleration_m_per_s2 = 2.0,
      .jerk_m_per_s3 = 3.0,
  }));
  EXPECT_NO_THROW(gaffa::validate_periodic_motion({
      .order = gaffa::MotionOrder::Snap,
      .frequency_hz = 1.0,
      .acceleration_m_per_s2 = 2.0,
      .jerk_m_per_s3 = 3.0,
      .snap_m_per_s4 = 4.0,
  }));
}

TEST(PeriodicMotion, RejectsTermsAboveDeclaredOrder) {
  EXPECT_THROW(gaffa::validate_periodic_motion({
                   .frequency_hz = 1.0,
                   .acceleration_m_per_s2 = 2.0,
               }),
               std::invalid_argument);
  EXPECT_THROW(gaffa::validate_periodic_motion({
                   .order = gaffa::MotionOrder::Acceleration,
                   .frequency_hz = 1.0,
                   .jerk_m_per_s3 = 3.0,
               }),
               std::invalid_argument);
  EXPECT_THROW(gaffa::validate_periodic_motion({
                   .order = gaffa::MotionOrder::Jerk,
                   .frequency_hz = 1.0,
                   .snap_m_per_s4 = 4.0,
               }),
               std::invalid_argument);
}

TEST(PeriodicMotion, RejectsInvalidFrequencyAndValues) {
  EXPECT_THROW(gaffa::validate_periodic_motion({.frequency_hz = 0.0}),
               std::invalid_argument);
  EXPECT_THROW(gaffa::validate_periodic_motion({
                   .frequency_hz = std::numeric_limits<double>::infinity(),
               }),
               std::invalid_argument);
  EXPECT_THROW(gaffa::validate_periodic_motion({
                   .order = static_cast<gaffa::MotionOrder>(99),
                   .frequency_hz = 1.0,
               }),
               std::invalid_argument);
}

TEST(PeriodicMotion, FrequencyOnlyPhaseAndFrequencyAreExact) {
  const gaffa::PeriodicMotion motion{.frequency_hz = 8.0};

  EXPECT_DOUBLE_EQ(gaffa::periodic_phase_offset_cycles(motion, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(gaffa::periodic_phase_offset_cycles(motion, 1.25), 10.0);
  EXPECT_DOUBLE_EQ(gaffa::periodic_frequency_hz_at(motion, 1.25), 8.0);
}

TEST(PeriodicMotion, DopplerPhaseMatchesAccelerationJerkAndSnapModel) {
  const gaffa::PeriodicMotion motion{
      .order = gaffa::MotionOrder::Snap,
      .frequency_hz = 100.0,
      .acceleration_m_per_s2 = 12.0,
      .jerk_m_per_s3 = -3.0,
      .snap_m_per_s4 = 0.5,
  };
  constexpr double dt = 2.0;
  const double displacement_m =
      0.5 * 12.0 * dt * dt + (-3.0 * dt * dt * dt) / 6.0 +
      (0.5 * dt * dt * dt * dt) / 24.0;
  const double velocity_m_per_s =
      12.0 * dt + 0.5 * -3.0 * dt * dt +
      (0.5 * dt * dt * dt) / 6.0;

  EXPECT_DOUBLE_EQ(
      gaffa::periodic_phase_offset_cycles(motion, dt),
      100.0 * (dt - displacement_m / kSpeedOfLightMPerS));
  EXPECT_DOUBLE_EQ(
      gaffa::periodic_frequency_hz_at(motion, dt),
      100.0 * (1.0 - velocity_m_per_s / kSpeedOfLightMPerS));
}

TEST(PeriodicMotion, PhaseDerivativeMatchesInstantaneousFrequency) {
  const gaffa::PeriodicMotion motion{
      .order = gaffa::MotionOrder::Jerk,
      .frequency_hz = 300.0,
      .acceleration_m_per_s2 = 20.0,
      .jerk_m_per_s3 = -4.0,
  };
  constexpr double dt = 3.0;
  constexpr double epsilon = 1.0e-5;
  const double numerical_frequency =
      (gaffa::periodic_phase_offset_cycles(motion, dt + epsilon) -
       gaffa::periodic_phase_offset_cycles(motion, dt - epsilon)) /
      (2.0 * epsilon);

  EXPECT_NEAR(numerical_frequency,
              gaffa::periodic_frequency_hz_at(motion, dt), 1.0e-7);
}
