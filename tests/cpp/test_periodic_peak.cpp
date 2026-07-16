#include "gaffa/periodic_peak.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

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
