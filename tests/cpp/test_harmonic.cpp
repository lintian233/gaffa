#include "gaffa/harmonic.h"

#include <gtest/gtest.h>

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

gaffa::DmPeak dm_peak(double dm,
                      std::size_t dm_index,
                      double frequency,
                      std::size_t width,
                      float snr) {
  return gaffa::DmPeak{
      .dm = dm,
      .dm_index = dm_index,
      .peak = {
          .motion = {.frequency_hz = frequency},
          .phase_bin = 0,
          .phase_bins = 100,
          .boxcar_width_bins = width,
          .duty_cycle = static_cast<double>(width) / 100.0,
          .snr = snr,
      },
  };
}

gaffa::Candidate candidate(double dm,
                           std::size_t dm_index,
                           double frequency,
                           float snr) {
  const auto peak = dm_peak(dm, dm_index, frequency, 2, snr);
  return gaffa::Candidate{
      .best = peak,
      .peak_count = 1,
      .extent = {
          .dm_index_min = dm_index,
          .dm_index_max = dm_index,
          .frequency_hz = {.minimum = frequency, .maximum = frequency},
      },
  };
}

gaffa::HarmonicContext context() {
  return gaffa::HarmonicContext{
      .observation_seconds = 100.0,
      .frequency_low_mhz = 1000.0,
      .frequency_high_mhz = 1500.0,
  };
}

}  // namespace

TEST(HarmonicFlagging, EmptyInputReturnsEmpty) {
  const auto flagged = gaffa::flag_harmonics_cpu(
      std::span<const gaffa::Candidate>{}, context());

  EXPECT_TRUE(flagged.empty());
}

TEST(HarmonicFlagging, RejectsInvalidInputs) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 5.0F)};

  EXPECT_THROW((void)gaffa::flag_harmonics_cpu(
                   candidates,
                   gaffa::HarmonicContext{.observation_seconds = 0.0,
                                          .frequency_low_mhz = 1000.0,
                                          .frequency_high_mhz = 1500.0}),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::flag_harmonics_cpu(
                   candidates, context(),
                   gaffa::HarmonicOptions{.max_harmonic = 1}),
               std::invalid_argument);

  auto bad_candidate = candidates.front();
  bad_candidate.best.peak.motion.frequency_hz = INFINITY;
  EXPECT_THROW((void)gaffa::flag_harmonics_cpu(
                   std::vector<gaffa::Candidate>{bad_candidate}, context()),
               std::invalid_argument);

  bad_candidate = candidates.front();
  bad_candidate.best.peak.motion.acceleration_m_per_s2 = INFINITY;
  EXPECT_THROW((void)gaffa::flag_harmonics_cpu(
                   std::vector<gaffa::Candidate>{bad_candidate}, context()),
               std::invalid_argument);

  bad_candidate = candidates.front();
  bad_candidate.best.peak.motion.acceleration_m_per_s2 = 1.0;
  EXPECT_THROW((void)gaffa::flag_harmonics_cpu(
                   std::vector<gaffa::Candidate>{bad_candidate}, context()),
               std::invalid_argument);

  bad_candidate = candidates.front();
  bad_candidate.best.peak.motion.order = gaffa::MotionOrder::Acceleration;
  bad_candidate.best.peak.motion.acceleration_m_per_s2 = 1.0;
  EXPECT_THROW((void)gaffa::flag_harmonics_cpu(
                   std::vector<gaffa::Candidate>{bad_candidate}, context()),
               std::invalid_argument);
}

TEST(HarmonicFlagging, FlagsIntegerHarmonicFromStrongerCandidate) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 30.0F),
      candidate(10.0, 0, 2.0, 20.0F),
  };

  const auto flagged = gaffa::flag_harmonics_cpu(candidates, context());

  ASSERT_EQ(flagged.size(), 2);
  EXPECT_FALSE(flagged[0].harmonic.is_harmonic);
  ASSERT_TRUE(flagged[1].harmonic.is_harmonic);
  EXPECT_EQ(flagged[1].harmonic.parent_index, 0);
  EXPECT_EQ(flagged[1].harmonic.numerator, 2);
  EXPECT_EQ(flagged[1].harmonic.denominator, 1);
  EXPECT_DOUBLE_EQ(flagged[1].harmonic.frequency_error_bins, 0.0);
}

TEST(HarmonicFlagging, FlagsFractionalHarmonic) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 30.0F),
      candidate(10.0, 0, 1.5, 20.0F),
  };

  const auto flagged = gaffa::flag_harmonics_cpu(
      candidates, context(),
      gaffa::HarmonicOptions{
          .max_harmonic = 16,
          .denominator_max = 4,
      });

  ASSERT_TRUE(flagged[1].harmonic.is_harmonic);
  EXPECT_EQ(flagged[1].harmonic.numerator, 3);
  EXPECT_EQ(flagged[1].harmonic.denominator, 2);
}

TEST(HarmonicFlagging, DefaultOptionsIgnoreHighDenominatorRatios) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 30.0F),
      candidate(10.0, 0, 10.0 / 9.0, 20.0F),
  };

  const auto flagged = gaffa::flag_harmonics_cpu(candidates, context());

  EXPECT_FALSE(flagged[1].harmonic.is_harmonic);
}

TEST(HarmonicFlagging, KeepsCandidateOutsideFrequencyTolerance) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 30.0F),
      candidate(10.0, 0, 2.02, 20.0F),
  };

  const auto flagged = gaffa::flag_harmonics_cpu(candidates, context());

  EXPECT_FALSE(flagged[1].harmonic.is_harmonic);
}

TEST(HarmonicFlagging, KeepsCandidateOutsidePhaseDistance) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 30.0F),
      candidate(10.0, 0, 2.01, 20.0F),
  };

  const auto flagged = gaffa::flag_harmonics_cpu(
      candidates, context(),
      gaffa::HarmonicOptions{
          .max_harmonic = 16,
          .denominator_max = 5,
          .frequency_tolerance_bins = 1.5,
          .phase_distance_max = 1.0,
      });

  EXPECT_FALSE(flagged[1].harmonic.is_harmonic);
}

TEST(HarmonicFlagging, KeepsCandidateOutsidePhysicalDmDistance) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 30.0F),
      candidate(1000010.0, 100, 2.0, 20.0F),
  };

  const auto flagged = gaffa::flag_harmonics_cpu(candidates, context());

  EXPECT_FALSE(flagged[1].harmonic.is_harmonic);
}

TEST(HarmonicFlagging, DoesNotLetFlaggedCandidateBecomeParent) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 30.0F),
      candidate(10.0, 0, 1.5, 20.0F),
      candidate(10.0, 0, 2.25, 10.0F),
  };

  const auto flagged = gaffa::flag_harmonics_cpu(
      candidates, context(),
      gaffa::HarmonicOptions{
          .max_harmonic = 3,
          .denominator_max = 2,
      });

  ASSERT_EQ(flagged.size(), 3);
  EXPECT_TRUE(flagged[1].harmonic.is_harmonic);
  EXPECT_FALSE(flagged[2].harmonic.is_harmonic);
}

TEST(HarmonicFlagging, SnrConsistencyIsDiagnosticByDefault) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 30.0F),
      candidate(10.0, 0, 2.0, 1.0F),
  };

  const auto flagged = gaffa::flag_harmonics_cpu(candidates, context());

  ASSERT_TRUE(flagged[1].harmonic.is_harmonic);
  EXPECT_GT(flagged[1].harmonic.snr_distance, 3.0);
}

TEST(HarmonicFlagging, CanUseSnrConsistencyAsHardCondition) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 30.0F),
      candidate(10.0, 0, 2.0, 1.0F),
  };

  const auto flagged = gaffa::flag_harmonics_cpu(
      candidates, context(),
      gaffa::HarmonicOptions{
          .max_harmonic = 16,
          .denominator_max = 16,
          .frequency_tolerance_bins = 1.5,
          .dm_distance_max = 3.0,
          .use_snr_consistency = true,
          .snr_distance_max = 3.0,
      });

  EXPECT_FALSE(flagged[1].harmonic.is_harmonic);
}

TEST(HarmonicFiltering, RemovesFlaggedHarmonicsAndAppliesCap) {
  const std::vector<gaffa::Candidate> candidates{
      candidate(10.0, 0, 1.0, 30.0F),
      candidate(10.0, 0, 2.0, 20.0F),
      candidate(10.0, 0, 1.7, 15.0F),
  };

  const auto flagged = gaffa::flag_harmonics_cpu(candidates, context());
  const auto filtered = gaffa::remove_harmonics_cpu(flagged, 1);

  ASSERT_EQ(filtered.size(), 1);
  EXPECT_DOUBLE_EQ(filtered.front().best.peak.motion.frequency_hz, 1.0);
}
