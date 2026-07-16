#include "gaffa/candidate.h"

#include <gtest/gtest.h>

#include <cmath>
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
                           float snr,
                           std::size_t dm_index_min,
                           std::size_t dm_index_max) {
  const auto peak = dm_peak(dm, dm_index, frequency, 2, snr);
  return gaffa::Candidate{
      .best = peak,
      .peak_count = 1,
      .extent = {
          .dm_index_min = dm_index_min,
          .dm_index_max = dm_index_max,
          .frequency_hz = {.minimum = frequency, .maximum = frequency},
      },
  };
}

}  // namespace

TEST(CandidateSelection, EmptyInputReturnsEmpty) {
  const auto candidates =
      gaffa::select_candidates_cpu(std::span<const gaffa::DmPeak>{}, 100.0);

  EXPECT_TRUE(candidates.empty());
}

TEST(CandidateSelection, RejectsInvalidInputs) {
  const std::vector<gaffa::DmPeak> peaks{dm_peak(10.0, 0, 1.0, 1, 5.0F)};

  EXPECT_THROW((void)gaffa::select_candidates_cpu(peaks, 0.0),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::select_candidates_cpu(
                   peaks, 100.0,
                   gaffa::CandidateSelectionOptions{
                       .frequency_cluster_radius = -1.0,
                   }),
               std::invalid_argument);

  auto bad_peak = peaks.front();
  bad_peak.peak.motion.frequency_hz = INFINITY;
  EXPECT_THROW((void)gaffa::select_candidates_cpu(
                   std::vector<gaffa::DmPeak>{bad_peak}, 100.0),
               std::invalid_argument);

  bad_peak = peaks.front();
  bad_peak.peak.motion.frequency_hz = 0.0;
  EXPECT_THROW((void)gaffa::select_candidates_cpu(
                   std::vector<gaffa::DmPeak>{bad_peak}, 100.0),
               std::invalid_argument);

  bad_peak = peaks.front();
  bad_peak.peak.motion.acceleration_m_per_s2 = 1.0;
  EXPECT_THROW((void)gaffa::select_candidates_cpu(
                   std::vector<gaffa::DmPeak>{bad_peak}, 100.0),
               std::invalid_argument);

  bad_peak = peaks.front();
  bad_peak.peak.motion.order = gaffa::MotionOrder::Acceleration;
  bad_peak.peak.motion.acceleration_m_per_s2 = 1.0;
  EXPECT_THROW((void)gaffa::select_candidates_cpu(
                   std::vector<gaffa::DmPeak>{bad_peak}, 100.0),
               std::invalid_argument);
}

TEST(CandidateSelection, ClustersNearbyFrequencyAndDmPeaks) {
  const std::vector<gaffa::DmPeak> peaks{
      dm_peak(10.0, 0, 1.0000, 2, 7.0F),
      dm_peak(11.0, 1, 1.0005, 2, 10.0F),
      dm_peak(12.0, 2, 1.0009, 2, 8.0F),
  };

  const auto candidates = gaffa::select_candidates_cpu(
      peaks, 100.0,
      gaffa::CandidateSelectionOptions{
          .frequency_cluster_radius = 0.1,
          .dm_cluster_radius = 1,
      });

  ASSERT_EQ(candidates.size(), 1);
  EXPECT_EQ(candidates.front().peak_count, 3);
  EXPECT_EQ(candidates.front().best.dm_index, 1);
  EXPECT_EQ(candidates.front().extent.dm_index_min, 0);
  EXPECT_EQ(candidates.front().extent.dm_index_max, 2);
  EXPECT_FLOAT_EQ(candidates.front().best.peak.snr, 10.0F);
  EXPECT_DOUBLE_EQ(candidates.front().extent.frequency_hz.minimum, 1.0000);
  EXPECT_DOUBLE_EQ(candidates.front().extent.frequency_hz.maximum, 1.0009);
}

TEST(CandidateSelection, SeparatesFrequencyClustersOutsideRadius) {
  const std::vector<gaffa::DmPeak> peaks{
      dm_peak(10.0, 0, 1.0000, 2, 7.0F),
      dm_peak(10.0, 0, 1.0020, 2, 8.0F),
  };

  const auto candidates = gaffa::select_candidates_cpu(
      peaks, 100.0,
      gaffa::CandidateSelectionOptions{.frequency_cluster_radius = 0.1});

  ASSERT_EQ(candidates.size(), 2);
  EXPECT_FLOAT_EQ(candidates.front().best.peak.snr, 8.0F);
}

TEST(CandidateSelection, SeparatesDmClustersOutsideRadius) {
  const std::vector<gaffa::DmPeak> peaks{
      dm_peak(10.0, 0, 1.0000, 2, 7.0F),
      dm_peak(13.0, 3, 1.0005, 2, 9.0F),
  };

  const auto candidates = gaffa::select_candidates_cpu(
      peaks, 100.0,
      gaffa::CandidateSelectionOptions{
          .frequency_cluster_radius = 0.1,
          .dm_cluster_radius = 1,
      });

  ASSERT_EQ(candidates.size(), 2);
  EXPECT_FLOAT_EQ(candidates.front().best.peak.snr, 9.0F);
}

TEST(CandidateSelection, CanKeepWidthsSeparate) {
  const std::vector<gaffa::DmPeak> peaks{
      dm_peak(10.0, 0, 1.0000, 2, 7.0F),
      dm_peak(11.0, 1, 1.0005, 4, 9.0F),
  };

  const auto candidates = gaffa::select_candidates_cpu(
      peaks, 100.0,
      gaffa::CandidateSelectionOptions{
          .frequency_cluster_radius = 0.1,
          .dm_cluster_radius = 1,
          .cluster_across_widths = false,
      });

  ASSERT_EQ(candidates.size(), 2);
  EXPECT_FLOAT_EQ(candidates.front().best.peak.snr, 9.0F);
}

TEST(CandidateSelection, ClustersAcrossWidthsByDefault) {
  const std::vector<gaffa::DmPeak> peaks{
      dm_peak(10.0, 0, 1.0000, 2, 7.0F),
      dm_peak(11.0, 1, 1.0005, 4, 9.0F),
  };

  const auto candidates = gaffa::select_candidates_cpu(
      peaks, 100.0,
      gaffa::CandidateSelectionOptions{
          .frequency_cluster_radius = 0.1,
          .dm_cluster_radius = 1,
      });

  ASSERT_EQ(candidates.size(), 1);
  EXPECT_EQ(candidates.front().peak_count, 2);
  EXPECT_EQ(candidates.front().best.peak.boxcar_width_bins, 4);
}

TEST(CandidateSelection, AppliesFinalCandidateCapAfterSorting) {
  const std::vector<gaffa::DmPeak> peaks{
      dm_peak(10.0, 0, 1.000, 2, 7.0F),
      dm_peak(20.0, 4, 1.010, 2, 11.0F),
      dm_peak(30.0, 8, 1.020, 2, 9.0F),
  };

  const auto candidates = gaffa::select_candidates_cpu(
      peaks, 100.0,
      gaffa::CandidateSelectionOptions{
          .frequency_cluster_radius = 0.1,
          .max_candidates = 2,
      });

  ASSERT_EQ(candidates.size(), 2);
  EXPECT_FLOAT_EQ(candidates[0].best.peak.snr, 11.0F);
  EXPECT_FLOAT_EQ(candidates[1].best.peak.snr, 9.0F);
}
