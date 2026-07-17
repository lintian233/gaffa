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

gaffa::DmPeakGroups group(std::vector<gaffa::DmPeak> peaks) {
  return gaffa::group_dm_peaks_cpu(peaks, 100.0);
}

gaffa::CandidateSet cluster(std::vector<gaffa::DmPeakGroups> groups,
                            gaffa::CandidateClusteringOptions options = {}) {
  return gaffa::cluster_dm_peak_groups_cpu(groups, 100.0, options);
}

}  // namespace

TEST(CandidateClustering, EmptyInputReturnsEmpty) {
  const auto result = cluster({});
  EXPECT_TRUE(result.members.empty());
  EXPECT_TRUE(result.candidates.empty());
}

TEST(CandidateClustering, RejectsInvalidInputs) {
  auto peaks = group({dm_peak(10.0, 0, 1.0, 1, 5.0F)});
  EXPECT_THROW((void)gaffa::cluster_dm_peak_groups_cpu(
                   std::vector<gaffa::DmPeakGroups>{peaks}, 0.0),
               std::invalid_argument);
  EXPECT_THROW((void)cluster(
                   {peaks},
                   {.max_phase_distance_cycles = -1.0}),
               std::invalid_argument);

  auto bad = peaks;
  bad.members.front().peak.motion.frequency_hz = INFINITY;
  EXPECT_THROW((void)cluster(std::vector<gaffa::DmPeakGroups>{bad}),
               std::invalid_argument);
}

TEST(CandidateClustering, ClustersNearbyTrajectoryAcrossDmTrials) {
  const auto result = cluster({
      group({dm_peak(10.0, 0, 1.0000, 2, 7.0F)}),
      group({dm_peak(11.0, 1, 1.0005, 2, 10.0F)}),
      group({dm_peak(12.0, 2, 1.0009, 2, 8.0F)}),
  });

  ASSERT_EQ(result.candidates.size(), 1);
  const auto& candidate = result.candidates.front();
  EXPECT_EQ(candidate.member_count, 3);
  EXPECT_EQ(candidate.best.dm_index, 1);
  EXPECT_EQ(candidate.extent.dm_index_min, 0);
  EXPECT_EQ(candidate.extent.dm_index_max, 2);
  EXPECT_EQ(result.members_of(candidate).size(), 3);
}

TEST(CandidateClustering, PreservesEveryRawMemberOfLinkedGroups) {
  const auto result = cluster({
      group({dm_peak(10.0, 0, 1.0000, 2, 7.0F),
             dm_peak(10.0, 0, 1.0001, 4, 6.0F)}),
      group({dm_peak(11.0, 1, 1.0002, 2, 9.0F)}),
  });

  ASSERT_EQ(result.candidates.size(), 1);
  EXPECT_EQ(result.candidates.front().member_count, 3);
  EXPECT_EQ(result.members.size(), 3);
}

TEST(CandidateClustering, SeparatesOutsideTrajectoryOrDmRadius) {
  const auto result = cluster({
      group({dm_peak(10.0, 0, 1.0000, 2, 7.0F)}),
      group({dm_peak(10.0, 1, 1.0020, 2, 8.0F)}),
      group({dm_peak(13.0, 3, 1.0005, 2, 9.0F)}),
  });

  EXPECT_EQ(result.candidates.size(), 3);
  EXPECT_EQ(result.members.size(), 3);
}

TEST(CandidateClustering, WidthPolicyIsExplicit) {
  const std::vector<gaffa::DmPeakGroups> groups{
      group({dm_peak(10.0, 0, 1.0000, 2, 7.0F)}),
      group({dm_peak(11.0, 1, 1.0005, 4, 9.0F)}),
  };
  const auto separate = cluster(
      groups, {.cluster_across_widths = false});
  const auto merged = cluster(groups);

  EXPECT_EQ(separate.candidates.size(), 2);
  ASSERT_EQ(merged.candidates.size(), 1);
  EXPECT_EQ(merged.candidates.front().best.peak.boxcar_width_bins, 4);
}

TEST(CandidateClustering, SupportsTaylorMotion) {
  auto first = dm_peak(10.0, 0, 10.0, 2, 7.0F);
  first.peak.motion = {
      .order = gaffa::MotionOrder::Jerk,
      .reference_time_seconds = 50.0,
      .frequency_hz = 10.0,
      .acceleration_m_per_s2 = 2.0,
      .jerk_m_per_s3 = 0.1,
  };
  auto second = first;
  second.dm = 11.0;
  second.dm_index = 1;
  second.peak.snr = 9.0F;

  const auto result = cluster({group({first}), group({second})});
  ASSERT_EQ(result.candidates.size(), 1);
  EXPECT_EQ(result.members.size(), 2);
}

TEST(CandidateSelection, AppliesThresholdAndFinalCapWithoutCopying) {
  const auto clustered = cluster({
      group({dm_peak(10.0, 0, 1.000, 2, 7.0F)}),
      group({dm_peak(20.0, 4, 1.010, 2, 11.0F)}),
      group({dm_peak(30.0, 8, 1.020, 2, 9.0F)}),
  });
  const auto selected = gaffa::select_candidates_cpu(
      clustered,
      gaffa::CandidateSelectionOptions{.snr_min = 8.0F, .max_candidates = 2});

  ASSERT_EQ(selected.size(), 2);
  EXPECT_FLOAT_EQ(clustered.candidates[selected[0]].best.peak.snr, 11.0F);
  EXPECT_FLOAT_EQ(clustered.candidates[selected[1]].best.peak.snr, 9.0F);
}
