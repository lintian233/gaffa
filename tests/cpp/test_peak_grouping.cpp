#include "gaffa/peak_grouping.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

gaffa::DmPeak peak(double frequency, std::size_t width, float snr) {
  return {
      .dm = 42.0,
      .dm_index = 3,
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

}  // namespace

TEST(DmPeakGrouping, RetainsEveryInputPeakInContiguousGroups) {
  const std::vector<gaffa::DmPeak> peaks{
      peak(1.0000, 2, 10.0F), peak(1.0001, 4, 8.0F), peak(1.0100, 2, 7.0F),
  };
  const auto result = gaffa::group_dm_peaks_cpu(peaks, 100.0);

  ASSERT_EQ(result.members.size(), peaks.size());
  ASSERT_EQ(result.groups.size(), 2);
  EXPECT_EQ(result.members_of(0).size() + result.members_of(1).size(),
            peaks.size());
  EXPECT_FLOAT_EQ(result.best_of(0).peak.snr, 10.0F);
}

TEST(DmPeakGrouping, ZeroDistanceKeepsSingletonGroups) {
  const std::vector<gaffa::DmPeak> peaks{
      peak(1.0, 2, 10.0F), peak(1.0, 4, 8.0F)};
  const auto result = gaffa::group_dm_peaks_cpu(
      peaks, 100.0,
      {.max_phase_distance_cycles = 0.0});
  ASSERT_EQ(result.groups.size(), 2);
  EXPECT_EQ(result.members_of(0).size(), 1);
  EXPECT_EQ(result.members_of(1).size(), 1);
}

TEST(DmPeakGrouping, WidthPolicyIsExplicit) {
  const std::vector<gaffa::DmPeak> peaks{
      peak(1.0000, 2, 10.0F), peak(1.0001, 4, 8.0F),
  };
  EXPECT_EQ(gaffa::group_dm_peaks_cpu(peaks, 100.0).groups.size(), 1);
  EXPECT_EQ(gaffa::group_dm_peaks_cpu(
                peaks, 100.0, {.merge_widths = false})
                .groups.size(),
            2);
}

TEST(DmPeakGrouping, RejectsMixedDmTrials) {
  auto second = peak(1.0, 2, 8.0F);
  second.dm_index = 4;
  const std::vector<gaffa::DmPeak> peaks{peak(1.0, 2, 10.0F), second};
  EXPECT_THROW((void)gaffa::group_dm_peaks_cpu(
                   peaks, 100.0),
               std::invalid_argument);
}

TEST(DmPeakGrouping, BatchPartitionsByDmTrialInDmIndexOrder) {
  auto first = peak(1.0, 2, 10.0F);
  first.dm = 20.0;
  first.dm_index = 2;
  auto second = peak(2.0, 2, 9.0F);
  second.dm = 10.0;
  second.dm_index = 1;
  auto third = second;
  third.peak.snr = 8.0F;

  const auto result = gaffa::group_dm_peak_batch_cpu(
      std::vector<gaffa::DmPeak>{first, second, third}, 100.0);

  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].best_of(0).dm_index, 1);
  EXPECT_EQ(result[0].members.size(), 2);
  EXPECT_EQ(result[1].best_of(0).dm_index, 2);
  EXPECT_EQ(result[1].members.size(), 1);
}

TEST(DmPeakGrouping, BatchRejectsInconsistentDmForOneDmIndex) {
  auto first = peak(1.0, 2, 10.0F);
  first.dm = 10.0;
  first.dm_index = 1;
  auto second = first;
  second.dm = 11.0;

  EXPECT_THROW((void)gaffa::group_dm_peak_batch_cpu(
                   std::vector<gaffa::DmPeak>{first, second}, 100.0),
               std::invalid_argument);
}
