#include "gaffa/candidate.h"
#include "gaffa/periodic_match.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
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

std::vector<std::size_t> brute_force_component_sizes(
    const std::vector<gaffa::DmPeakGroups>& sources,
    const gaffa::CandidateClusteringOptions& options) {
  struct LocalGroup {
    const gaffa::DmPeak* best = nullptr;
    std::size_t member_count = 0;
  };
  std::vector<LocalGroup> groups;
  for (const gaffa::DmPeakGroups& source : sources) {
    for (const gaffa::DmPeakGroup& group : source.groups) {
      groups.push_back({.best = &source.members[group.best_index],
                        .member_count = group.member_count});
    }
  }

  std::vector<std::size_t> labels(groups.size());
  std::iota(labels.begin(), labels.end(), std::size_t{0});
  const auto unite = [&](std::size_t lhs, std::size_t rhs) {
    const std::size_t replacement = std::min(labels[lhs], labels[rhs]);
    const std::size_t removed = std::max(labels[lhs], labels[rhs]);
    if (replacement == removed) {
      return;
    }
    for (std::size_t& label : labels) {
      if (label == removed) {
        label = replacement;
      }
    }
  };

  for (std::size_t current = 0; current < groups.size(); ++current) {
    for (std::size_t previous = 0; previous < current; ++previous) {
      const gaffa::DmPeak& lhs = *groups[current].best;
      const gaffa::DmPeak& rhs = *groups[previous].best;
      if (lhs.dm_index == rhs.dm_index ||
          std::abs(lhs.dm - rhs.dm) > options.max_dm_distance ||
          (!options.cluster_across_widths &&
           (lhs.peak.phase_bins != rhs.peak.phase_bins ||
            lhs.peak.boxcar_width_bins != rhs.peak.boxcar_width_bins))) {
        continue;
      }
      if (gaffa::periodic_phase_drift(lhs.peak.motion, rhs.peak.motion, 100.0)
              .maximum_cycles <= options.max_phase_distance_cycles) {
        unite(current, previous);
      }
    }
  }

  std::unordered_map<std::size_t, std::size_t> counts;
  for (std::size_t index = 0; index < groups.size(); ++index) {
    counts[labels[index]] += groups[index].member_count;
  }
  std::vector<std::size_t> result;
  result.reserve(counts.size());
  for (const auto& [label, count] : counts) {
    (void)label;
    result.push_back(count);
  }
  std::sort(result.begin(), result.end());
  return result;
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
  EXPECT_THROW((void)cluster(
                   {peaks},
                   {.max_dm_distance = -1.0}),
               std::invalid_argument);

  auto bad = peaks;
  bad.members.front().peak.motion.frequency_hz = INFINITY;
  EXPECT_THROW((void)cluster(std::vector<gaffa::DmPeakGroups>{bad}),
               std::invalid_argument);

  EXPECT_THROW(
      (void)cluster(std::vector<gaffa::DmPeakGroups>{
          gaffa::DmPeakGroups{.members = {},
                              .groups = {gaffa::DmPeakGroup{
                                  .best_index = 0,
                                  .member_begin = 0,
                                  .member_count = 1,
                              }}}}),
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

TEST(CandidateClustering, UsesPhysicalDmDistanceForNonUniformTrials) {
  const auto result = cluster(
      {group({dm_peak(100.0, 0, 1.0000, 2, 7.0F)}),
       group({dm_peak(100.5, 1, 1.0001, 2, 8.0F)}),
       group({dm_peak(101.5, 2, 1.0002, 2, 6.0F)})},
      {.max_phase_distance_cycles = 0.01, .max_dm_distance = 0.5});

  // The first two trials are within the physical radius. The third trial is
  // one DM unit from the second one even though its trial index is adjacent.
  EXPECT_EQ(result.candidates.size(), 2U);
  EXPECT_EQ(result.candidates.front().member_count, 2U);
}

TEST(CandidateClustering, PhysicalDmWindowIsIndependentOfInputOrder) {
  const auto result = cluster(
      {group({dm_peak(102.0, 4, 1.0003, 2, 6.0F)}),
       group({dm_peak(100.5, 1, 1.0001, 2, 8.0F)}),
       group({dm_peak(100.0, 0, 1.0000, 2, 7.0F)}),
       group({dm_peak(101.0, 3, 1.0002, 2, 9.0F)})},
      {.max_phase_distance_cycles = 0.02, .max_dm_distance = 0.5});

  ASSERT_EQ(result.candidates.size(), 2U);
  EXPECT_EQ(result.candidates[0].member_count, 3U);
  EXPECT_EQ(result.candidates[1].member_count, 1U);
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

TEST(CandidateClustering, StrictWidthPolicyAlsoRequiresProfileResolution) {
  auto first = dm_peak(10.0, 0, 1.0000, 4, 7.0F);
  first.peak.phase_bins = 180;
  auto second = dm_peak(11.0, 1, 1.0005, 4, 9.0F);
  second.peak.phase_bins = 256;

  const auto groups = std::vector<gaffa::DmPeakGroups>{
      group({first}),
      group({second}),
  };
  const auto strict = cluster(
      groups, {.cluster_across_widths = false});
  const auto permissive = cluster(groups);

  EXPECT_EQ(strict.candidates.size(), 2);
  EXPECT_EQ(permissive.candidates.size(), 1);
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

TEST(CandidateClustering, ZeroPhaseRadiusLinksIdenticalTrajectories) {
  const auto result = cluster(
      {group({dm_peak(10.0, 0, 1.0, 2, 7.0F)}),
       group({dm_peak(11.0, 1, 1.0, 2, 9.0F)})},
      {.max_phase_distance_cycles = 0.0, .max_dm_distance = 1.0});

  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front().member_count, 2U);
}

TEST(CandidateClustering, IndexedGraphMatchesBruteForceTaylorGraph) {
  std::vector<gaffa::DmPeakGroups> sources;
  for (std::size_t dm_index = 0; dm_index < 40; ++dm_index) {
    const double dm = 100.0 + 0.5 * static_cast<double>(dm_index);
    std::vector<gaffa::DmPeak> peaks;

    peaks.push_back(dm_peak(dm, dm_index,
                            1.0 + 0.0002 * static_cast<double>(dm_index),
                            2, 20.0F));
    peaks.push_back(dm_peak(dm, dm_index,
                            2.0 + 0.0015 * static_cast<double>(dm_index),
                            2, 15.0F));

    auto acceleration = dm_peak(dm, dm_index, 3.0, 4, 18.0F);
    acceleration.peak.motion = {
        .order = gaffa::MotionOrder::Acceleration,
        .reference_time_seconds = 50.0,
        .frequency_hz = 3.0,
        .acceleration_m_per_s2 =
            1.0 + 0.01 * static_cast<double>(dm_index),
    };
    peaks.push_back(acceleration);

    auto jerk = dm_peak(dm, dm_index, 4.0, 4, 16.0F);
    jerk.peak.motion = {
        .order = gaffa::MotionOrder::Jerk,
        .reference_time_seconds = 50.0,
        .frequency_hz = 4.0,
        .acceleration_m_per_s2 = 2.0,
        .jerk_m_per_s3 = 0.02 * static_cast<double>(dm_index),
    };
    peaks.push_back(jerk);
    sources.push_back(gaffa::group_dm_peaks_cpu(
        peaks, 100.0,
        {.max_phase_distance_cycles = 0.01, .merge_widths = true}));
  }

  const gaffa::CandidateClusteringOptions options{
      .max_phase_distance_cycles = 0.1,
      .max_dm_distance = 0.5,
      .cluster_across_widths = true,
  };
  const auto actual = cluster(sources, options);
  std::vector<std::size_t> actual_sizes;
  actual_sizes.reserve(actual.candidates.size());
  for (const gaffa::Candidate& candidate : actual.candidates) {
    actual_sizes.push_back(candidate.member_count);
  }
  std::sort(actual_sizes.begin(), actual_sizes.end());

  EXPECT_EQ(actual_sizes, brute_force_component_sizes(sources, options));
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
