#include "gaffa/candidate_analysis.h"

#include <gtest/gtest.h>

#include <vector>

namespace {

gaffa::DmPeak peak(double dm,
                   std::size_t dm_index,
                   double frequency_hz,
                   float snr) {
  return {
      .dm = dm,
      .dm_index = dm_index,
      .peak = {
          .motion = {.frequency_hz = frequency_hz},
          .phase_bin = 0,
          .phase_bins = 100,
          .boxcar_width_bins = 2,
          .duty_cycle = 0.02,
          .snr = snr,
      },
  };
}

gaffa::HarmonicContext context() {
  return {
      .observation_seconds = 100.0,
      .frequency_low_mhz = 1000.0,
      .frequency_high_mhz = 1500.0,
  };
}

gaffa::CandidateOptions options() {
  return {
      .grouping = {.max_phase_distance_cycles = 0.0},
      .clustering = {
          .max_phase_distance_cycles = 0.0,
          .max_dm_index_distance = 1,
      },
      .selection = {
          .snr_min = 10.0F,
          .max_candidates = 2,
      },
  };
}

}  // namespace

TEST(CandidateAnalysis, MatchesTheAtomicPipelineAndPreservesDiagnostics) {
  const std::vector<gaffa::DmPeak> peaks{
      peak(10.0, 0, 1.0, 20.0F),
      peak(11.0, 1, 1.0, 19.0F),
      peak(12.0, 2, 2.0, 18.0F),
      peak(100.0, 8, 3.7, 12.0F),
  };

  const auto result = gaffa::make_candidates_cpu(peaks, context(), options());

  const auto groups = gaffa::group_dm_peak_batch_cpu(
      peaks, context().observation_seconds, options().grouping);
  const auto candidates = gaffa::cluster_dm_peak_groups_cpu(
      groups, context().observation_seconds, options().clustering);
  const auto relations = gaffa::flag_harmonics_cpu(
      candidates, context(), options().harmonic);
  const auto non_harmonic =
      gaffa::remove_harmonics_cpu(candidates, relations);

  ASSERT_EQ(result.candidate_set.candidates.size(), candidates.candidates.size());
  ASSERT_EQ(result.candidate_set.members.size(), peaks.size());
  ASSERT_EQ(result.harmonic_relations.size(), relations.size());
  ASSERT_EQ(result.selected.size(), 2);
  ASSERT_EQ(non_harmonic.size(), 2);
  EXPECT_TRUE(result.harmonic_relations[1].is_harmonic);
  EXPECT_DOUBLE_EQ(
      result.candidate_set.candidates[result.selected[0]].best.peak.motion.frequency_hz,
      1.0);
  EXPECT_DOUBLE_EQ(
      result.candidate_set.candidates[result.selected[1]].best.peak.motion.frequency_hz,
      3.7);
}

TEST(CandidateAnalysis, AppliesSnrThresholdAfterHarmonicRemoval) {
  const std::vector<gaffa::DmPeak> peaks{
      peak(10.0, 0, 1.0, 20.0F),
      peak(11.0, 1, 2.0, 18.0F),
      peak(20.0, 8, 4.0, 9.0F),
  };
  auto analysis_options = options();
  analysis_options.selection.max_candidates = 0;

  const auto result =
      gaffa::make_candidates_cpu(peaks, context(), analysis_options);

  ASSERT_EQ(result.selected.size(), 1);
  EXPECT_DOUBLE_EQ(
      result.candidate_set.candidates[result.selected.front()]
          .best.peak.motion.frequency_hz,
      1.0);
}
