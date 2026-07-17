#include "gaffa/harmonic.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

gaffa::DmPeak dm_peak(double dm,
                      std::size_t dm_index,
                      double frequency,
                      float snr) {
  return gaffa::DmPeak{
      .dm = dm,
      .dm_index = dm_index,
      .peak = {
          .motion = {.frequency_hz = frequency},
          .phase_bin = 0,
          .phase_bins = 100,
          .boxcar_width_bins = 2,
          .duty_cycle = 0.02,
          .snr = snr,
      },
  };
}

gaffa::CandidateSet candidate_set(std::vector<gaffa::DmPeak> peaks) {
  const auto groups = gaffa::group_dm_peak_batch_cpu(peaks, 100.0);
  return gaffa::cluster_dm_peak_groups_cpu(
      groups, 100.0,
      gaffa::CandidateClusteringOptions{
          .max_phase_distance_cycles = 0.0,
          .max_dm_index_distance = 1,
      });
}

gaffa::HarmonicContext context() {
  return {
      .observation_seconds = 100.0,
      .frequency_low_mhz = 1000.0,
      .frequency_high_mhz = 1500.0,
  };
}

std::size_t find_frequency(const gaffa::CandidateSet& set, double frequency) {
  for (std::size_t index = 0; index < set.candidates.size(); ++index) {
    if (set.candidates[index].best.peak.motion.frequency_hz == frequency) {
      return index;
    }
  }
  throw std::logic_error("test candidate frequency not found");
}

}  // namespace

TEST(HarmonicFlagging, EmptyInputReturnsEmpty) {
  EXPECT_TRUE(gaffa::flag_harmonics_cpu({}, context()).empty());
}

TEST(HarmonicFlagging, RejectsInvalidInputs) {
  auto candidates = candidate_set({dm_peak(10.0, 0, 1.0, 5.0F)});
  EXPECT_THROW((void)gaffa::flag_harmonics_cpu(
                   candidates,
                   {.observation_seconds = 0.0,
                    .frequency_low_mhz = 1000.0,
                    .frequency_high_mhz = 1500.0}),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::flag_harmonics_cpu(
                   candidates, context(), {.max_harmonic = 1}),
               std::invalid_argument);

  candidates.candidates.front().best.peak.motion.frequency_hz = INFINITY;
  EXPECT_THROW((void)gaffa::flag_harmonics_cpu(candidates, context()),
               std::invalid_argument);
}

TEST(HarmonicFlagging, FlagsIntegerHarmonicWithStableParentIndex) {
  const auto candidates = candidate_set({
      dm_peak(10.0, 0, 1.0, 30.0F),
      dm_peak(10.0, 0, 2.0, 20.0F),
  });
  const auto relations = gaffa::flag_harmonics_cpu(candidates, context());
  const std::size_t parent = find_frequency(candidates, 1.0);
  const std::size_t child = find_frequency(candidates, 2.0);

  EXPECT_FALSE(relations[parent].is_harmonic);
  ASSERT_TRUE(relations[child].is_harmonic);
  EXPECT_EQ(relations[child].parent_index, parent);
  EXPECT_EQ(relations[child].numerator, 2);
  EXPECT_EQ(relations[child].denominator, 1);
  EXPECT_NEAR(relations[child].maximum_phase_drift_cycles, 0.0, 1.0e-12);
}

TEST(HarmonicFlagging, FlagsFractionalButNotHighDenominatorHarmonic) {
  const auto fractional = candidate_set({
      dm_peak(10.0, 0, 1.0, 30.0F),
      dm_peak(10.0, 0, 1.5, 20.0F),
  });
  const auto fractional_relations = gaffa::flag_harmonics_cpu(
      fractional, context(), {.max_harmonic = 16, .denominator_max = 4});
  const auto child = find_frequency(fractional, 1.5);
  EXPECT_TRUE(fractional_relations[child].is_harmonic);
  EXPECT_EQ(fractional_relations[child].numerator, 3);
  EXPECT_EQ(fractional_relations[child].denominator, 2);

  const auto high_denominator = candidate_set({
      dm_peak(10.0, 0, 1.0, 30.0F),
      dm_peak(10.0, 0, 10.0 / 9.0, 20.0F),
  });
  const auto high_relations =
      gaffa::flag_harmonics_cpu(high_denominator, context());
  EXPECT_FALSE(high_relations[find_frequency(high_denominator, 10.0 / 9.0)]
                   .is_harmonic);
}

TEST(HarmonicFlagging, ExactPhaseAndOptionalFrequencyGuardsRejectMismatch) {
  const auto candidates = candidate_set({
      dm_peak(10.0, 0, 1.0, 30.0F),
      dm_peak(10.0, 0, 2.02, 20.0F),
  });
  const std::size_t child = find_frequency(candidates, 2.02);
  EXPECT_FALSE(
      gaffa::flag_harmonics_cpu(candidates, context())[child].is_harmonic);
  EXPECT_FALSE(gaffa::flag_harmonics_cpu(
                   candidates, context(),
                   {.frequency_tolerance_bins = 1.5,
                    .phase_distance_max = 1000.0})[child]
                   .is_harmonic);
}

TEST(HarmonicFlagging, SupportsConsistentTaylorMotion) {
  auto parent = dm_peak(10.0, 0, 10.0, 30.0F);
  parent.peak.motion = {
      .order = gaffa::MotionOrder::Snap,
      .reference_time_seconds = 50.0,
      .frequency_hz = 10.0,
      .acceleration_m_per_s2 = 4.0,
      .jerk_m_per_s3 = 0.2,
      .snap_m_per_s4 = 0.01,
  };
  auto child = parent;
  child.peak.motion.frequency_hz = 20.0;
  child.peak.snr = 20.0F;
  const auto candidates = candidate_set({parent, child});
  const auto relations = gaffa::flag_harmonics_cpu(candidates, context());

  const auto child_index = find_frequency(candidates, 20.0);
  ASSERT_TRUE(relations[child_index].is_harmonic);
  EXPECT_NEAR(relations[child_index].maximum_phase_drift_cycles, 0.0, 1.0e-12);
}

TEST(HarmonicFlagging, FlaggedCandidateDoesNotBecomeParent) {
  const auto candidates = candidate_set({
      dm_peak(10.0, 0, 1.0, 30.0F),
      dm_peak(10.0, 0, 1.5, 20.0F),
      dm_peak(10.0, 0, 2.25, 10.0F),
  });
  const auto relations = gaffa::flag_harmonics_cpu(
      candidates, context(), {.max_harmonic = 3, .denominator_max = 2});

  EXPECT_TRUE(relations[find_frequency(candidates, 1.5)].is_harmonic);
  EXPECT_FALSE(relations[find_frequency(candidates, 2.25)].is_harmonic);
}

TEST(HarmonicFlagging, PhysicalDmAndOptionalSnrConditionsRemainIndependent) {
  const auto distant = candidate_set({
      dm_peak(10.0, 0, 1.0, 30.0F),
      dm_peak(1000010.0, 100, 2.0, 20.0F),
  });
  EXPECT_FALSE(gaffa::flag_harmonics_cpu(distant, context())
                   [find_frequency(distant, 2.0)]
                       .is_harmonic);

  const auto weak = candidate_set({
      dm_peak(10.0, 0, 1.0, 30.0F),
      dm_peak(10.0, 0, 2.0, 1.0F),
  });
  const auto diagnostic = gaffa::flag_harmonics_cpu(weak, context());
  EXPECT_TRUE(diagnostic[find_frequency(weak, 2.0)].is_harmonic);
  const auto strict = gaffa::flag_harmonics_cpu(
      weak, context(),
      {.use_snr_consistency = true, .snr_distance_max = 3.0});
  EXPECT_FALSE(strict[find_frequency(weak, 2.0)].is_harmonic);
}

TEST(HarmonicFiltering, ReturnsCandidateIndicesAndAppliesCap) {
  const auto candidates = candidate_set({
      dm_peak(10.0, 0, 1.0, 30.0F),
      dm_peak(10.0, 0, 2.0, 20.0F),
      dm_peak(10.0, 0, 1.7, 15.0F),
  });
  const auto relations = gaffa::flag_harmonics_cpu(candidates, context());
  const auto filtered =
      gaffa::remove_harmonics_cpu(candidates, relations, 1);

  ASSERT_EQ(filtered.size(), 1);
  EXPECT_DOUBLE_EQ(
      candidates.candidates[filtered.front()].best.peak.motion.frequency_hz,
      1.0);
}
