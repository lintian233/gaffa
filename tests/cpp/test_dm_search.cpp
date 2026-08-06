#include "gaffa/dm_search.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

gaffa::RiptideFfaPlanOptions small_plan_options() {
  return gaffa::RiptideFfaPlanOptions{
      .period_min = 2.0,
      .period_max = 3.0,
      .bins_min = 2,
      .bins_max = 2,
      .min_periods = 1,
      .duty_cycle_max = 0.75,
      .width_trial_spacing = 2.0,
  };
}

gaffa::FfaSearchPlan small_plan(std::size_t nsamples = 8,
                                double tsamp = 1.0) {
  return gaffa::make_riptide_ffa_plan(nsamples, tsamp,
                                      small_plan_options());
}

gaffa::DmFfaOptions small_search_options() {
  return gaffa::DmFfaOptions{
      .search = {.snr_threshold = 0.0F},
  };
}

}  // namespace

TEST(DmSearch, FindsEveryDmPeakAndAttachesMetadata) {
  const gaffa::DedispersedResult<float> input{
      .data = {
          0.0F, 0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 5.0F,
          0.0F, 0.0F, 0.0F, 10.0F, 0.0F, 0.0F, 0.0F, 10.0F,
      },
      .shape = {.ndm = 2, .nsamples = 8},
  };
  const std::vector<double> dms{12.5, 20.0};

  const auto result = gaffa::search_dm_ffa_cpu(
      input, {.values = dms, .index_offset = 10}, small_plan(),
      small_search_options());

  ASSERT_FALSE(result.empty());
  EXPECT_EQ(result.front().dm_index, 10);
  EXPECT_DOUBLE_EQ(result.back().dm, 20.0);
  for (const auto& peak : result) {
    EXPECT_DOUBLE_EQ(peak.peak.motion.reference_time_seconds, 4.0);
  }
}

TEST(DmSearch, NonOwningViewMatchesOwningInput) {
  const gaffa::DedispersedResult<float> input{
      .data = {0.0F, 0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 5.0F},
      .shape = {.ndm = 1, .nsamples = 8},
  };
  const std::array<double, 1> dms{12.5};
  const auto plan = small_plan(input.shape.nsamples, 1.0);

  const auto owning = gaffa::search_dm_ffa_cpu(
      input, gaffa::DmTrialView{.values = dms}, plan);
  const auto viewing = gaffa::search_dm_ffa_cpu(
      input.view(), gaffa::DmTrialView{.values = dms}, plan);

  ASSERT_EQ(viewing.size(), owning.size());
  for (std::size_t index = 0; index < owning.size(); ++index) {
    EXPECT_EQ(viewing[index].dm, owning[index].dm);
    EXPECT_EQ(viewing[index].dm_index, owning[index].dm_index);
    EXPECT_EQ(viewing[index].peak.snr, owning[index].peak.snr);
    EXPECT_EQ(viewing[index].peak.motion.frequency_hz,
              owning[index].peak.motion.frequency_hz);
  }
}

TEST(DmSearch, AppliesPreprocessPlanBeforeSearch) {
  const gaffa::DedispersedResult<float> input{
      .data = {2.0F, 2.0F, 2.0F, 8.0F, 2.0F, 2.0F, 2.0F, 8.0F},
      .shape = {.ndm = 1, .nsamples = 8},
  };
  const std::vector<double> dms{30.0};
  auto options = small_search_options();
  options.preprocess.steps.push_back(gaffa::PreprocessStep{
      .kind = gaffa::PreprocessStepKind::Normalise,
  });

  const auto result = gaffa::search_dm_ffa_cpu(
      input, {.values = dms}, small_plan(), options);

  ASSERT_FALSE(result.empty());
  EXPECT_EQ(result.front().dm_index, 0);
  EXPECT_DOUBLE_EQ(result.front().dm, 30.0);
}

TEST(DmSearch, FloatEmptyPreprocessMatchesDirectFfa) {
  const gaffa::DedispersedResult<float> input{
      .data = {0.0F, 0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 5.0F},
      .shape = {.ndm = 1, .nsamples = 8},
  };
  const std::array<double, 1> dms{12.5};
  const auto plan = small_plan(input.shape.nsamples, 1.0);
  const auto options = small_search_options();

  const auto actual = gaffa::search_dm_ffa_cpu(
      input.view(), {.values = dms, .index_offset = 3}, plan, options);
  const auto periodic = gaffa::search_ffa_cpu(
      input.view().dm_series(0), plan, options.search);
  const auto expected = gaffa::attach_dm_peaks(periodic, dms[0], 3);

  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_EQ(actual[index].dm, expected[index].dm);
    EXPECT_EQ(actual[index].dm_index, expected[index].dm_index);
    EXPECT_EQ(actual[index].peak.snr, expected[index].peak.snr);
    EXPECT_EQ(actual[index].peak.motion.frequency_hz,
              expected[index].peak.motion.frequency_hz);
    EXPECT_EQ(actual[index].peak.phase_bin, expected[index].peak.phase_bin);
  }
}

TEST(DmSearch, Uint32PreprocessMatchesFloatReference) {
  const std::vector<std::uint32_t> samples{
      2U, 2U, 2U, 8U, 2U, 2U, 2U, 8U,
  };
  const gaffa::DedispersedResult<std::uint32_t> integer_input{
      .data = samples,
      .shape = {.ndm = 1, .nsamples = samples.size()},
  };
  const gaffa::DedispersedResult<float> float_input{
      .data = {2.0F, 2.0F, 2.0F, 8.0F, 2.0F, 2.0F, 2.0F, 8.0F},
      .shape = integer_input.shape,
  };
  const std::array<double, 1> dms{30.0};
  auto options = small_search_options();
  options.preprocess.steps.push_back(gaffa::PreprocessStep{
      .kind = gaffa::PreprocessStepKind::Normalise,
  });
  const auto plan = small_plan(integer_input.shape.nsamples, 1.0);

  const auto actual = gaffa::search_dm_ffa_cpu(
      integer_input.view(), {.values = dms}, plan, options);
  const auto expected = gaffa::search_dm_ffa_cpu(
      float_input.view(), {.values = dms}, plan, options);

  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_EQ(actual[index].dm, expected[index].dm);
    EXPECT_EQ(actual[index].dm_index, expected[index].dm_index);
    EXPECT_EQ(actual[index].peak.snr, expected[index].peak.snr);
    EXPECT_EQ(actual[index].peak.motion.frequency_hz,
              expected[index].peak.motion.frequency_hz);
    EXPECT_EQ(actual[index].peak.phase_bin, expected[index].peak.phase_bin);
  }
}

TEST(DmSearch, KeepsAllSignificantPeaksInsteadOfTopK) {
  const gaffa::DedispersedResult<float> input{
      .data = {
          0.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 0.0F, 4.0F,
          0.0F, 0.0F, 0.0F, 8.0F, 0.0F, 0.0F, 0.0F, 8.0F,
      },
      .shape = {.ndm = 2, .nsamples = 8},
  };
  const std::vector<double> dms{10.0, 20.0};

  const auto result = gaffa::search_dm_ffa_cpu(
      input, {.values = dms}, small_plan(), small_search_options());

  EXPECT_GE(result.size(), 2);
}

TEST(DmSearch, ParallelPathMergesPeaks) {
  const gaffa::DedispersedResult<float> input{
      .data = {
          0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 2.0F,
          0.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 0.0F, 4.0F,
          0.0F, 0.0F, 0.0F, 6.0F, 0.0F, 0.0F, 0.0F, 6.0F,
          0.0F, 0.0F, 0.0F, 8.0F, 0.0F, 0.0F, 0.0F, 8.0F,
          0.0F, 0.0F, 0.0F, 10.0F, 0.0F, 0.0F, 0.0F, 10.0F,
      },
      .shape = {.ndm = 5, .nsamples = 8},
  };
  const std::vector<double> dms{10.0, 20.0, 30.0, 40.0, 50.0};

  const auto result = gaffa::search_dm_ffa_cpu(
      input, {.values = dms}, small_plan(), small_search_options());

  ASSERT_FALSE(result.empty());
  EXPECT_EQ(result.front().dm_index, 0);
  EXPECT_DOUBLE_EQ(result.back().dm, 50.0);
}

TEST(DmSearch, RejectsInvalidInputs) {
  const gaffa::DedispersedResult<float> input{
      .data = {0.0F, 1.0F, 0.0F, 1.0F},
      .shape = {.ndm = 1, .nsamples = 4},
  };
  const std::vector<double> dms{10.0};
  const auto options = small_search_options();

  EXPECT_THROW((void)gaffa::search_dm_ffa_cpu(
                   gaffa::DedispersedResult<float>{
                       .data = {},
                       .shape = {.ndm = 0, .nsamples = 4},
                   },
                   {.values = dms}, small_plan(4), options),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::search_dm_ffa_cpu(
                   input, {.values = std::span<const double>{}},
                   small_plan(4), options),
               std::invalid_argument);

  auto bad_plan = small_plan(4);
  bad_plan.observation.tsamp_seconds = 0.0;
  EXPECT_THROW((void)gaffa::search_dm_ffa_cpu(
                   input, {.values = dms}, bad_plan, options),
               std::invalid_argument);

  auto bad_options = options;
  bad_options.search.snr_threshold = INFINITY;
  EXPECT_THROW((void)gaffa::search_dm_ffa_cpu(
                   input, {.values = dms}, small_plan(4), bad_options),
               std::invalid_argument);
}
