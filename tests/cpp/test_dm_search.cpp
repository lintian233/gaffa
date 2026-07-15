#include "gaffa/dm_search.h"

#include <gtest/gtest.h>

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

gaffa::DmSearchOptions small_search_options() {
  return gaffa::DmSearchOptions{
      .plan = small_plan_options(),
      .snr_threshold = 0.0F,
  };
}

}  // namespace

TEST(DmSearch, ExtractsDmTimeSeriesAsFloat) {
  const gaffa::DedispersedResult<std::uint32_t> input{
      .data = {1, 2, 3, 4, 10, 20, 30, 40},
      .shape = {.ndm = 2, .nsamples = 4},
  };

  const auto time_series = gaffa::dm_time_series_cpu(input, 1, 0.001);

  EXPECT_DOUBLE_EQ(time_series.tsamp, 0.001);
  EXPECT_EQ(time_series.data, (std::vector<float>{10.0F, 20.0F, 30.0F, 40.0F}));
}

TEST(DmSearch, FindsEveryDmPeakAndAttachesMetadata) {
  const gaffa::DedispersedResult<float> input{
      .data = {
          0.0F, 0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 5.0F,
          0.0F, 0.0F, 0.0F, 10.0F, 0.0F, 0.0F, 0.0F, 10.0F,
      },
      .shape = {.ndm = 2, .nsamples = 8},
  };
  const std::vector<double> dms{12.5, 20.0};

  const auto result =
      gaffa::search_dedispersed_ffa_cpu(input, dms, 1.0,
                                         small_search_options());

  ASSERT_GE(result.peaks.size(), 2);
  EXPECT_EQ(result.peaks.front().dm_index, 1);
  EXPECT_DOUBLE_EQ(result.peaks.front().dm, 20.0);
  EXPECT_GT(result.peaks.front().peak.snr, 0.0F);
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

  const auto result = gaffa::search_dedispersed_ffa_cpu(input, dms, 1.0,
                                                         options);

  ASSERT_FALSE(result.peaks.empty());
  EXPECT_EQ(result.peaks.front().dm_index, 0);
  EXPECT_DOUBLE_EQ(result.peaks.front().dm, 30.0);
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

  const auto result =
      gaffa::search_dedispersed_ffa_cpu(input, dms, 1.0,
                                         small_search_options());

  ASSERT_GE(result.peaks.size(), 2);
  EXPECT_EQ(result.peaks.front().dm_index, 1);
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

  const auto result =
      gaffa::search_dedispersed_ffa_cpu(input, dms, 1.0,
                                         small_search_options());

  ASSERT_GE(result.peaks.size(), 5);
  EXPECT_EQ(result.peaks.front().dm_index, 4);
  EXPECT_DOUBLE_EQ(result.peaks.front().dm, 50.0);
}

TEST(DmSearch, RejectsInvalidInputs) {
  const gaffa::DedispersedResult<float> input{
      .data = {0.0F, 1.0F, 0.0F, 1.0F},
      .shape = {.ndm = 1, .nsamples = 4},
  };
  const std::vector<double> dms{10.0};
  const auto options = small_search_options();

  EXPECT_THROW((void)gaffa::search_dedispersed_ffa_cpu(
                   gaffa::DedispersedResult<float>{
                       .data = {},
                       .shape = {.ndm = 0, .nsamples = 4},
                   },
                   dms, 1.0, options),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::search_dedispersed_ffa_cpu(
                   input, std::vector<double>{}, 1.0, options),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::search_dedispersed_ffa_cpu(input, dms, 0.0,
                                                        options),
               std::invalid_argument);

  auto bad_options = options;
  bad_options.snr_threshold = INFINITY;
  EXPECT_THROW((void)gaffa::search_dedispersed_ffa_cpu(input, dms, 1.0,
                                                        bad_options),
               std::invalid_argument);

  EXPECT_THROW((void)gaffa::dm_time_series_cpu(input, 1, 1.0),
               std::out_of_range);
}
