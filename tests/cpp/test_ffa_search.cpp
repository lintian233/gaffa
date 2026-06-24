#include "gaffa/ffa_search.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

gaffa::FfaSearchTask make_task(std::size_t input_nsamples,
                               double downsample_factor,
                               std::size_t prepared_nsamples,
                               std::size_t rows,
                               std::size_t rows_eval,
                               std::size_t bins) {
  return gaffa::FfaSearchTask{
      .downsample_factor = downsample_factor,
      .effective_tsamp = downsample_factor,
      .input_nsamples = input_nsamples,
      .prepared_nsamples = prepared_nsamples,
      .bins = bins,
      .rows = rows,
      .rows_eval = rows_eval,
      .period_begin = downsample_factor * static_cast<double>(bins),
      .period_end = downsample_factor * static_cast<double>(bins + 1),
  };
}

gaffa::FfaSearchPlan single_task_plan(std::size_t input_nsamples) {
  return gaffa::FfaSearchPlan{
      .tasks = {make_task(input_nsamples, 1.0, input_nsamples, 4, 4, 2)},
      .width_trials = {1},
  };
}

}  // namespace

TEST(FfaSearchCpu, RejectsInvalidOptions) {
  const std::vector<float> input{1, 0, 2, 0, 3, 0, 4, 0};
  const auto plan = single_task_plan(input.size());

  EXPECT_THROW((void)gaffa::search_ffa_cpu(
                   input, plan, gaffa::FfaSearchOptions{.max_candidates = 0}),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::search_ffa_cpu(
                   input, plan,
                   gaffa::FfaSearchOptions{
                       .snr_threshold = INFINITY,
                   }),
               std::invalid_argument);

  auto empty_widths = plan;
  empty_widths.width_trials.clear();
  EXPECT_THROW((void)gaffa::search_ffa_cpu(input, empty_widths),
               std::invalid_argument);
}

TEST(FfaSearchCpu, ReturnsEmptyWhenNoCandidatePassesThreshold) {
  const std::vector<float> input{1, 0, 2, 0, 3, 0, 4, 0};
  const auto plan = single_task_plan(input.size());

  const auto result = gaffa::search_ffa_cpu(
      input, plan,
      gaffa::FfaSearchOptions{
          .snr_threshold = 1000.0F,
      });

  EXPECT_TRUE(result.candidates.empty());
}

TEST(FfaSearchCpu, FindsCandidateThroughExecutorAndDetection) {
  const std::vector<float> input{0, 0, 0, 5, 0, 0, 0, 5};
  const auto plan = single_task_plan(input.size());

  const auto result = gaffa::search_ffa_cpu(
      input, plan,
      gaffa::FfaSearchOptions{
          .snr_threshold = 0.0F,
      });

  ASSERT_FALSE(result.candidates.empty());
  EXPECT_EQ(result.candidates.front().width, 1);
  EXPECT_EQ(result.candidates.front().bins, 2);
}

TEST(FfaSearchCpu, LimitsGlobalCandidates) {
  const std::vector<float> input{1, 0, 2, 0, 3, 0, 4, 0};
  const auto plan = single_task_plan(input.size());

  const auto result = gaffa::search_ffa_cpu(
      input, plan,
      gaffa::FfaSearchOptions{
          .snr_threshold = 0.0F,
          .max_candidates = 1,
      });

  EXPECT_EQ(result.candidates.size(), 1);
}

TEST(FfaSearchCpu, KeepsBestGlobalCandidatesAcrossBlocks) {
  const std::vector<float> input{0, 0, 0, 1, 0, 0, 0, 10};
  const gaffa::FfaSearchPlan plan{
      .tasks = {
          make_task(input.size(), 1.0, input.size(), 4, 4, 2),
          make_task(input.size(), 1.0, input.size(), 2, 2, 4),
      },
      .width_trials = {1},
  };

  const auto result = gaffa::search_ffa_cpu(
      input, plan,
      gaffa::FfaSearchOptions{
          .snr_threshold = 0.0F,
          .max_candidates = 1,
      });

  ASSERT_EQ(result.candidates.size(), 1);
  EXPECT_GT(result.candidates.front().snr, 0.0F);
  EXPECT_EQ(result.candidates.front().bins, 4);
  EXPECT_DOUBLE_EQ(result.candidates.front().period, 4.0);
}

TEST(FfaSearchCpu, ReportsDownsampledTaskPeriod) {
  const std::vector<float> input{0, 0, 0, 4, 0, 0, 0, 4};
  const gaffa::FfaSearchPlan plan{
      .tasks = {make_task(input.size(), 2.0, 4, 2, 2, 2)},
      .width_trials = {1},
  };

  const auto result = gaffa::search_ffa_cpu(
      input, plan,
      gaffa::FfaSearchOptions{
          .snr_threshold = 0.0F,
          .max_candidates = 1,
      });

  ASSERT_EQ(result.candidates.size(), 1);
  EXPECT_EQ(result.candidates.front().bins, 2);
  EXPECT_EQ(result.candidates.front().width, 1);
  EXPECT_DOUBLE_EQ(result.candidates.front().period, 4.0);
}

TEST(FfaSearchCpu, AllowsExternalPlan) {
  const std::vector<float> input{0, 0, 0, 5, 0, 0};
  const gaffa::FfaSearchPlan custom_plan{
      .tasks = {make_task(input.size(), 1.0, input.size(), 3, 2, 2)},
      .width_trials = {1},
  };

  const auto result = gaffa::search_ffa_cpu(
      input, custom_plan,
      gaffa::FfaSearchOptions{
          .snr_threshold = 0.0F,
      });

  ASSERT_FALSE(result.candidates.empty());
  EXPECT_EQ(result.candidates.front().bins, 2);
}
