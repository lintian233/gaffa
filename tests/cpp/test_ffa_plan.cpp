#include "gaffa/ffa_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

gaffa::RiptideFfaPlanOptions valid_options() {
  return gaffa::RiptideFfaPlanOptions{
      .period_min = 1.0,
      .period_max = 4.0,
      .bins_min = 4,
      .bins_max = 5,
      .min_periods = 2,
      .duty_cycle_max = 0.50,
      .width_trial_spacing = 1.5,
      .max_tasks = 1000,
  };
}

}  // namespace

TEST(FfaPlan, RejectsInvalidOptions) {
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(0, 0.25, valid_options()),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.0, valid_options()),
               std::invalid_argument);

  auto options = valid_options();
  options.period_min = 0.0;
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::invalid_argument);

  options = valid_options();
  options.period_max = options.period_min;
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::invalid_argument);

  options = valid_options();
  options.bins_min = 1;
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::invalid_argument);

  options = valid_options();
  options.bins_max = options.bins_min - 1;
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::invalid_argument);

  options = valid_options();
  options.min_periods = 0;
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::invalid_argument);

  options = valid_options();
  options.duty_cycle_max = 1.0;
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::invalid_argument);

  options = valid_options();
  options.width_trial_spacing = 1.0;
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::invalid_argument);

  options = valid_options();
  options.max_tasks = 0;
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::invalid_argument);

  options = valid_options();
  options.period_min = 0.5;
  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::invalid_argument);
}

TEST(FfaPlan, GeneratesWidthTrialsLikeRiptide) {
  auto options = valid_options();
  options.period_min = 240.0;
  options.period_max = 300.0;
  options.bins_min = 240;
  options.bins_max = 260;
  options.duty_cycle_max = 0.20;
  options.width_trial_spacing = 1.5;

  const auto plan = gaffa::make_riptide_ffa_plan(1000, 1.0, options);

  EXPECT_EQ(plan.width_trials,
            (std::vector<std::size_t>{1, 2, 3, 4, 6, 9, 13, 19, 28, 42}));
}

TEST(FfaPlan, MatchesSmallKnownPlan) {
  const auto plan = gaffa::make_riptide_ffa_plan(64, 0.25, valid_options());

  ASSERT_EQ(plan.tasks.size(), 7);
  EXPECT_EQ(plan.width_trials, (std::vector<std::size_t>{1, 2}));

  const auto& first = plan.tasks.front();
  EXPECT_DOUBLE_EQ(first.downsample_factor, 1.0);
  EXPECT_DOUBLE_EQ(first.effective_tsamp, 0.25);
  EXPECT_EQ(first.input_nsamples, 64);
  EXPECT_EQ(first.prepared_nsamples, 64);
  EXPECT_EQ(first.bins, 4);
  EXPECT_EQ(first.rows, 16);
  EXPECT_EQ(first.rows_eval, 12);
  EXPECT_DOUBLE_EQ(first.period_begin, 1.0);
  EXPECT_NEAR(first.period_end, 1.2244897959183674, 1.0e-12);

  const auto& last = plan.tasks.back();
  EXPECT_EQ(last.bins, 4);
  EXPECT_EQ(last.rows, 4);
  EXPECT_EQ(last.rows_eval, 2);
  EXPECT_DOUBLE_EQ(last.period_begin, 3.375);
  EXPECT_NEAR(last.period_end, 3.681818181818182, 1.0e-12);
}

TEST(FfaPlan, KeepsTasksWithinRequestedBins) {
  const auto options = valid_options();
  const auto plan = gaffa::make_riptide_ffa_plan(64, 0.25, options);

  ASSERT_FALSE(plan.tasks.empty());
  for (const auto& task : plan.tasks) {
    EXPECT_GE(task.bins, options.bins_min);
    EXPECT_LE(task.bins, options.bins_max);
    EXPECT_GE(task.rows, options.min_periods);
    EXPECT_GT(task.rows_eval, 0);
    EXPECT_LE(task.rows_eval, task.rows);
    EXPECT_LE(task.period_begin, task.period_end);
  }
}

TEST(FfaPlan, AppliesMinPeriodsCap) {
  auto options = valid_options();
  options.period_max = 100.0;
  options.min_periods = 8;

  const auto plan = gaffa::make_riptide_ffa_plan(64, 0.25, options);
  const double capped_period_max = 64.0 * 0.25 / 8.0;

  ASSERT_FALSE(plan.tasks.empty());
  for (const auto& task : plan.tasks) {
    EXPECT_LE(task.period_end, capped_period_max + 1.0e-12);
    EXPECT_GE(task.rows, options.min_periods);
  }
}

TEST(FfaPlan, ProducesMonotonicPeriodCoverage) {
  const auto plan = gaffa::make_riptide_ffa_plan(64, 0.25, valid_options());

  ASSERT_FALSE(plan.tasks.empty());
  double previous_begin = 0.0;
  double previous_end = 0.0;
  for (const auto& task : plan.tasks) {
    EXPECT_GE(task.period_begin, previous_begin);
    EXPECT_GE(task.period_end, task.period_begin);
    EXPECT_GE(task.period_end, previous_end);
    previous_begin = task.period_begin;
    previous_end = task.period_end;
  }
}

TEST(FfaPlan, EnforcesMaxTasks) {
  auto options = valid_options();
  options.max_tasks = 1;

  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::runtime_error);
}

TEST(FfaPlan, RejectsEmptySearchRangeAfterMinPeriodsCap) {
  auto options = valid_options();
  options.period_min = 4.0;
  options.period_max = 8.0;
  options.bins_min = 4;
  options.bins_max = 5;
  options.min_periods = 8;

  EXPECT_THROW((void)gaffa::make_riptide_ffa_plan(64, 0.25, options),
               std::invalid_argument);
}
