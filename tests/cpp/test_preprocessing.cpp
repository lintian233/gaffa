#include "gaffa/preprocessing.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

TEST(Preprocessing, ConvertsRiptideWindowSecondsToSamples) {
  EXPECT_EQ(gaffa::running_median_window_seconds(5.0, 0.5).window_samples, 11);
  EXPECT_EQ(gaffa::running_median_window_seconds(1.25, 0.5).window_samples, 3);
  EXPECT_EQ(gaffa::running_median_window_seconds(0.1, 1.0).window_samples, 1);
}

TEST(Preprocessing, RejectsInvalidRiptideWindowSeconds) {
  EXPECT_THROW((void)gaffa::running_median_window_seconds(0.0, 0.5),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::running_median_window_seconds(1.0, 0.0),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::running_median_window_seconds(INFINITY, 0.5),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::running_median_window_seconds(1.0e308, 1.0e-308),
               std::invalid_argument);
}

TEST(Preprocessing, MakesRiptidePreprocessPlan) {
  const auto plan = gaffa::make_riptide_preprocess_plan(
      0.5, gaffa::RiptidePreprocessOptions{
               .running_median_width_seconds = 5.0,
               .running_median_min_points = 5,
           });

  ASSERT_EQ(plan.steps.size(), 2);
  EXPECT_EQ(plan.steps[0].kind, gaffa::PreprocessStepKind::DetrendRunningMedian);
  EXPECT_EQ(plan.steps[0].detrend_running_median.window_samples, 11);
  EXPECT_EQ(plan.steps[0].detrend_running_median.min_points, 5);
  EXPECT_EQ(plan.steps[1].kind, gaffa::PreprocessStepKind::Normalise);
}

TEST(Preprocessing, MakesRiptidePreprocessPlanWithoutNormalise) {
  const auto plan = gaffa::make_riptide_preprocess_plan(
      0.5, gaffa::RiptidePreprocessOptions{
               .running_median_width_seconds = 5.0,
               .normalise = false,
           });

  ASSERT_EQ(plan.steps.size(), 1);
  EXPECT_EQ(plan.steps[0].kind, gaffa::PreprocessStepKind::DetrendRunningMedian);
}

TEST(Preprocessing, RunningMedianDetrendRejectsInvalidInputs) {
  std::vector<float> output(3);

  EXPECT_THROW(gaffa::detrend_running_median_cpu(
                   std::span<const float>{},
                   gaffa::DetrendRunningMedianOptions{.window_samples = 3},
                   output),
               std::invalid_argument);
  EXPECT_THROW(gaffa::detrend_running_median_cpu(
                   std::vector<float>{1.0F, 2.0F, 3.0F},
                   gaffa::DetrendRunningMedianOptions{.window_samples = 0},
                   output),
               std::invalid_argument);
  EXPECT_THROW(gaffa::detrend_running_median_cpu(
                   std::vector<float>{1.0F, 2.0F, 3.0F},
                   gaffa::DetrendRunningMedianOptions{.window_samples = 2},
                   output),
               std::invalid_argument);
  EXPECT_THROW(gaffa::detrend_running_median_cpu(
                   std::vector<float>{1.0F, 2.0F, 3.0F},
                   gaffa::DetrendRunningMedianOptions{
                       .window_samples = 3,
                       .min_points = 2,
                   },
                   output),
               std::invalid_argument);
  EXPECT_THROW(gaffa::detrend_running_median_cpu(
                   std::vector<float>{1.0F, 2.0F, 3.0F},
                   gaffa::DetrendRunningMedianOptions{.window_samples = 3},
                   output),
               std::invalid_argument);
  EXPECT_THROW(gaffa::detrend_running_median_cpu(
                   std::vector<float>{1.0F, 2.0F, 3.0F},
                   gaffa::DetrendRunningMedianOptions{.window_samples = 3},
                   std::span<float>{}),
               std::invalid_argument);
  EXPECT_THROW(gaffa::detrend_running_median_cpu(
                   std::vector<float>{1.0F, NAN, 3.0F},
                   gaffa::DetrendRunningMedianOptions{.window_samples = 3},
                   output),
               std::invalid_argument);
  EXPECT_THROW(gaffa::detrend_running_median_cpu(
                   std::vector<float>{1.0F, INFINITY, 3.0F},
                   gaffa::DetrendRunningMedianOptions{.window_samples = 3},
                   output),
               std::invalid_argument);
}

TEST(Preprocessing, FastRunningMedianRejectsTooShortLowResolutionInput) {
  const std::vector<float> input{
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
      6.0F, 7.0F, 8.0F, 9.0F, 10.0F,
  };

  EXPECT_THROW(
      (void)gaffa::detrend_running_median_cpu(
          input, gaffa::DetrendRunningMedianOptions{
                     .window_samples = 11,
                     .min_points = 5,
                 }),
      std::invalid_argument);
}

TEST(Preprocessing, ExactRunningMedianUsesRiptideEdgePadding) {
  const std::vector<float> input{100.0F, 0.0F, 1.0F, 2.0F, 3.0F};

  const auto output = gaffa::detrend_running_median_cpu(
      input, gaffa::DetrendRunningMedianOptions{.window_samples = 3});

  ASSERT_EQ(output.size(), input.size());
  EXPECT_FLOAT_EQ(output[0], 0.0F);
  EXPECT_FLOAT_EQ(output[1], -1.0F);
  EXPECT_FLOAT_EQ(output[2], 0.0F);
  EXPECT_FLOAT_EQ(output[3], 0.0F);
  EXPECT_FLOAT_EQ(output[4], 0.0F);
}

TEST(Preprocessing, FastRunningMedianScrunchesAndInterpolates) {
  const std::vector<float> input{
      1.0F, 1.0F, 1.0F, 9.0F, 9.0F, 9.0F,
      1.0F, 1.0F, 1.0F, 9.0F, 9.0F, 9.0F,
  };

  const auto output = gaffa::detrend_running_median_cpu(
      input, gaffa::DetrendRunningMedianOptions{
                 .window_samples = 7,
                 .min_points = 3,
             });

  ASSERT_EQ(output.size(), input.size());
  EXPECT_FLOAT_EQ(output[0], 0.0F);
  EXPECT_FLOAT_EQ(output[1], -1.0F);
  EXPECT_FLOAT_EQ(output[2], -3.0F);
  EXPECT_FLOAT_EQ(output[3], 4.0F);
  EXPECT_FLOAT_EQ(output[9], 3.0F);
  EXPECT_FLOAT_EQ(output[10], 1.0F);
  EXPECT_FLOAT_EQ(output[11], 0.0F);
}

TEST(Preprocessing, RunningMedianDetrendPreservesLengthAndRemovesBaseline) {
  const std::vector<float> input{10.0F, 10.0F, 20.0F, 10.0F, 10.0F};

  const auto output = gaffa::detrend_running_median_cpu(
      input, gaffa::DetrendRunningMedianOptions{.window_samples = 3});

  ASSERT_EQ(output.size(), input.size());
  EXPECT_FLOAT_EQ(output[0], 0.0F);
  EXPECT_FLOAT_EQ(output[1], 0.0F);
  EXPECT_FLOAT_EQ(output[2], 10.0F);
  EXPECT_FLOAT_EQ(output[3], 0.0F);
  EXPECT_FLOAT_EQ(output[4], 0.0F);
}

TEST(Preprocessing, RunsStepsInPlanOrder) {
  const gaffa::TimeSeries input{
      .data = {10.0F, 10.0F, 20.0F, 10.0F, 10.0F},
      .tsamp = 0.001,
  };
  const gaffa::PreprocessPlan plan{
      .steps = {
          gaffa::PreprocessStep{
              .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
              .detrend_running_median = {.window_samples = 3},
          },
          gaffa::PreprocessStep{
              .kind = gaffa::PreprocessStepKind::Normalise,
              .normalise = {.reject_constant = true},
          },
      },
  };

  const auto output = gaffa::preprocess_time_series_cpu(input, plan);

  EXPECT_DOUBLE_EQ(output.tsamp, input.tsamp);
  ASSERT_EQ(output.data.size(), input.data.size());
  const auto stats = gaffa::time_series_stats_cpu(output.data);
  EXPECT_NEAR(stats.mean, 0.0, 1.0e-6);
  EXPECT_NEAR(stats.stddev, 1.0, 1.0e-6);
  EXPECT_GT(output.data[2], output.data[0]);
}

TEST(Preprocessing, ReturningApiDoesNotMutateInput) {
  const gaffa::TimeSeries input{
      .data = {1.0F, 2.0F, 3.0F},
      .tsamp = 0.001,
  };
  const gaffa::PreprocessPlan plan{
      .steps = {
          gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise},
      },
  };

  const auto output = gaffa::preprocess_time_series_cpu(input, plan);

  EXPECT_EQ(input.data, (std::vector<float>{1.0F, 2.0F, 3.0F}));
  EXPECT_DOUBLE_EQ(output.tsamp, input.tsamp);
  EXPECT_NE(input.data, output.data);
}

TEST(Preprocessing, InplaceApiMutatesDataAndPreservesTsamp) {
  gaffa::TimeSeries input{
      .data = {1.0F, 2.0F, 3.0F},
      .tsamp = 0.001,
  };
  const gaffa::PreprocessPlan plan{
      .steps = {
          gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise},
      },
  };

  gaffa::preprocess_time_series_inplace_cpu(input, plan);

  EXPECT_DOUBLE_EQ(input.tsamp, 0.001);
  const auto stats = gaffa::time_series_stats_cpu(input.data);
  EXPECT_NEAR(stats.mean, 0.0, 1.0e-6);
  EXPECT_NEAR(stats.stddev, 1.0, 1.0e-6);
}

TEST(Preprocessing, RejectsInvalidTimeSeriesAndEmptyPlan) {
  const gaffa::PreprocessPlan empty_plan{};
  const gaffa::PreprocessPlan normalise_plan{
      .steps = {
          gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise},
      },
  };

  EXPECT_THROW(gaffa::preprocess_time_series_cpu(
                   gaffa::TimeSeries{.data = {1.0F}, .tsamp = 0.0},
                   normalise_plan),
               std::invalid_argument);
  EXPECT_THROW(gaffa::preprocess_time_series_cpu(
                   gaffa::TimeSeries{.data = {1.0F}, .tsamp = 0.001},
                   empty_plan),
               std::invalid_argument);
}
