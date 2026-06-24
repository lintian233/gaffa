#include "gaffa/time_series.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

TEST(TimeSeries, ValidatesTimeSeriesShape) {
  EXPECT_THROW(gaffa::validate_time_series(gaffa::TimeSeries{}),
               std::invalid_argument);
  EXPECT_THROW(gaffa::validate_time_series(gaffa::TimeSeries{
                   .data = {1.0F},
                   .tsamp = 0.0,
               }),
               std::invalid_argument);

  EXPECT_NO_THROW(gaffa::validate_time_series(gaffa::TimeSeries{
      .data = {1.0F},
      .tsamp = 0.001,
  }));
}

TEST(TimeSeries, ComputesStats) {
  const std::vector<float> input{1.0F, 2.0F, 3.0F};

  const auto stats = gaffa::time_series_stats_cpu(input);

  EXPECT_DOUBLE_EQ(stats.mean, 2.0);
  EXPECT_NEAR(stats.variance, 2.0 / 3.0, 1.0e-12);
  EXPECT_NEAR(stats.stddev, std::sqrt(2.0 / 3.0), 1.0e-12);
}

TEST(TimeSeries, StatsRejectsNonFiniteByDefault) {
  const std::vector<float> input{1.0F, NAN, 3.0F};

  EXPECT_THROW((void)gaffa::time_series_stats_cpu(input),
               std::invalid_argument);
}

TEST(TimeSeries, NormaliseWritesToOutputSpan) {
  const std::vector<float> input{1.0F, 2.0F, 3.0F};
  std::vector<float> output(input.size());

  gaffa::normalise_cpu(input, output);

  const auto stats = gaffa::time_series_stats_cpu(output);
  EXPECT_NEAR(stats.mean, 0.0, 1.0e-6);
  EXPECT_NEAR(stats.stddev, 1.0, 1.0e-6);
  EXPECT_LT(output[0], output[1]);
  EXPECT_LT(output[1], output[2]);
}

TEST(TimeSeries, NormaliseReturnsVector) {
  const std::vector<float> input{2.0F, 4.0F, 6.0F};

  const auto output = gaffa::normalise_cpu(input);

  ASSERT_EQ(output.size(), input.size());
  const auto stats = gaffa::time_series_stats_cpu(output);
  EXPECT_NEAR(stats.mean, 0.0, 1.0e-6);
  EXPECT_NEAR(stats.stddev, 1.0, 1.0e-6);
}

TEST(TimeSeries, NormaliseInplaceMutatesData) {
  std::vector<float> data{1.0F, 2.0F, 3.0F};

  gaffa::normalise_inplace_cpu(data);

  const auto stats = gaffa::time_series_stats_cpu(data);
  EXPECT_NEAR(stats.mean, 0.0, 1.0e-6);
  EXPECT_NEAR(stats.stddev, 1.0, 1.0e-6);
}

TEST(TimeSeries, NormaliseRejectsConstantInputByDefault) {
  const std::vector<float> input{2.0F, 2.0F, 2.0F};
  std::vector<float> output(input.size());

  EXPECT_THROW(gaffa::normalise_cpu(input, output), std::invalid_argument);
}

TEST(TimeSeries, NormaliseCanMapConstantInputToZero) {
  const std::vector<float> input{2.0F, 2.0F, 2.0F};

  const auto output =
      gaffa::normalise_cpu(input, gaffa::NormaliseOptions{
                                      .reject_constant = false,
                                  });

  EXPECT_EQ(output, (std::vector<float>{0.0F, 0.0F, 0.0F}));
}

TEST(TimeSeries, DownsampledSizeUsesFloor) {
  EXPECT_EQ(gaffa::downsampled_size(10, 2.0), 5);
  EXPECT_EQ(gaffa::downsampled_size(10, 3.0), 3);
  EXPECT_EQ(gaffa::downsampled_size(10, 2.5), 4);
}

TEST(TimeSeries, RejectsInvalidDownsampleFactor) {
  EXPECT_THROW((void)gaffa::downsampled_size(0, 2.0), std::invalid_argument);
  EXPECT_THROW((void)gaffa::downsampled_size(4, 1.0), std::invalid_argument);
  EXPECT_THROW((void)gaffa::downsampled_size(4, 0.5), std::invalid_argument);
  EXPECT_THROW((void)gaffa::downsampled_size(4, 5.0), std::invalid_argument);
  EXPECT_THROW((void)gaffa::downsampled_size(4, INFINITY),
               std::invalid_argument);
}

TEST(TimeSeries, WeightedSumHandlesIntegerFactor) {
  const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};

  const std::vector<float> output =
      gaffa::downsample_weighted_sum_cpu(input, 2.0);

  EXPECT_EQ(output, (std::vector<float>{3.0F, 7.0F}));
}

TEST(TimeSeries, WeightedSumHandlesFractionalFactor) {
  const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};

  const std::vector<float> output =
      gaffa::downsample_weighted_sum_cpu(input, 2.5);

  ASSERT_EQ(output.size(), 2);
  EXPECT_FLOAT_EQ(output[0], 4.5F);
  EXPECT_FLOAT_EQ(output[1], 10.5F);
}

TEST(TimeSeries, WeightedSumHandlesNonAlignedTail) {
  const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};

  const std::vector<float> output =
      gaffa::downsample_weighted_sum_cpu(input, 2.25);

  ASSERT_EQ(output.size(), 2);
  EXPECT_FLOAT_EQ(output[0], 1.0F + 2.0F + 0.25F * 3.0F);
  EXPECT_FLOAT_EQ(output[1], 0.75F * 3.0F + 4.0F + 0.5F * 5.0F);
}

TEST(TimeSeries, OutputSpanMustMatchDownsampledSize) {
  const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> output(1);

  EXPECT_THROW(gaffa::downsample_weighted_sum_cpu(input, 2.0, output),
               std::invalid_argument);
}

TEST(TimeSeries, DownsampledVarianceMatchesKnownCases) {
  EXPECT_NEAR(gaffa::downsampled_variance(16, 2.0), 2.0, 1.0e-12);
  EXPECT_NEAR(gaffa::downsampled_variance(16, 4.0), 10.0, 1.0e-12);
  EXPECT_NEAR(gaffa::downsampled_variance(8, 2.5), 2.5 - 1.0 / 3.0,
              1.0e-12);
  EXPECT_NEAR(gaffa::downsampled_variance(5, 2.5), 5.0 / 3.0, 1.0e-12);
}
