#include "gaffa/ffa.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace {

std::vector<float> flatten(const std::vector<std::vector<float>>& rows) {
  std::vector<float> values;
  for (const auto& row : rows) {
    values.insert(values.end(), row.begin(), row.end());
  }
  return values;
}

std::vector<float> rotate_rows(const std::vector<float>& values,
                               std::size_t rows,
                               std::size_t bins,
                               std::size_t shift) {
  std::vector<float> rotated(values.size());
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t bin = 0; bin < bins; ++bin) {
      rotated[row * bins + ((bin + shift) % bins)] = values[row * bins + bin];
    }
  }
  return rotated;
}

std::vector<float> append_zero_columns(const std::vector<float>& values,
                                       std::size_t rows,
                                       std::size_t bins,
                                       std::size_t extra_bins) {
  std::vector<float> expanded(rows * (bins + extra_bins), 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t bin = 0; bin < bins; ++bin) {
      expanded[row * (bins + extra_bins) + bin] = values[row * bins + bin];
    }
  }
  return expanded;
}

const std::vector<float>& ffa_input_8x8() {
  static const std::vector<float> input = flatten({
      {0, 0, 0, 0, 0, 0, 0, 1},
      {0, 0, 0, 0, 0, 0, 0, 1},
      {0, 0, 0, 0, 0, 0, 0, 1},
      {0, 0, 0, 0, 0, 0, 0, 1},
      {0, 0, 0, 0, 0, 0, 0, 1},
      {0, 0, 0, 0, 0, 0, 0, 1},
      {0, 0, 0, 0, 0, 0, 0, 1},
      {0, 0, 0, 0, 0, 0, 0, 1},
  });
  return input;
}

const std::vector<float>& ffa_output_8x8() {
  static const std::vector<float> output = flatten({
      {0, 0, 0, 0, 0, 0, 0, 8},
      {0, 0, 0, 0, 0, 0, 4, 4},
      {0, 0, 0, 0, 0, 2, 4, 2},
      {0, 0, 0, 0, 2, 2, 2, 2},
      {0, 0, 0, 1, 2, 2, 2, 1},
      {0, 0, 1, 2, 1, 1, 2, 1},
      {0, 1, 1, 1, 2, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1},
  });
  return output;
}

}  // namespace

TEST(FfaTransformCpu, MatchesRiptideHandCalculatedEightByEight) {
  const auto result = gaffa::ffa_transform_cpu(ffa_input_8x8(),
                                               gaffa::FfaTransformPlan{.bins = 8});

  EXPECT_EQ(result.shape.rows, 8);
  EXPECT_EQ(result.shape.bins, 8);
  EXPECT_EQ(result.data, ffa_output_8x8());
}

TEST(FfaTransformCpu, MatchesOddRowGoldenTransform) {
  const std::vector<float> input = flatten({
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
      {17, 18, 19, 20},
  });
  const std::vector<float> expected = flatten({
      {45, 50, 55, 60},
      {46, 51, 56, 57},
      {50, 55, 56, 49},
      {52, 53, 54, 51},
      {51, 52, 53, 54},
  });

  const auto result =
      gaffa::ffa_transform_cpu(input, gaffa::FfaTransformPlan{.bins = 4});

  EXPECT_EQ(result.shape.rows, 5);
  EXPECT_EQ(result.shape.bins, 4);
  EXPECT_EQ(result.data, expected);
}

TEST(FfaTransformCpu, IsPhaseRotationInvariant) {
  constexpr std::size_t rows = 8;
  constexpr std::size_t bins = 8;

  for (std::size_t shift = 0; shift < bins; ++shift) {
    const auto input = rotate_rows(ffa_input_8x8(), rows, bins, shift);
    const auto expected = rotate_rows(ffa_output_8x8(), rows, bins, shift);

    const auto result =
        gaffa::ffa_transform_cpu(input, gaffa::FfaTransformPlan{.bins = bins});

    EXPECT_EQ(result.data, expected);
  }
}

TEST(FfaTransformCpu, PreservesExtraZeroColumns) {
  constexpr std::size_t rows = 8;
  constexpr std::size_t bins = 8;

  for (std::size_t extra_bins = 1; extra_bins < bins; ++extra_bins) {
    const auto input =
        append_zero_columns(ffa_input_8x8(), rows, bins, extra_bins);
    const auto expected =
        append_zero_columns(ffa_output_8x8(), rows, bins, extra_bins);

    const auto result = gaffa::ffa_transform_cpu(
        input, gaffa::FfaTransformPlan{.bins = bins + extra_bins});

    EXPECT_EQ(result.shape.rows, rows);
    EXPECT_EQ(result.shape.bins, bins + extra_bins);
    EXPECT_EQ(result.data, expected);
  }
}

TEST(FfaTransformCpu, IgnoresIncompleteTailSamples) {
  std::vector<float> time_series = ffa_input_8x8();
  time_series.push_back(42.0F);
  time_series.push_back(43.0F);

  const auto result =
      gaffa::ffa_transform_cpu(time_series, gaffa::FfaTransformPlan{.bins = 8});

  EXPECT_EQ(result.shape.rows, 8);
  EXPECT_EQ(result.shape.bins, 8);
  EXPECT_EQ(result.data, ffa_output_8x8());
}

TEST(FfaTransformCpu, ComputesTrialPeriodsForTransformRows) {
  const auto periods =
      gaffa::ffa_trial_periods(gaffa::FfaTransformShape{.rows = 42, .bins = 127},
                               0.003141592653589793);

  ASSERT_EQ(periods.size(), 42);
  for (std::size_t shift = 0; shift < periods.size(); ++shift) {
    const double expected =
        127.0 * 127.0 /
        (127.0 - static_cast<double>(shift) / 41.0) * 0.003141592653589793;
    EXPECT_DOUBLE_EQ(periods[shift], expected);
  }
}

TEST(FfaTransformCpu, ComputesSingleRowTrialPeriod) {
  const auto periods =
      gaffa::ffa_trial_periods(gaffa::FfaTransformShape{.rows = 1, .bins = 32},
                               0.25);

  ASSERT_EQ(periods.size(), 1);
  EXPECT_DOUBLE_EQ(periods[0], 8.0);
}

TEST(FfaTransformCpu, RejectsInvalidTransformArguments) {
  const std::vector<float> data{1, 2, 3, 4};

  EXPECT_THROW((void)gaffa::ffa_transform_cpu(
                   data, gaffa::FfaTransformPlan{.bins = 0}),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::ffa_transform_cpu(
                   data, gaffa::FfaTransformPlan{.bins = 1}),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::ffa_transform_cpu(
                   data, gaffa::FfaTransformPlan{.bins = 5}),
               std::invalid_argument);
}

TEST(FfaTransformCpu, RejectsInvalidTrialPeriodArguments) {
  EXPECT_THROW((void)gaffa::ffa_trial_periods(
                   gaffa::FfaTransformShape{.rows = 0, .bins = 8}, 1.0),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::ffa_trial_periods(
                   gaffa::FfaTransformShape{.rows = 8, .bins = 1}, 1.0),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::ffa_trial_periods(
                   gaffa::FfaTransformShape{.rows = 8, .bins = 8}, 0.0),
               std::invalid_argument);
}
