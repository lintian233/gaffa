#include "gaffa/ffa_detection.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

gaffa::FfaSearchTask task_for_shape(gaffa::FfaTransformShape shape) {
  return gaffa::FfaSearchTask{
      .downsample_factor = 1.0,
      .effective_tsamp = 0.25,
      .input_nsamples = shape.rows * shape.bins,
      .prepared_nsamples = shape.rows * shape.bins,
      .bins = shape.bins,
      .rows = shape.rows,
      .rows_eval = shape.rows,
      .period_begin = 0.25 * static_cast<double>(shape.bins),
      .period_end = 0.25 * static_cast<double>(shape.bins + 1),
  };
}

float expected_riptide_snr(std::size_t bins,
                           std::size_t width,
                           float max_boxcar_sum,
                           float profile_sum,
                           float stdnoise) {
  const auto bins_f = static_cast<float>(bins);
  const auto width_f = static_cast<float>(width);
  const float height = std::sqrt((bins_f - width_f) / (bins_f * width_f));
  const float baseline = width_f / (bins_f - width_f) * height;
  return ((height + baseline) * max_boxcar_sum - baseline * profile_sum) /
         stdnoise;
}

}  // namespace

TEST(FfaDetectionCpu, RejectsInvalidShape) {
  const std::vector<float> transform{1.0F, 2.0F, 3.0F, 4.0F};
  const auto task = task_for_shape({.rows = 2, .bins = 2});
  const std::vector<std::size_t> widths{1};

  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(
                   transform, {.rows = 0, .bins = 2}, task, widths, 1.0F),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(
                   transform, {.rows = 2, .bins = 1}, task, widths, 1.0F),
               std::invalid_argument);
}

TEST(FfaDetectionCpu, RejectsTransformSizeMismatch) {
  const std::vector<float> transform{1.0F, 2.0F, 3.0F};
  const gaffa::FfaTransformShape shape{.rows = 2, .bins = 2};
  const auto task = task_for_shape(shape);
  const std::vector<std::size_t> widths{1};

  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(transform, shape, task, widths,
                                               1.0F),
               std::invalid_argument);
}

TEST(FfaDetectionCpu, RejectsTaskShapeMismatch) {
  const gaffa::FfaTransformShape shape{.rows = 2, .bins = 4};
  const std::vector<float> transform(shape.rows * shape.bins, 0.0F);
  const std::vector<std::size_t> widths{1};

  auto task = task_for_shape(shape);
  task.bins = 5;
  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(transform, shape, task, widths,
                                               1.0F),
               std::invalid_argument);

  task = task_for_shape(shape);
  task.rows_eval = 1;
  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(transform, shape, task, widths,
                                               1.0F),
               std::invalid_argument);

  task = task_for_shape(shape);
  task.rows = 1;
  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(transform, shape, task, widths,
                                               1.0F),
               std::invalid_argument);

  task = task_for_shape(shape);
  task.effective_tsamp = 0.0;
  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(transform, shape, task, widths,
                                               1.0F),
               std::invalid_argument);
}

TEST(FfaDetectionCpu, RejectsInvalidStdnoiseAndOptions) {
  const gaffa::FfaTransformShape shape{.rows = 1, .bins = 4};
  const std::vector<float> transform(shape.rows * shape.bins, 0.0F);
  const auto task = task_for_shape(shape);
  const std::vector<std::size_t> widths{1};

  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(transform, shape, task, widths,
                                               0.0F),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(transform, shape, task, widths,
                                               INFINITY),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(
                   transform, shape, task, widths, 1.0F,
                   gaffa::FfaDetectionOptions{.snr_threshold = INFINITY}),
               std::invalid_argument);
}

TEST(FfaDetectionCpu, RejectsInvalidWidths) {
  const gaffa::FfaTransformShape shape{.rows = 1, .bins = 4};
  const std::vector<float> transform(shape.rows * shape.bins, 0.0F);
  const auto task = task_for_shape(shape);

  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(
                   transform, shape, task, std::span<const std::size_t>{},
                   1.0F),
               std::invalid_argument);

  const std::vector<std::size_t> zero_width{0};
  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(transform, shape, task,
                                               zero_width, 1.0F),
               std::invalid_argument);

  const std::vector<std::size_t> full_width{4};
  EXPECT_THROW((void)gaffa::find_ffa_peaks_cpu(transform, shape, task,
                                               full_width, 1.0F),
               std::invalid_argument);
}

TEST(FfaDetectionCpu, DetectsSingleBinPeak) {
  const gaffa::FfaTransformShape shape{.rows = 1, .bins = 4};
  const std::vector<float> transform{0.0F, 0.0F, 5.0F, 0.0F};
  const auto task = task_for_shape(shape);
  const std::vector<std::size_t> widths{1};

  const auto peaks = gaffa::find_ffa_peaks_cpu(
      transform, shape, task, widths, 1.0F,
      gaffa::FfaDetectionOptions{.snr_threshold = 0.0F});

  ASSERT_EQ(peaks.size(), 1);
  EXPECT_EQ(peaks[0].width, 1);
  EXPECT_EQ(peaks[0].width_index, 0);
  EXPECT_EQ(peaks[0].period_index, 0);
  EXPECT_EQ(peaks[0].phase, 2);
  EXPECT_EQ(peaks[0].shift, 0);
  EXPECT_EQ(peaks[0].bins, 4);
  EXPECT_DOUBLE_EQ(peaks[0].period, 1.0);
  EXPECT_DOUBLE_EQ(peaks[0].frequency, 1.0);
  EXPECT_DOUBLE_EQ(peaks[0].duty_cycle, 0.25);
  EXPECT_FLOAT_EQ(peaks[0].snr,
                  expected_riptide_snr(4, 1, 5.0F, 5.0F, 1.0F));
}

TEST(FfaDetectionCpu, DetectsCircularBoxcarAcrossBoundary) {
  const gaffa::FfaTransformShape shape{.rows = 1, .bins = 4};
  const std::vector<float> transform{4.0F, 0.0F, 0.0F, 5.0F};
  const auto task = task_for_shape(shape);
  const std::vector<std::size_t> widths{2};

  const auto peaks = gaffa::find_ffa_peaks_cpu(
      transform, shape, task, widths, 1.0F,
      gaffa::FfaDetectionOptions{.snr_threshold = 0.0F});

  ASSERT_EQ(peaks.size(), 1);
  EXPECT_EQ(peaks[0].phase, 3);
  EXPECT_FLOAT_EQ(peaks[0].snr,
                  expected_riptide_snr(4, 2, 9.0F, 9.0F, 1.0F));
}

TEST(FfaDetectionCpu, ComputesPeriodFromTaskRowsNotRowsEval) {
  const gaffa::FfaTransformShape shape{.rows = 4, .bins = 4};
  std::vector<float> transform(shape.rows * shape.bins, 0.0F);
  transform[3 * shape.bins + 1] = 10.0F;
  auto task = task_for_shape(shape);
  task.rows = 8;
  task.rows_eval = shape.rows;
  const std::vector<std::size_t> widths{1};

  const auto peaks = gaffa::find_ffa_peaks_cpu(
      transform, shape, task, widths, 1.0F,
      gaffa::FfaDetectionOptions{
          .snr_threshold = 0.0F,
      });

  ASSERT_FALSE(peaks.empty());
  const auto peak = std::find_if(
      peaks.begin(), peaks.end(),
      [](const gaffa::FfaPeak& item) { return item.shift == 3; });
  ASSERT_NE(peak, peaks.end());
  const double expected_period = 0.25 * 4.0 * 4.0 / (4.0 - 3.0 / 7.0);
  EXPECT_DOUBLE_EQ(peak->period, expected_period);
}

TEST(FfaDetectionCpu, SortsPeaksBySnrDescending) {
  const gaffa::FfaTransformShape shape{.rows = 2, .bins = 4};
  const std::vector<float> transform{
      0.0F, 0.0F, 2.0F, 0.0F,
      0.0F, 0.0F, 8.0F, 0.0F,
  };
  const auto task = task_for_shape(shape);
  const std::vector<std::size_t> widths{1};

  const auto peaks = gaffa::find_ffa_peaks_cpu(
      transform, shape, task, widths, 1.0F,
      gaffa::FfaDetectionOptions{
          .snr_threshold = 0.0F,
      });

  ASSERT_EQ(peaks.size(), 2);
  EXPECT_GT(peaks[0].snr, peaks[1].snr);
  EXPECT_EQ(peaks[0].shift, 1);
}

TEST(FfaDetectionCpu, ReturnsRawAdjacentFrequencyTrialsWithinWidth) {
  const gaffa::FfaTransformShape shape{.rows = 3, .bins = 4};
  const std::vector<float> transform{
      0.0F, 0.0F, 2.0F, 0.0F,
      0.0F, 0.0F, 8.0F, 0.0F,
      0.0F, 0.0F, 4.0F, 0.0F,
  };
  const auto task = task_for_shape(shape);
  const std::vector<std::size_t> widths{1};

  const auto peaks = gaffa::find_ffa_peaks_cpu(
      transform, shape, task, widths, 1.0F,
      gaffa::FfaDetectionOptions{
          .snr_threshold = 0.0F,
      });

  ASSERT_EQ(peaks.size(), 3);
  EXPECT_EQ(peaks.front().shift, 1);
}

TEST(FfaDetectionCpu, DoesNotClusterAcrossWidthTrials) {
  const gaffa::FfaTransformShape shape{.rows = 1, .bins = 4};
  const std::vector<float> transform{0.0F, 0.0F, 5.0F, 0.0F};
  const auto task = task_for_shape(shape);
  const std::vector<std::size_t> widths{1, 2};

  const auto peaks = gaffa::find_ffa_peaks_cpu(
      transform, shape, task, widths, 1.0F,
      gaffa::FfaDetectionOptions{.snr_threshold = 0.0F});

  ASSERT_EQ(peaks.size(), 2);
  EXPECT_NE(peaks[0].width, peaks[1].width);
}

TEST(FfaDetectionCpu, FiltersBelowThreshold) {
  const gaffa::FfaTransformShape shape{.rows = 1, .bins = 4};
  const std::vector<float> transform{0.0F, 0.0F, 5.0F, 0.0F};
  const auto task = task_for_shape(shape);
  const std::vector<std::size_t> widths{1};

  const auto peaks = gaffa::find_ffa_peaks_cpu(
      transform, shape, task, widths, 1.0F,
      gaffa::FfaDetectionOptions{.snr_threshold = 100.0F});

  EXPECT_TRUE(peaks.empty());
}
