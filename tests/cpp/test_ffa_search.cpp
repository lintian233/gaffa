#include "gaffa/ffa_search.h"

#include "gaffa/ffa_detection.h"
#include "gaffa/ffa_executor.h"

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

TEST(FfaSearchCpu, ReturnsEmptyWhenNoPeakPassesThreshold) {
  const std::vector<float> input{1, 0, 2, 0, 3, 0, 4, 0};
  const auto plan = single_task_plan(input.size());

  const auto result = gaffa::search_ffa_cpu(
      input, plan,
      gaffa::FfaSearchOptions{
          .snr_threshold = 1000.0F,
      });

  EXPECT_TRUE(result.peaks.empty());
}

TEST(FfaSearchCpu, FindsPeaksThroughExecutorAndDetection) {
  const std::vector<float> input{0, 0, 0, 5, 0, 0, 0, 5};
  const auto plan = single_task_plan(input.size());

  const auto result = gaffa::search_ffa_cpu(
      input, plan,
      gaffa::FfaSearchOptions{
          .snr_threshold = 0.0F,
      });

  ASSERT_FALSE(result.peaks.empty());
  EXPECT_EQ(result.peaks.front().width, 1);
  EXPECT_EQ(result.peaks.front().bins, 2);
}

TEST(FfaSearchCpu, CollectsPeaksAcrossBlocks) {
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
      });

  ASSERT_GE(result.peaks.size(), 2);
  EXPECT_GT(result.peaks.front().snr, 0.0F);
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
      });

  ASSERT_FALSE(result.peaks.empty());
  EXPECT_EQ(result.peaks.front().bins, 2);
  EXPECT_EQ(result.peaks.front().width, 1);
  EXPECT_DOUBLE_EQ(result.peaks.front().period, 4.0);
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

  ASSERT_FALSE(result.peaks.empty());
  EXPECT_EQ(result.peaks.front().bins, 2);
}

TEST(FfaSearchCpu, MatchesMaterializedBlockReference) {
  const std::vector<float> input{0, 0, 3, 0, 1, 0, 3, 0,
                                 0, 4, 0, 1, 0, 4, 0, 1};
  const gaffa::FfaSearchPlan plan{
      .tasks = {
          make_task(input.size(), 1.0, input.size(), 4, 4, 4),
          make_task(input.size(), 2.0, 8, 2, 2, 4),
      },
      .width_trials = {1, 2},
  };
  const gaffa::FfaSearchOptions search_options{
      .snr_threshold = 0.0F,
  };

  const auto result = gaffa::search_ffa_cpu(input, plan, search_options);

  std::vector<gaffa::FfaPeak> reference;
  const gaffa::FfaDetectionOptions detection_options{
      .snr_threshold = search_options.snr_threshold,
      .max_peaks = search_options.max_peaks,
  };
  gaffa::for_each_ffa_block_cpu(
      input, plan, [&](const gaffa::FfaBlockView& block) {
        const auto peaks = gaffa::find_ffa_peaks_cpu(
            block.transform, block.shape, *block.task, plan.width_trials,
            block.stdnoise, detection_options);
        reference.insert(reference.end(), peaks.begin(), peaks.end());
      });
  gaffa::sort_ffa_peaks(reference);

  ASSERT_EQ(result.peaks.size(), reference.size());
  for (std::size_t index = 0; index < reference.size(); ++index) {
    EXPECT_EQ(result.peaks[index].width, reference[index].width);
    EXPECT_EQ(result.peaks[index].width_index, reference[index].width_index);
    EXPECT_EQ(result.peaks[index].period_index,
              reference[index].period_index);
    EXPECT_EQ(result.peaks[index].phase, reference[index].phase);
    EXPECT_EQ(result.peaks[index].shift, reference[index].shift);
    EXPECT_EQ(result.peaks[index].bins, reference[index].bins);
    EXPECT_DOUBLE_EQ(result.peaks[index].period, reference[index].period);
    EXPECT_DOUBLE_EQ(result.peaks[index].frequency,
                     reference[index].frequency);
    EXPECT_FLOAT_EQ(result.peaks[index].snr, reference[index].snr);
  }
}
