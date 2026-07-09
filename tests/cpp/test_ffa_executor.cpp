#include "gaffa/ffa_executor.h"

#include "gaffa/time_series.h"

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

std::vector<float> run_manual_transform(std::span<const float> input,
                                        gaffa::FfaTransformShape shape) {
  std::vector<float> scratch(input.size());
  std::vector<float> output(input.size());
  gaffa::ffa_transform_block_cpu(input, shape, scratch, output);
  return output;
}

}  // namespace

TEST(FfaExecutorCpu, RejectsInvalidInputs) {
  const std::vector<float> input{1, 2, 3, 4};
  const gaffa::FfaSearchPlan plan{
      .tasks = {make_task(input.size(), 1.0, input.size(), 2, 2, 2)},
  };

  EXPECT_THROW(gaffa::for_each_ffa_block_cpu({}, plan, [](const auto&) {}),
               std::invalid_argument);
  EXPECT_THROW(gaffa::for_each_ffa_block_cpu(input, {}, [](const auto&) {}),
               std::invalid_argument);
  EXPECT_THROW(gaffa::for_each_ffa_block_cpu(input, plan, {}),
               std::invalid_argument);
}

TEST(FfaExecutorCpu, RejectsInvalidTask) {
  const std::vector<float> input{1, 2, 3, 4};

  auto task = make_task(input.size(), 1.0, input.size(), 2, 2, 2);
  task.input_nsamples = input.size() + 1;
  EXPECT_THROW(gaffa::for_each_ffa_block_cpu(
                   input, gaffa::FfaSearchPlan{.tasks = {task}},
                   [](const auto&) {}),
               std::invalid_argument);

  task = make_task(input.size(), 1.0, input.size(), 2, 3, 2);
  EXPECT_THROW(gaffa::for_each_ffa_block_cpu(
                   input, gaffa::FfaSearchPlan{.tasks = {task}},
                   [](const auto&) {}),
               std::invalid_argument);

  task = make_task(input.size(), 1.0, input.size(), 2, 2, 2);
  task.effective_tsamp = 0.0;
  EXPECT_THROW(gaffa::for_each_ffa_block_cpu(
                   input, gaffa::FfaSearchPlan{.tasks = {task}},
                   [](const auto&) {}),
               std::invalid_argument);
}

TEST(FfaExecutorCpu, RunsSingleTaskWithoutDownsample) {
  const std::vector<float> input{
      1, 0,
      2, 0,
      3, 0,
      4, 0,
  };
  const auto task = make_task(input.size(), 1.0, input.size(), 4, 2, 2);
  const gaffa::FfaSearchPlan plan{.tasks = {task}};

  int calls = 0;
  gaffa::FfaBlockView observed;
  std::vector<float> observed_transform;
  gaffa::for_each_ffa_block_cpu(input, plan, [&](const gaffa::FfaBlockView& block) {
    ++calls;
    observed = block;
    observed_transform.assign(block.transform.begin(), block.transform.end());
  });

  const auto expected_full =
      run_manual_transform(input, gaffa::FfaTransformShape{.rows = 4, .bins = 2});

  EXPECT_EQ(calls, 1);
  EXPECT_EQ(observed.task, &plan.tasks.front());
  EXPECT_EQ(observed.shape.rows, 2);
  EXPECT_EQ(observed.shape.bins, 2);
  EXPECT_EQ(observed_transform,
            (std::vector<float>(expected_full.begin(), expected_full.begin() + 4)));
  EXPECT_FLOAT_EQ(observed.stdnoise, std::sqrt(4.0F));
}

TEST(FfaExecutorCpu, RunsSingleTaskWithWeightedDownsample) {
  const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8};
  const auto task = make_task(input.size(), 2.0, 4, 2, 2, 2);
  const gaffa::FfaSearchPlan plan{.tasks = {task}};

  std::vector<float> observed_transform;
  float observed_stdnoise = 0.0F;
  gaffa::for_each_ffa_block_cpu(input, plan, [&](const gaffa::FfaBlockView& block) {
    observed_transform.assign(block.transform.begin(), block.transform.end());
    observed_stdnoise = block.stdnoise;
  });

  const auto downsampled = gaffa::downsample_weighted_sum_cpu(input, 2.0);
  const auto expected_full = run_manual_transform(
      downsampled, gaffa::FfaTransformShape{.rows = 2, .bins = 2});

  EXPECT_EQ(observed_transform, expected_full);
  EXPECT_FLOAT_EQ(
      observed_stdnoise,
      static_cast<float>(std::sqrt(2.0 * gaffa::downsampled_variance(8, 2.0))));
}

TEST(FfaExecutorCpu, CallsConsumerForEachTask) {
  const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8};
  const gaffa::FfaSearchPlan plan{
      .tasks = {
          make_task(input.size(), 1.0, input.size(), 4, 4, 2),
          make_task(input.size(), 2.0, 4, 2, 1, 2),
      },
  };

  std::vector<gaffa::FfaTransformShape> shapes;
  gaffa::for_each_ffa_block_cpu(input, plan, [&](const gaffa::FfaBlockView& block) {
    shapes.push_back(block.shape);
  });

  ASSERT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[0].rows, 4);
  EXPECT_EQ(shapes[0].bins, 2);
  EXPECT_EQ(shapes[1].rows, 1);
  EXPECT_EQ(shapes[1].bins, 2);
}

TEST(FfaExecutorCpu, ReusesPreparedDownsampleForAdjacentTasks) {
  const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8};
  const gaffa::FfaSearchPlan plan{
      .tasks = {
          make_task(input.size(), 2.0, 4, 2, 2, 2),
          make_task(input.size(), 2.0, 4, 1, 1, 4),
      },
  };

  std::vector<std::vector<float>> observed;
  gaffa::for_each_ffa_block_cpu(input, plan, [&](const gaffa::FfaBlockView& block) {
    observed.emplace_back(block.transform.begin(), block.transform.end());
  });

  const auto downsampled = gaffa::downsample_weighted_sum_cpu(input, 2.0);
  const auto first_expected = run_manual_transform(
      downsampled, gaffa::FfaTransformShape{.rows = 2, .bins = 2});
  const auto second_expected = run_manual_transform(
      downsampled, gaffa::FfaTransformShape{.rows = 1, .bins = 4});

  ASSERT_EQ(observed.size(), 2);
  EXPECT_EQ(observed[0], first_expected);
  EXPECT_EQ(observed[1], second_expected);
}

TEST(FfaExecutorCpu, RowExecutorMatchesMaterializedBlockRows) {
  const std::vector<float> input{
      1, 0,
      2, 0,
      3, 0,
      4, 0,
  };
  const auto task = make_task(input.size(), 1.0, input.size(), 4, 2, 2);
  const gaffa::FfaSearchPlan plan{.tasks = {task}};

  std::vector<std::vector<float>> rows;
  std::vector<std::size_t> shifts;
  gaffa::for_each_ffa_row_cpu(input, plan, [&](const gaffa::FfaRowView& row) {
    EXPECT_EQ(row.task, &plan.tasks.front());
    EXPECT_FLOAT_EQ(row.stdnoise, std::sqrt(4.0F));
    shifts.push_back(row.shift);
    rows.emplace_back(row.profile.begin(), row.profile.end());
  });

  const auto expected_full =
      run_manual_transform(input, gaffa::FfaTransformShape{.rows = 4, .bins = 2});

  ASSERT_EQ(rows.size(), 2);
  EXPECT_EQ(shifts, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(rows[0], (std::vector<float>(expected_full.begin(),
                                         expected_full.begin() + 2)));
  EXPECT_EQ(rows[1], (std::vector<float>(expected_full.begin() + 2,
                                         expected_full.begin() + 4)));
}

TEST(FfaExecutorCpu, RowExecutorMatchesDownsampledBlockRows) {
  const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8};
  const auto task = make_task(input.size(), 2.0, 4, 2, 2, 2);
  const gaffa::FfaSearchPlan plan{.tasks = {task}};

  std::vector<std::vector<float>> rows;
  gaffa::for_each_ffa_row_cpu(input, plan, [&](const gaffa::FfaRowView& row) {
    rows.emplace_back(row.profile.begin(), row.profile.end());
  });

  const auto downsampled = gaffa::downsample_weighted_sum_cpu(input, 2.0);
  const auto expected_full = run_manual_transform(
      downsampled, gaffa::FfaTransformShape{.rows = 2, .bins = 2});

  ASSERT_EQ(rows.size(), 2);
  EXPECT_EQ(rows[0], (std::vector<float>(expected_full.begin(),
                                         expected_full.begin() + 2)));
  EXPECT_EQ(rows[1], (std::vector<float>(expected_full.begin() + 2,
                                         expected_full.begin() + 4)));
}
