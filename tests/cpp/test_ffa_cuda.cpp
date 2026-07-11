#include "gaffa/ffa_cuda.h"

#include "gaffa/cuda_memory.h"
#include "gaffa/ffa.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/time_series.h"
#include "gaffa/vector_add.hpp"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
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

gaffa::FfaSearchPlan valid_plan() {
  return gaffa::FfaSearchPlan{
      .tasks = {
          make_task(2048, 1.0, 2048, 16, 16, 64),
          make_task(2048, 2.0, 1024, 8, 8, 128),
      },
      .width_trials = {1, 2, 3},
  };
}

gaffa::FfaSearchPlan grouped_plan() {
  return gaffa::FfaSearchPlan{
      .tasks = {
          make_task(64, 1.0, 64, 4, 4, 8),
          make_task(64, 2.0, 32, 4, 4, 8),
          make_task(64, 1.0, 64, 8, 8, 8),
      },
      .width_trials = {1, 2, 4},
  };
}

gaffa::CudaTimeSeriesBatchView valid_input(const float* data) {
  return gaffa::CudaTimeSeriesBatchView{
      .data = data,
      .nseries = 1,
      .nsamples = 2048,
      .tsamp = 1.0,
      .device_id = 0,
  };
}

bool has_cuda_device() {
  return gaffa::cuda_device_count() != 0;
}

gaffa::CudaSpan<float> fake_cuda_span(float* data,
                                      std::size_t count,
                                      int device_id = 0) {
  return gaffa::CudaSpan<float>{
      .data = data,
      .count = count,
      .device_id = device_id,
  };
}

std::vector<float> prepare_on_cuda(const std::vector<float>& input,
                                   std::size_t nseries,
                                   std::size_t nsamples,
                                   const gaffa::FfaSearchTask& task,
                                   gaffa::CudaFfaOptions options = {}) {
  gaffa::CudaDeviceBuffer<float> device_input(input.size());
  cudaMemcpy(device_input.data(), input.data(), input.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  const std::size_t output_size = nseries * task.prepared_nsamples;
  gaffa::CudaDeviceBuffer<float> device_output(output_size);
  gaffa::prepare_ffa_input_cuda(
      gaffa::CudaTimeSeriesBatchView{
          .data = device_input.data(),
          .nseries = nseries,
          .nsamples = nsamples,
          .tsamp = 1.0,
          .device_id = options.device_id,
      },
      task, device_output.as_span(options.device_id), options);

  std::vector<float> output(output_size);
  cudaMemcpy(output.data(), device_output.data(), output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  return output;
}

std::vector<float> expected_prepared(const std::vector<float>& input,
                                     std::size_t nseries,
                                     std::size_t nsamples,
                                     double factor) {
  std::vector<float> expected;
  const std::size_t prepared_nsamples =
      factor == 1.0 ? nsamples : gaffa::downsampled_size(nsamples, factor);
  expected.reserve(nseries * prepared_nsamples);
  for (std::size_t series = 0; series < nseries; ++series) {
    const auto row = std::span<const float>(input).subspan(
        series * nsamples, nsamples);
    if (factor == 1.0) {
      expected.insert(expected.end(), row.begin(), row.end());
    } else {
      const auto downsampled = gaffa::downsample_weighted_sum_cpu(row, factor);
      expected.insert(expected.end(), downsampled.begin(), downsampled.end());
    }
  }
  return expected;
}

std::vector<float> transform_on_cuda(const std::vector<float>& input,
                                     std::size_t nseries,
                                     gaffa::FfaTransformShape shape,
                                     gaffa::CudaFfaOptions options = {}) {
  gaffa::CudaDeviceBuffer<float> device_input(input.size());
  cudaMemcpy(device_input.data(), input.data(), input.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  const std::size_t series_elements = shape.rows * shape.bins;
  const std::size_t output_size = nseries * series_elements;
  gaffa::CudaDeviceBuffer<float> device_scratch(output_size);
  gaffa::CudaDeviceBuffer<float> device_output(output_size);
  gaffa::ffa_transform_block_cuda(
      gaffa::CudaFfaInput{
          .data = device_input.data(),
          .nseries = nseries,
          .nsamples = series_elements,
          .stride = series_elements,
          .shape = shape,
          .device_id = options.device_id,
      },
      gaffa::CudaFfaBuffer{
          .data = device_scratch.data(),
          .nseries = nseries,
          .stride = series_elements,
          .shape = shape,
          .device_id = options.device_id,
      },
      gaffa::CudaFfaBuffer{
          .data = device_output.data(),
          .nseries = nseries,
          .stride = series_elements,
          .shape = shape,
          .device_id = options.device_id,
      },
      options);

  std::vector<float> output(output_size);
  cudaMemcpy(output.data(), device_output.data(), output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  return output;
}

std::vector<float> transform_with_program_on_cuda(
    const gaffa::CudaFfaProgram& program,
    std::size_t group_index,
    std::size_t task_index,
    const std::vector<float>& input,
    std::size_t nseries,
    gaffa::FfaTransformShape shape,
    gaffa::CudaFfaOptions options = {}) {
  gaffa::CudaDeviceBuffer<float> device_input(input.size());
  cudaMemcpy(device_input.data(), input.data(), input.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  const std::size_t series_elements = shape.rows * shape.bins;
  const std::size_t output_size = nseries * series_elements;
  gaffa::CudaDeviceBuffer<float> device_scratch(output_size);
  gaffa::CudaDeviceBuffer<float> device_output(output_size);
  gaffa::ffa_transform_block_cuda(
      program, group_index, task_index,
      gaffa::CudaFfaInput{
          .data = device_input.data(),
          .nseries = nseries,
          .nsamples = series_elements,
          .stride = series_elements,
          .shape = shape,
          .device_id = options.device_id,
      },
      gaffa::CudaFfaBuffer{
          .data = device_scratch.data(),
          .nseries = nseries,
          .stride = series_elements,
          .shape = shape,
          .device_id = options.device_id,
      },
      gaffa::CudaFfaBuffer{
          .data = device_output.data(),
          .nseries = nseries,
          .stride = series_elements,
          .shape = shape,
          .device_id = options.device_id,
      },
      options);

  std::vector<float> output(output_size);
  cudaMemcpy(output.data(), device_output.data(), output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  return output;
}

std::vector<float> expected_transform(const std::vector<float>& input,
                                      std::size_t nseries,
                                      gaffa::FfaTransformShape shape) {
  const std::size_t series_elements = shape.rows * shape.bins;
  std::vector<float> expected(input.size());
  std::vector<float> scratch(series_elements);
  for (std::size_t series = 0; series < nseries; ++series) {
    const auto input_series = std::span<const float>(input).subspan(
        series * series_elements, series_elements);
    auto output_series =
        std::span<float>(expected).subspan(series * series_elements,
                                           series_elements);
    gaffa::ffa_transform_block_cpu(input_series, shape, scratch, output_series);
  }
  return expected;
}

void expect_transform_matches_cpu(const std::vector<float>& input,
                                  std::size_t nseries,
                                  gaffa::FfaTransformShape shape) {
  const auto output = transform_on_cuda(input, nseries, shape);
  const auto expected = expected_transform(input, nseries, shape);
  ASSERT_EQ(output.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(output[index], expected[index]) << "index=" << index;
  }
}

const gaffa::CudaFfaPrepareGroup* find_group(
    const gaffa::CudaFfaExecutionPlan& plan,
    double downsample_factor) {
  const auto group = std::find_if(
      plan.groups().begin(), plan.groups().end(),
      [&](const gaffa::CudaFfaPrepareGroup& candidate) {
        return candidate.prepare_key.downsample_factor == downsample_factor;
      });
  if (group == plan.groups().end()) {
    return nullptr;
  }
  return &*group;
}

}  // namespace

TEST(FfaCuda, CompilesExecutionPlanGroupedByDownsampleFactor) {
  const auto compiled = gaffa::make_ffa_cuda_execution_plan(grouped_plan());

  ASSERT_EQ(compiled.groups().size(), 2);
  ASSERT_NE(find_group(compiled, 1.0), nullptr);
  ASSERT_NE(find_group(compiled, 2.0), nullptr);
  EXPECT_EQ(find_group(compiled, 1.0)->prepare_key.input_nsamples, 64);
  EXPECT_EQ(find_group(compiled, 2.0)->prepare_key.input_nsamples, 64);
  EXPECT_EQ(find_group(compiled, 1.0)->tasks.size(), 2);
  EXPECT_EQ(find_group(compiled, 2.0)->tasks.size(), 1);
}

TEST(FfaCuda, ExecutionPlanTracksPreparedSizes) {
  const auto compiled = gaffa::make_ffa_cuda_execution_plan(grouped_plan());

  ASSERT_NE(find_group(compiled, 1.0), nullptr);
  ASSERT_NE(find_group(compiled, 2.0), nullptr);
  EXPECT_EQ(find_group(compiled, 1.0)->prepared_nsamples, 64);
  EXPECT_EQ(find_group(compiled, 2.0)->prepared_nsamples, 32);
  EXPECT_EQ(compiled.max_prepared_nsamples(), 64);
  EXPECT_EQ(compiled.max_transform_elements(), 64);
}

TEST(FfaCuda, ExecutionPlanTracksTaskShapeAndElements) {
  const auto plan = grouped_plan();
  const auto compiled = gaffa::make_ffa_cuda_execution_plan(plan);

  const auto* group = find_group(compiled, 1.0);
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->tasks.size(), 2);
  EXPECT_EQ(group->tasks[0].shape.rows, plan.tasks[0].rows);
  EXPECT_EQ(group->tasks[0].shape.bins, plan.tasks[0].bins);
  EXPECT_EQ(group->tasks[0].transform_elements,
            plan.tasks[0].rows * plan.tasks[0].bins);
  EXPECT_EQ(group->tasks[1].shape.rows, plan.tasks[2].rows);
  EXPECT_EQ(group->tasks[1].shape.bins, plan.tasks[2].bins);
  EXPECT_EQ(group->tasks[1].transform_elements,
            plan.tasks[2].rows * plan.tasks[2].bins);
}

TEST(FfaCuda, CompileExecutionPlanRejectsInvalidPlan) {
  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(gaffa::FfaSearchPlan{}),
               std::invalid_argument);

  auto plan = grouped_plan();
  plan.width_trials.clear();
  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(plan),
               std::invalid_argument);

  plan = grouped_plan();
  plan.tasks.front().rows = 0;
  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(plan),
               std::invalid_argument);

  plan = grouped_plan();
  plan.tasks.front().bins = 1;
  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(plan),
               std::invalid_argument);

  plan = grouped_plan();
  plan.tasks.front().prepared_nsamples = 0;
  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(plan),
               std::invalid_argument);

  plan = grouped_plan();
  plan.tasks[1].prepared_nsamples = 31;
  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(plan),
               std::invalid_argument);

  plan = grouped_plan();
  plan.tasks.front().rows = 9;
  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(plan),
               std::invalid_argument);

  plan = grouped_plan();
  plan.tasks.front().downsample_factor = 0.0;
  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(plan),
               std::invalid_argument);

  plan = grouped_plan();
  plan.tasks.front().downsample_factor =
      std::numeric_limits<double>::infinity();
  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(plan),
               std::invalid_argument);

  plan = grouped_plan();
  plan.tasks.front().downsample_factor =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(plan),
               std::invalid_argument);
}

TEST(FfaCuda, ExecutionPlanRejectsTasksWithDifferentInputLengths) {
  auto plan = grouped_plan();
  plan.tasks.back().input_nsamples = 128;
  plan.tasks.back().prepared_nsamples = 128;
  plan.tasks.back().rows = 16;

  EXPECT_THROW((void)gaffa::make_ffa_cuda_execution_plan(plan),
               std::invalid_argument);
}

TEST(FfaCuda, ExecutionPlanPreservesRiptidePrepareReuseContract) {
  constexpr std::size_t nsamples = 4096;
  const auto logical_plan = gaffa::make_riptide_ffa_plan(
      nsamples, 1.0,
      gaffa::RiptideFfaPlanOptions{
          .period_min = 32.0,
          .period_max = 256.0,
          .bins_min = 8,
          .bins_max = 16,
          .min_periods = 2,
      });
  const auto execution_plan =
      gaffa::make_ffa_cuda_execution_plan(logical_plan);

  ASSERT_FALSE(execution_plan.groups().empty());
  for (const auto& group : execution_plan.groups()) {
    EXPECT_EQ(group.prepare_key.input_nsamples, nsamples);
    ASSERT_FALSE(group.tasks.empty());
    for (const auto& task : group.tasks) {
      EXPECT_EQ(task.task.input_nsamples, group.prepare_key.input_nsamples);
      EXPECT_EQ(task.task.downsample_factor,
                group.prepare_key.downsample_factor);
      EXPECT_EQ(task.task.prepared_nsamples, group.prepared_nsamples);
    }
  }
}

TEST(FfaCuda, ProgramRejectsInvalidOptionsBeforeCudaAllocation) {
  const auto execution_plan =
      gaffa::make_ffa_cuda_execution_plan(grouped_plan());

  EXPECT_THROW(
      (void)gaffa::CudaFfaProgram(
          execution_plan, gaffa::CudaFfaOptions{.series_tile_size = 0}),
      std::invalid_argument);
  EXPECT_THROW(
      (void)gaffa::CudaFfaProgram(
          execution_plan, gaffa::CudaFfaOptions{.threads_per_block = 0}),
      std::invalid_argument);
}

TEST(FfaCuda, ProgramExposesExecutionPlan) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const auto plan = grouped_plan();
  const auto expected = gaffa::make_ffa_cuda_execution_plan(plan);
  const gaffa::CudaFfaProgram program(plan);

  EXPECT_EQ(program.device_id(), 0);
  EXPECT_EQ(program.execution_plan().groups().size(),
            expected.groups().size());
  EXPECT_EQ(program.execution_plan().max_prepared_nsamples(),
            expected.max_prepared_nsamples());
  EXPECT_EQ(program.execution_plan().max_transform_elements(),
            expected.max_transform_elements());
  EXPECT_GT(program.device_metadata_bytes(), 0);
  ASSERT_EQ(program.execution_plan().groups().size(), 2);
  EXPECT_EQ(program.execution_plan().groups()[0].tasks.size(), 2);
  EXPECT_EQ(program.execution_plan().groups()[1].tasks.size(), 1);
}

TEST(FfaCuda, ProgramAcceptsExecutionPlanAndSurvivesMove) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  gaffa::CudaFfaProgram source(
      gaffa::make_ffa_cuda_execution_plan(grouped_plan()));
  gaffa::CudaFfaProgram program(std::move(source));

  EXPECT_TRUE(source.empty());
  EXPECT_FALSE(program.empty());
  EXPECT_THROW((void)source.execution_plan(), std::logic_error);
  EXPECT_EQ(program.execution_plan().groups().size(), 2);

  const auto shape = program.execution_plan().groups()[0].tasks[0].shape;
  const std::vector<float> input{
      1, 2, 3, 4, 5, 6, 7, 8,
      8, 7, 6, 5, 4, 3, 2, 1,
      2, 4, 6, 8, 10, 12, 14, 16,
      16, 14, 12, 10, 8, 6, 4, 2,
  };
  const auto output = transform_with_program_on_cuda(program, 0, 0, input, 1,
                                                      shape);
  EXPECT_EQ(output, expected_transform(input, 1, shape));
}

TEST(FfaCuda, ProgramTransformMatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const auto plan = grouped_plan();
  const gaffa::CudaFfaProgram program(plan);
  const auto shape = program.execution_plan().groups()[0].tasks[1].shape;
  const std::size_t nseries = 2;
  std::vector<float> input(nseries * shape.rows * shape.bins);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>((index * 11 + 5) % 29) - 14.0F;
  }

  const auto output =
      transform_with_program_on_cuda(program, 0, 1, input, nseries, shape);
  const auto expected = expected_transform(input, nseries, shape);
  ASSERT_EQ(output.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(output[index], expected[index]) << "index=" << index;
  }
}

TEST(FfaCuda, ProgramTransformAcceptsExplicitStream) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  const auto plan = grouped_plan();
  const gaffa::CudaFfaProgram program(plan);
  const auto shape = program.execution_plan().groups()[1].tasks[0].shape;
  const std::size_t nseries = 2;
  std::vector<float> input(nseries * shape.rows * shape.bins);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>((index * 13 + 7) % 31) - 15.0F;
  }

  const auto output = transform_with_program_on_cuda(
      program, 1, 0, input, nseries, shape,
      gaffa::CudaFfaOptions{
          .stream = stream,
      });
  const auto expected = expected_transform(input, nseries, shape);
  ASSERT_EQ(output.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(output[index], expected[index]) << "index=" << index;
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(FfaCuda, ProgramTransformSupportsAsyncReturn) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  const auto plan = grouped_plan();
  const gaffa::CudaFfaProgram program(plan);
  const auto shape = program.execution_plan().groups()[0].tasks[0].shape;
  const std::size_t nseries = 2;
  const std::size_t series_elements = shape.rows * shape.bins;
  std::vector<float> input(nseries * series_elements);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>((index * 17 + 3) % 37) - 18.0F;
  }

  gaffa::CudaDeviceBuffer<float> device_input(input.size());
  cudaMemcpy(device_input.data(), input.data(), input.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  gaffa::CudaDeviceBuffer<float> device_scratch(input.size());
  gaffa::CudaDeviceBuffer<float> device_output(input.size());

  gaffa::ffa_transform_block_cuda(
      program, 0, 0,
      gaffa::CudaFfaInput{
          .data = device_input.data(),
          .nseries = nseries,
          .nsamples = series_elements,
          .stride = series_elements,
          .shape = shape,
          .device_id = 0,
      },
      gaffa::CudaFfaBuffer{
          .data = device_scratch.data(),
          .nseries = nseries,
          .stride = series_elements,
          .shape = shape,
          .device_id = 0,
      },
      gaffa::CudaFfaBuffer{
          .data = device_output.data(),
          .nseries = nseries,
          .stride = series_elements,
          .shape = shape,
          .device_id = 0,
      },
      gaffa::CudaFfaOptions{
          .stream = stream,
          .synchronize_after_call = false,
      });

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  std::vector<float> output(input.size());
  cudaMemcpy(output.data(), device_output.data(), output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  const auto expected = expected_transform(input, nseries, shape);
  ASSERT_EQ(output.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(output[index], expected[index]) << "index=" << index;
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(FfaCuda, ProgramTransformRejectsInvalidArguments) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const gaffa::CudaFfaProgram program(grouped_plan());
  float fake_data = 0.0F;
  float fake_scratch = 0.0F;
  float fake_output = 0.0F;
  const auto shape = program.execution_plan().groups()[0].tasks[0].shape;
  const auto input = gaffa::CudaFfaInput{
      .data = &fake_data,
      .nseries = 1,
      .nsamples = shape.rows * shape.bins,
      .stride = shape.rows * shape.bins,
      .shape = shape,
      .device_id = 0,
  };
  const auto scratch = gaffa::CudaFfaBuffer{
      .data = &fake_scratch,
      .nseries = 1,
      .stride = shape.rows * shape.bins,
      .shape = shape,
      .device_id = 0,
  };
  const auto output = gaffa::CudaFfaBuffer{
      .data = &fake_output,
      .nseries = 1,
      .stride = shape.rows * shape.bins,
      .shape = shape,
      .device_id = 0,
  };

  EXPECT_THROW(gaffa::ffa_transform_block_cuda(program, 2, 0, input, scratch,
                                               output),
               std::out_of_range);
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(program, 0, 2, input, scratch,
                                               output),
               std::out_of_range);
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(
                   program, 0, 0, input, scratch, output,
                   gaffa::CudaFfaOptions{.device_id = 1}),
               std::invalid_argument);

  auto wrong_shape = input;
  wrong_shape.shape.bins += 1;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(program, 0, 0, wrong_shape,
                                               scratch, output),
               std::invalid_argument);

  auto short_input = input;
  short_input.nsamples = input.nsamples - 1;
  short_input.stride = input.stride - 1;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(program, 0, 0, short_input,
                                               scratch, output),
               std::invalid_argument);
}

TEST(FfaCuda, EstimatesWorkspaceFromPlan) {
  const gaffa::CudaFfaOptions options{
      .series_tile_size = 4,
  };
  const auto plan = gaffa::make_ffa_cuda_execution_plan(valid_plan());

  const auto shape = gaffa::estimate_ffa_cuda_workspace(plan, options);

  EXPECT_EQ(shape.series_tile_size, 4);
  EXPECT_EQ(shape.max_prepared_nsamples, 2048);
  EXPECT_EQ(shape.max_task_elements, 1024);
  EXPECT_EQ(shape.prepared_bytes, 4 * 2048 * sizeof(float));
  EXPECT_EQ(shape.ping_bytes, 4 * 1024 * sizeof(float));
  EXPECT_EQ(shape.pong_bytes, 4 * 1024 * sizeof(float));
  EXPECT_EQ(shape.total_bytes,
            shape.prepared_bytes + shape.ping_bytes + shape.pong_bytes);
}

TEST(FfaCuda, EstimatesWorkspaceFromExecutionPlan) {
  const gaffa::CudaFfaOptions options{
      .series_tile_size = 4,
  };
  const auto execution_plan =
      gaffa::make_ffa_cuda_execution_plan(valid_plan());

  const auto shape =
      gaffa::estimate_ffa_cuda_workspace(execution_plan, options);

  EXPECT_EQ(shape.series_tile_size, 4);
  EXPECT_EQ(shape.max_prepared_nsamples,
            execution_plan.max_prepared_nsamples());
  EXPECT_EQ(shape.max_task_elements,
            execution_plan.max_transform_elements());
  EXPECT_EQ(shape.prepared_bytes,
            4 * execution_plan.max_prepared_nsamples() * sizeof(float));
  EXPECT_EQ(shape.ping_bytes,
            4 * execution_plan.max_transform_elements() * sizeof(float));
  EXPECT_EQ(shape.pong_bytes,
            4 * execution_plan.max_transform_elements() * sizeof(float));
}

TEST(FfaCuda, RejectsInvalidWorkspaceInputs) {
  const auto plan = gaffa::make_ffa_cuda_execution_plan(valid_plan());
  EXPECT_THROW(
      (void)gaffa::estimate_ffa_cuda_workspace(
          plan, gaffa::CudaFfaOptions{.series_tile_size = 0}),
      std::invalid_argument);
  EXPECT_THROW(
      (void)gaffa::estimate_ffa_cuda_workspace(
          plan, gaffa::CudaFfaOptions{.threads_per_block = 0}),
      std::invalid_argument);
}

TEST(FfaCuda, EnforcesWorkspaceByteLimit) {
  EXPECT_THROW(
      (void)gaffa::estimate_ffa_cuda_workspace(
          gaffa::make_ffa_cuda_execution_plan(valid_plan()),
          gaffa::CudaFfaOptions{.workspace_bytes_limit = 1}),
      std::runtime_error);
}

TEST(FfaCuda, RejectsInvalidSearchInputsBeforeCudaAllocation) {
  float fake_data = 0.0F;
  const auto plan = valid_plan();
  const auto input = valid_input(&fake_data);

  auto null_input = input;
  null_input.data = nullptr;
  EXPECT_THROW((void)gaffa::search_ffa_cuda(null_input, plan),
               std::invalid_argument);

  auto empty_series = input;
  empty_series.nseries = 0;
  EXPECT_THROW((void)gaffa::search_ffa_cuda(empty_series, plan),
               std::invalid_argument);

  auto empty_samples = input;
  empty_samples.nsamples = 0;
  EXPECT_THROW((void)gaffa::search_ffa_cuda(empty_samples, plan),
               std::invalid_argument);

  auto bad_tsamp = input;
  bad_tsamp.tsamp = 0.0;
  EXPECT_THROW((void)gaffa::search_ffa_cuda(bad_tsamp, plan),
               std::invalid_argument);

  auto device_mismatch = input;
  device_mismatch.device_id = 1;
  EXPECT_THROW((void)gaffa::search_ffa_cuda(device_mismatch, plan),
               std::invalid_argument);

  EXPECT_THROW((void)gaffa::search_ffa_cuda(
                   input, plan,
                   gaffa::FfaSearchOptions{.snr_threshold = INFINITY}),
               std::invalid_argument);
}

TEST(FfaCuda, LegalSearchPathAllocatesWorkspaceThenReportsNotImplemented) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  float fake_data = 0.0F;

  EXPECT_THROW((void)gaffa::search_ffa_cuda(valid_input(&fake_data),
                                            valid_plan()),
               std::runtime_error);
}

TEST(FfaCuda, PrepareMatchesCpuWithoutDownsample) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> input{1, 2, 3, 4, 5, 10, 20, 30, 40, 50};
  const auto task = make_task(5, 1.0, 5, 1, 1, 5);

  const auto output = prepare_on_cuda(input, 2, 5, task);
  EXPECT_EQ(output, expected_prepared(input, 2, 5, 1.0));
}

TEST(FfaCuda, PrepareMatchesCpuIntegerDownsample) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> input{1, 2, 3, 4, 5, 6,
                                 7, 8, 9, 10, 11, 12};
  const auto task = make_task(6, 2.0, 3, 1, 1, 3);

  const auto output = prepare_on_cuda(input, 2, 6, task);
  EXPECT_EQ(output, expected_prepared(input, 2, 6, 2.0));
}

TEST(FfaCuda, PrepareMatchesCpuFractionalDownsample) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> input{1, 2, 3, 4, 5, 6, 7,
                                 3, 1, 4, 1, 5, 9, 2};
  const double factor = 2.5;
  const auto task = make_task(7, factor, gaffa::downsampled_size(7, factor),
                              1, 1, 2);

  const auto output = prepare_on_cuda(input, 2, 7, task);
  const auto expected = expected_prepared(input, 2, 7, factor);
  ASSERT_EQ(output.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(output[index], expected[index]);
  }
}

TEST(FfaCuda, PrepareAcceptsExplicitStream) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  const std::vector<float> input{1, 2, 3, 4, 5, 6,
                                 7, 8, 9, 10, 11, 12};
  const auto task = make_task(6, 2.0, 3, 1, 1, 3);
  const auto output = prepare_on_cuda(
      input, 2, 6, task,
      gaffa::CudaFfaOptions{
          .stream = stream,
      });

  EXPECT_EQ(output, expected_prepared(input, 2, 6, 2.0));
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(FfaCuda, PrepareRejectsInvalidArguments) {
  float fake_data = 0.0F;
  float fake_output = 0.0F;
  const auto output = fake_cuda_span(&fake_output, 4);
  const auto input = gaffa::CudaTimeSeriesBatchView{
      .data = &fake_data,
      .nseries = 1,
      .nsamples = 4,
      .tsamp = 1.0,
      .device_id = 0,
  };
  const auto task = make_task(4, 1.0, 4, 1, 1, 4);

  auto null_input = input;
  null_input.data = nullptr;
  EXPECT_THROW(gaffa::prepare_ffa_input_cuda(null_input, task, output),
               std::invalid_argument);

  auto bad_device = input;
  bad_device.device_id = 1;
  EXPECT_THROW(gaffa::prepare_ffa_input_cuda(bad_device, task, output),
               std::invalid_argument);

  EXPECT_THROW(gaffa::prepare_ffa_input_cuda(
                   input, task, fake_cuda_span(&fake_output, 4, 1)),
               std::invalid_argument);

  auto mismatched_task = task;
  mismatched_task.input_nsamples = 5;
  EXPECT_THROW(gaffa::prepare_ffa_input_cuda(input, mismatched_task, output),
               std::invalid_argument);

  auto wrong_size_task = task;
  wrong_size_task.prepared_nsamples = 3;
  EXPECT_THROW(gaffa::prepare_ffa_input_cuda(input, wrong_size_task, output),
               std::invalid_argument);

  auto bad_factor = task;
  bad_factor.downsample_factor = 0.5;
  EXPECT_THROW(gaffa::prepare_ffa_input_cuda(input, bad_factor, output),
               std::invalid_argument);
}

TEST(FfaCuda, TransformBaseCasesMatchCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  expect_transform_matches_cpu({1, 2, 3, 4, 5, 6, 7, 8}, 2,
                               gaffa::FfaTransformShape{.rows = 1, .bins = 4});
  expect_transform_matches_cpu({1, 2, 3, 4, 5, 6, 7, 8}, 1,
                               gaffa::FfaTransformShape{.rows = 2, .bins = 4});
}

TEST(FfaCuda, TransformPowerOfTwoRowsMatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  std::vector<float> input(8 * 8);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 11) - 5);
  }

  expect_transform_matches_cpu(input, 1,
                               gaffa::FfaTransformShape{.rows = 8, .bins = 8});
}

TEST(FfaCuda, TransformOddRowsMatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  std::vector<float> input(5 * 7);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>((index * 3 + 1) % 17) - 8.0F;
  }

  expect_transform_matches_cpu(input, 1,
                               gaffa::FfaTransformShape{.rows = 5, .bins = 7});
}

TEST(FfaCuda, TransformBatchSeriesMatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const auto shape = gaffa::FfaTransformShape{.rows = 7, .bins = 6};
  const std::size_t nseries = 3;
  std::vector<float> input(nseries * shape.rows * shape.bins);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>((index * 5 + 2) % 23) - 11.0F;
  }

  expect_transform_matches_cpu(input, nseries, shape);
}

TEST(FfaCuda, TransformAcceptsExplicitStream) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  const auto shape = gaffa::FfaTransformShape{.rows = 5, .bins = 7};
  const std::size_t nseries = 2;
  std::vector<float> input(nseries * shape.rows * shape.bins);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>((index * 7 + 3) % 19) - 9.0F;
  }

  const auto output = transform_on_cuda(
      input, nseries, shape,
      gaffa::CudaFfaOptions{
          .stream = stream,
      });
  const auto expected = expected_transform(input, nseries, shape);
  ASSERT_EQ(output.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(output[index], expected[index]) << "index=" << index;
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(FfaCuda, TransformUsesPreparedPrefixAndRespectsStride) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const auto shape = gaffa::FfaTransformShape{.rows = 3, .bins = 4};
  const std::size_t nseries = 2;
  const std::size_t task_elements = shape.rows * shape.bins;
  const std::size_t prepared_nsamples = task_elements + 3;
  const std::size_t input_stride = prepared_nsamples + 5;
  const std::size_t output_stride = task_elements + 7;

  std::vector<float> input(nseries * input_stride, -999.0F);
  std::vector<float> compact_input(nseries * task_elements);
  for (std::size_t series = 0; series < nseries; ++series) {
    for (std::size_t index = 0; index < task_elements; ++index) {
      const float value =
          static_cast<float>((series + 1) * 100 + static_cast<int>(index));
      input[series * input_stride + index] = value;
      compact_input[series * task_elements + index] = value;
    }
    for (std::size_t index = task_elements; index < prepared_nsamples; ++index) {
      input[series * input_stride + index] = 7777.0F;
    }
  }

  gaffa::CudaDeviceBuffer<float> device_input(input.size());
  cudaMemcpy(device_input.data(), input.data(), input.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  gaffa::CudaDeviceBuffer<float> device_scratch(nseries * output_stride);
  gaffa::CudaDeviceBuffer<float> device_output(nseries * output_stride);

  gaffa::ffa_transform_block_cuda(
      gaffa::CudaFfaInput{
          .data = device_input.data(),
          .nseries = nseries,
          .nsamples = prepared_nsamples,
          .stride = input_stride,
          .shape = shape,
          .device_id = 0,
      },
      gaffa::CudaFfaBuffer{
          .data = device_scratch.data(),
          .nseries = nseries,
          .stride = output_stride,
          .shape = shape,
          .device_id = 0,
      },
      gaffa::CudaFfaBuffer{
          .data = device_output.data(),
          .nseries = nseries,
          .stride = output_stride,
          .shape = shape,
          .device_id = 0,
      });

  std::vector<float> output(nseries * output_stride);
  cudaMemcpy(output.data(), device_output.data(), output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  const auto expected = expected_transform(compact_input, nseries, shape);

  for (std::size_t series = 0; series < nseries; ++series) {
    for (std::size_t index = 0; index < task_elements; ++index) {
      EXPECT_FLOAT_EQ(output[series * output_stride + index],
                      expected[series * task_elements + index])
          << "series=" << series << " index=" << index;
    }
  }
}

TEST(FfaCuda, TransformRejectsInvalidArguments) {
  float fake_data = 0.0F;
  float fake_scratch = 0.0F;
  float fake_output = 0.0F;
  const auto shape = gaffa::FfaTransformShape{.rows = 2, .bins = 2};
  const auto input = gaffa::CudaFfaInput{
      .data = &fake_data,
      .nseries = 1,
      .nsamples = 4,
      .stride = 4,
      .shape = shape,
      .device_id = 0,
  };
  const auto scratch = gaffa::CudaFfaBuffer{
      .data = &fake_scratch,
      .nseries = 1,
      .stride = 4,
      .shape = shape,
      .device_id = 0,
  };
  const auto output = gaffa::CudaFfaBuffer{
      .data = &fake_output,
      .nseries = 1,
      .stride = 4,
      .shape = shape,
      .device_id = 0,
  };

  auto null_input = input;
  null_input.data = nullptr;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(null_input, scratch, output),
               std::invalid_argument);

  auto empty_series = input;
  empty_series.nseries = 0;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(empty_series, scratch, output),
               std::invalid_argument);

  auto empty_input = input;
  empty_input.nsamples = 0;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(empty_input, scratch, output),
               std::invalid_argument);

  auto short_input_stride = input;
  short_input_stride.stride = 3;
  EXPECT_THROW(
      gaffa::ffa_transform_block_cuda(short_input_stride, scratch, output),
      std::invalid_argument);

  auto empty_rows = input;
  empty_rows.shape.rows = 0;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(empty_rows, scratch, output),
               std::invalid_argument);

  auto invalid_bins = input;
  invalid_bins.shape.bins = 1;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(invalid_bins, scratch, output),
               std::invalid_argument);

  auto bad_device = input;
  bad_device.device_id = 1;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(bad_device, scratch, output),
               std::invalid_argument);

  auto bad_scratch_device = scratch;
  bad_scratch_device.device_id = 1;
  EXPECT_THROW(
      gaffa::ffa_transform_block_cuda(input, bad_scratch_device, output),
      std::invalid_argument);

  auto bad_output_device = output;
  bad_output_device.device_id = 1;
  EXPECT_THROW(
      gaffa::ffa_transform_block_cuda(input, scratch, bad_output_device),
      std::invalid_argument);

  auto null_scratch = scratch;
  null_scratch.data = nullptr;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(input, null_scratch, output),
               std::invalid_argument);

  auto null_output = output;
  null_output.data = nullptr;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(input, scratch, null_output),
               std::invalid_argument);

  auto wrong_scratch_series = scratch;
  wrong_scratch_series.nseries = 2;
  EXPECT_THROW(
      gaffa::ffa_transform_block_cuda(input, wrong_scratch_series, output),
      std::invalid_argument);

  auto wrong_output_shape = output;
  wrong_output_shape.shape.bins = 3;
  EXPECT_THROW(
      gaffa::ffa_transform_block_cuda(input, scratch, wrong_output_shape),
      std::invalid_argument);

  auto short_scratch = scratch;
  short_scratch.stride = 3;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(input, short_scratch, output),
               std::invalid_argument);

  auto short_output = output;
  short_output.stride = 3;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(input, scratch, short_output),
               std::invalid_argument);

  auto short_input = input;
  short_input.nsamples = 3;
  short_input.stride = 3;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(short_input, scratch, output),
               std::invalid_argument);

  auto alias_scratch = scratch;
  alias_scratch.data = &fake_data;
  EXPECT_THROW(gaffa::ffa_transform_block_cuda(input, alias_scratch, output),
               std::invalid_argument);

  EXPECT_THROW(gaffa::ffa_transform_block_cuda(
                   input, scratch, output,
                   gaffa::CudaFfaOptions{.synchronize_after_call = false}),
               std::invalid_argument);
}
