#include "gaffa/cuda_memory.h"
#include "gaffa/preprocessing.h"
#include "gaffa/preprocessing_cuda.h"
#include "gaffa/vector_add.hpp"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool has_cuda_device() {
  return gaffa::cuda_device_count() != 0;
}

std::vector<float> preprocess_on_cuda(
    const std::vector<float>& input,
    std::size_t nseries,
    std::size_t nsamples,
    const gaffa::PreprocessPlan& plan,
  cudaStream_t stream = nullptr) {
  gaffa::CudaDeviceBuffer<float> device_data(input.size());
  if (cudaMemcpy(device_data.data(), input.data(),
                 input.size() * sizeof(float), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    throw std::runtime_error("CUDA preprocessing test input H2D failed");
  }
  gaffa::CudaPreprocessProgram program(
      plan, {}, gaffa::CudaPreprocessExecutionOptions{
                    .series_tile_size = nseries,
                    .max_nsamples = nsamples,
                    .stream = stream,
                });
  gaffa::preprocess_time_series_batch_inplace_cuda(
      program,
      gaffa::MutableCudaTimeSeriesBatchView{
          .data = device_data.data(),
          .nseries = nseries,
          .nsamples = nsamples,
          .device_id = 0,
      });
  program.synchronize();

  std::vector<float> output(input.size());
  EXPECT_EQ(cudaMemcpy(output.data(), device_data.data(),
                       output.size() * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
  return output;
}

std::vector<float> preprocess_on_cpu(const std::vector<float>& input,
                                     std::size_t nseries,
                                     std::size_t nsamples,
                                     const gaffa::PreprocessPlan& plan) {
  std::vector<float> output;
  output.reserve(input.size());
  for (std::size_t series = 0; series < nseries; ++series) {
    const auto begin = input.begin() + static_cast<std::ptrdiff_t>(series * nsamples);
    const std::vector<float> row(begin, begin + static_cast<std::ptrdiff_t>(nsamples));
    const auto processed = gaffa::preprocess_time_series_cpu(
        gaffa::TimeSeries{.data = row, .tsamp = 0.001}, plan);
    output.insert(output.end(), processed.data.begin(), processed.data.end());
  }
  return output;
}

void expect_near_vectors(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         float tolerance = 1.0e-5F) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], tolerance) << "index=" << index;
  }
}

}  // namespace

TEST(PreprocessingCuda, NormaliseBatchMatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> input{
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
      10.0F, 13.0F, 16.0F, 19.0F, 22.0F, 25.0F,
  };
  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise}},
  };

  const auto actual = preprocess_on_cuda(input, 2, 6, plan);
  const auto expected = preprocess_on_cpu(input, 2, 6, plan);
  expect_near_vectors(actual, expected);
}

TEST(PreprocessingCuda, ExactRunningMedianMatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> input{
      100.0F, 0.0F, 1.0F, 2.0F, 3.0F,
      10.0F, 10.0F, 20.0F, 10.0F, 10.0F,
  };
  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{
          .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
          .detrend_running_median = {.window_samples = 3, .min_points = 3},
      }},
  };

  const auto actual = preprocess_on_cuda(input, 2, 5, plan);
  const auto expected = preprocess_on_cpu(input, 2, 5, plan);
  expect_near_vectors(actual, expected);
}

TEST(PreprocessingCuda, ExactRunningMedianCrossWarpWindowsMatchCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  constexpr std::size_t nsamples = 301;
  std::vector<float> input(2 * nsamples);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>((index * 37 + 19) % 97) - 48.0F;
  }

  for (const std::size_t window_samples : {65U, 127U, 255U}) {
    const gaffa::PreprocessPlan plan{
        .steps = {gaffa::PreprocessStep{
            .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
            .detrend_running_median = {
                .window_samples = window_samples,
                .min_points = window_samples,
            },
        }},
    };
    const auto actual = preprocess_on_cuda(input, 2, nsamples, plan);
    const auto expected = preprocess_on_cpu(input, 2, nsamples, plan);
    expect_near_vectors(actual, expected);
  }
}

TEST(PreprocessingCuda, FastRunningMedianAndNormaliseMatchCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> input{
      1.0F, 1.0F, 1.0F, 9.0F, 9.0F, 9.0F,
      1.0F, 1.0F, 1.0F, 9.0F, 9.0F, 9.0F,
      3.0F, 3.0F, 3.0F, 8.0F, 8.0F, 8.0F,
      3.0F, 3.0F, 3.0F, 8.0F, 8.0F, 8.0F,
  };
  const gaffa::PreprocessPlan plan{
      .steps = {
          gaffa::PreprocessStep{
              .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
              .detrend_running_median = {.window_samples = 7, .min_points = 3},
          },
          gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise},
      },
  };

  const auto actual = preprocess_on_cuda(input, 2, 12, plan);
  const auto expected = preprocess_on_cpu(input, 2, 12, plan);
  expect_near_vectors(actual, expected, 2.0e-5F);
}

TEST(PreprocessingCuda, FastRunningMedianDropsCpuEquivalentTail) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  constexpr std::size_t nsamples = 103;
  std::vector<float> input(2 * nsamples);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>((index * 29 + 7) % 71) - 35.0F;
  }
  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{
          .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
          .detrend_running_median = {.window_samples = 41, .min_points = 5},
      }},
  };

  const auto actual = preprocess_on_cuda(input, 2, nsamples, plan);
  const auto expected = preprocess_on_cpu(input, 2, nsamples, plan);
  expect_near_vectors(actual, expected, 2.0e-5F);
}

TEST(PreprocessingCuda, NormaliseConstantWithoutRejectionMatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> input{2.0F, 2.0F, 2.0F, 2.0F,
                                 -3.0F, -3.0F, -3.0F, -3.0F};
  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{
          .kind = gaffa::PreprocessStepKind::Normalise,
          .normalise = {.reject_constant = false},
      }},
  };

  const auto actual = preprocess_on_cuda(input, 2, 4, plan);
  const auto expected = preprocess_on_cpu(input, 2, 4, plan);
  expect_near_vectors(actual, expected);
}

TEST(PreprocessingCuda, MultipleDetrendStepsMatchCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> input{
      5.0F, 7.0F, 9.0F, 6.0F, 4.0F, 12.0F, 8.0F, 3.0F, 11.0F,
      20.0F, 18.0F, 15.0F, 21.0F, 17.0F, 16.0F, 22.0F, 19.0F, 23.0F,
  };
  const gaffa::PreprocessPlan plan{
      .steps = {
          gaffa::PreprocessStep{
              .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
              .detrend_running_median = {.window_samples = 5, .min_points = 5},
          },
          gaffa::PreprocessStep{
              .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
              .detrend_running_median = {.window_samples = 3, .min_points = 3},
          },
      },
  };

  const auto actual = preprocess_on_cuda(input, 2, 9, plan);
  const auto expected = preprocess_on_cpu(input, 2, 9, plan);
  expect_near_vectors(actual, expected);
}

TEST(PreprocessingCuda, ConstantAndNonFiniteInputsReportDeferredErrors) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const gaffa::PreprocessPlan reject_constant{
      .steps = {gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise}},
  };
  gaffa::CudaDeviceBuffer<float> device_data(4);
  const std::vector<float> constant{2.0F, 2.0F, 2.0F, 2.0F};
  ASSERT_EQ(cudaMemcpy(device_data.data(), constant.data(),
                       constant.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  gaffa::CudaPreprocessProgram constant_program(
      reject_constant, {}, {.series_tile_size = 1, .max_nsamples = 4});
  gaffa::preprocess_time_series_batch_inplace_cuda(
      constant_program,
      {.data = device_data.data(), .nseries = 1, .nsamples = 4, .device_id = 0});
  EXPECT_THROW(constant_program.synchronize(), std::invalid_argument);

  const std::vector<float> nonfinite{1.0F, NAN, 3.0F, 4.0F};
  ASSERT_EQ(cudaMemcpy(device_data.data(), nonfinite.data(),
                       nonfinite.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  gaffa::CudaPreprocessProgram finite_program(
      reject_constant, {}, {.series_tile_size = 1, .max_nsamples = 4});
  gaffa::preprocess_time_series_batch_inplace_cuda(
      finite_program,
      {.data = device_data.data(), .nseries = 1, .nsamples = 4, .device_id = 0});
  EXPECT_THROW(finite_program.synchronize(), std::invalid_argument);
}

TEST(PreprocessingCuda, FastDetrendWorkspaceAvoidsFullResolutionPingPong) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{
          .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
          .detrend_running_median = {.window_samples = 101, .min_points = 5},
      }},
  };
  const gaffa::CudaPreprocessProgram program(
      plan, {}, {.series_tile_size = 2, .max_nsamples = 1000});
  const auto& shape = program.workspace_shape();

  EXPECT_EQ(shape.scrunched_bytes, 2 * (1000 / 20) * sizeof(float));
  EXPECT_EQ(shape.baseline_bytes, shape.scrunched_bytes);
  EXPECT_LT(shape.scrunched_bytes + shape.baseline_bytes,
            2 * 1000 * sizeof(float));
}

TEST(PreprocessingCuda, EstimatesWorkspaceAndEnforcesLimitBeforeAllocation) {
  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{
          .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
          .detrend_running_median = {.window_samples = 101, .min_points = 5},
      }},
  };
  const gaffa::CudaPreprocessExecutionOptions options{
      .series_tile_size = 2,
      .max_nsamples = 1000,
  };
  const auto shape = gaffa::estimate_cuda_preprocess_workspace(plan, options);
  EXPECT_EQ(shape.total_bytes,
            shape.scrunched_bytes + shape.baseline_bytes +
                shape.partial_stats_bytes + shape.series_stats_bytes +
                shape.status_bytes);
  EXPECT_THROW(
      (void)gaffa::estimate_cuda_preprocess_workspace(
          plan,
          {.series_tile_size = 2,
           .max_nsamples = 1000,
           .workspace_bytes_limit = shape.total_bytes - 1}),
      std::runtime_error);
}

TEST(PreprocessingCuda, RejectsReuseBeforeSynchronize) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise}},
  };
  gaffa::CudaDeviceBuffer<float> device_data(4);
  const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};
  ASSERT_EQ(cudaMemcpy(device_data.data(), input.data(),
                       input.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  gaffa::CudaPreprocessProgram program(
      plan, {}, {.series_tile_size = 1, .max_nsamples = 4});
  const gaffa::MutableCudaTimeSeriesBatchView batch{
      .data = device_data.data(),
      .nseries = 1,
      .nsamples = 4,
      .device_id = 0,
  };
  gaffa::preprocess_time_series_batch_inplace_cuda(program, batch);
  EXPECT_THROW(gaffa::preprocess_time_series_batch_inplace_cuda(program, batch),
               std::logic_error);
  EXPECT_NO_THROW(program.synchronize());
  EXPECT_NO_THROW(gaffa::preprocess_time_series_batch_inplace_cuda(program, batch));
  EXPECT_NO_THROW(program.synchronize());
}

TEST(PreprocessingCuda, SynchronizeWithoutActiveRunIsNoOp) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise}},
  };
  gaffa::CudaPreprocessProgram program(
      plan, {}, {.series_tile_size = 1, .max_nsamples = 4});
  EXPECT_NO_THROW(program.synchronize());
}

TEST(PreprocessingCuda, RejectsStreamFromAnotherDevice) {
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count < 2) {
    GTEST_SKIP() << "multiple CUDA devices are not visible";
  }

  int original_device = 0;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const int other_device = (original_device + 1) % device_count;
  cudaStream_t foreign_stream = nullptr;
  ASSERT_EQ(cudaSetDevice(other_device), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&foreign_stream), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(original_device), cudaSuccess);

  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise}},
  };
  EXPECT_THROW(
      (void)gaffa::CudaPreprocessProgram(
          plan,
          {.device_id = original_device},
          {.series_tile_size = 1, .max_nsamples = 4, .stream = foreign_stream}),
      std::invalid_argument);

  ASSERT_EQ(cudaSetDevice(other_device), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(foreign_stream), cudaSuccess);
  EXPECT_EQ(cudaSetDevice(original_device), cudaSuccess);
}

TEST(PreprocessingCuda, AcceptsExplicitStream) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};
  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise}},
  };
  const auto actual = preprocess_on_cuda(input, 1, 4, plan, stream);
  const auto expected = preprocess_on_cpu(input, 1, 4, plan);
  expect_near_vectors(actual, expected);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
