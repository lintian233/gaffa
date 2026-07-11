#include "gaffa/time_series_cuda.h"

#include "gaffa/cuda_memory.h"
#include "gaffa/time_series.h"
#include "gaffa/vector_add.hpp"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

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

std::vector<float> downsample_on_cuda(const std::vector<float>& input,
                                      std::size_t nseries,
                                      std::size_t nsamples,
                                      double factor,
                                      gaffa::CudaLaunchOptions options = {}) {
  gaffa::CudaDeviceBuffer<float> device_input(input.size());
  cudaMemcpy(device_input.data(), input.data(), input.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  const std::size_t output_nsamples =
      gaffa::downsampled_size(nsamples, factor);
  const std::size_t output_size = nseries * output_nsamples;
  gaffa::CudaDeviceBuffer<float> device_output(output_size);
  gaffa::downsample_weighted_sum_cuda(
      gaffa::CudaTimeSeriesBatchView{
          .data = device_input.data(),
          .nseries = nseries,
          .nsamples = nsamples,
          .tsamp = 1.0,
          .device_id = 0,
      },
      factor, device_output.as_span(0), options);

  std::vector<float> output(output_size);
  cudaMemcpy(output.data(), device_output.data(), output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  return output;
}

std::vector<float> expected_downsampled(const std::vector<float>& input,
                                        std::size_t nseries,
                                        std::size_t nsamples,
                                        double factor) {
  std::vector<float> expected;
  const std::size_t output_nsamples =
      gaffa::downsampled_size(nsamples, factor);
  expected.reserve(nseries * output_nsamples);
  for (std::size_t series = 0; series < nseries; ++series) {
    const auto row = std::span<const float>(input).subspan(
        series * nsamples, nsamples);
    const auto downsampled = gaffa::downsample_weighted_sum_cpu(row, factor);
    expected.insert(expected.end(), downsampled.begin(), downsampled.end());
  }
  return expected;
}

}  // namespace

TEST(TimeSeriesCuda, WeightedDownsampleMatchesCpuIntegerFactor) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> input{1, 2, 3, 4, 5, 6,
                                 7, 8, 9, 10, 11, 12};

  const auto output = downsample_on_cuda(input, 2, 6, 2.0);
  EXPECT_EQ(output, expected_downsampled(input, 2, 6, 2.0));
}

TEST(TimeSeriesCuda, WeightedDownsampleMatchesCpuFractionalFactor) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> input{1, 2, 3, 4, 5, 6, 7,
                                 3, 1, 4, 1, 5, 9, 2};
  const double factor = 2.5;

  const auto output = downsample_on_cuda(input, 2, 7, factor);
  const auto expected = expected_downsampled(input, 2, 7, factor);
  ASSERT_EQ(output.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(output[index], expected[index]);
  }
}

TEST(TimeSeriesCuda, WeightedDownsampleAcceptsExplicitStream) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  const std::vector<float> input{1, 2, 3, 4, 5, 6,
                                 7, 8, 9, 10, 11, 12};
  const auto output = downsample_on_cuda(
      input, 2, 6, 2.0,
      gaffa::CudaLaunchOptions{
          .stream = stream,
      });

  EXPECT_EQ(output, expected_downsampled(input, 2, 6, 2.0));
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(TimeSeriesCuda, WeightedDownsampleRejectsInvalidArguments) {
  float fake_data = 0.0F;
  float fake_output = 0.0F;
  const auto output = fake_cuda_span(&fake_output, 2);
  const auto input = gaffa::CudaTimeSeriesBatchView{
      .data = &fake_data,
      .nseries = 1,
      .nsamples = 4,
      .tsamp = 1.0,
      .device_id = 0,
  };

  auto null_input = input;
  null_input.data = nullptr;
  EXPECT_THROW(gaffa::downsample_weighted_sum_cuda(null_input, 2.0, output),
               std::invalid_argument);

  auto bad_device = input;
  bad_device.device_id = 1;
  EXPECT_THROW(gaffa::downsample_weighted_sum_cuda(bad_device, 2.0, output),
               std::invalid_argument);

  EXPECT_THROW(gaffa::downsample_weighted_sum_cuda(
                   input, 2.0, fake_cuda_span(&fake_output, 2, 1)),
               std::invalid_argument);

  EXPECT_THROW(gaffa::downsample_weighted_sum_cuda(input, 1.0, output),
               std::invalid_argument);

  EXPECT_THROW(gaffa::downsample_weighted_sum_cuda(
                   input, 2.0, fake_cuda_span(&fake_output, 1)),
               std::invalid_argument);
}
