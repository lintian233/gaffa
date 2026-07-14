#include "gaffa/cuda_memory.h"
#include "gaffa/vector_add.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <loki/algorithms/ffa.hpp>
#include <loki/algorithms/regions.hpp>
#include <loki/common/plans.hpp>
#include <loki/common/types.hpp>
#include <loki/search/configs.hpp>
#include <loki/utils/workspace.hpp>

namespace {

constexpr std::size_t kNsamples = 1U << 18U;
constexpr double kTsampSeconds = 64.0e-6;

loki::search::PulsarSearchConfig make_frequency_search_config(
    bool use_fourier) {
  const std::vector<loki::ParamLimit> parameter_limits{{100.0, 110.0}};
  return loki::search::PulsarSearchConfig(
      kNsamples,
      kTsampSeconds,
      64,
      1.0,
      parameter_limits,
      0.2,
      1.5,
      use_fourier,
      1,
      1.0,
      2.0,
      1024,
      64,
      16);
}

std::vector<float> make_signal() {
  std::vector<float> signal(kNsamples);
  for (std::size_t sample = 0; sample < signal.size(); ++sample) {
    const double time = static_cast<double>(sample) * kTsampSeconds;
    signal[sample] = static_cast<float>(std::sin(2.0 * std::acos(-1.0) *
                                                105.0 * time));
  }
  return signal;
}

void expect_finite_nonempty_scores(const std::vector<float>& scores) {
  ASSERT_FALSE(scores.empty());
  EXPECT_TRUE(std::ranges::all_of(scores, [](float score) {
    return std::isfinite(score);
  }));
}

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

class CudaStream final {
 public:
  CudaStream() = default;

  cudaError_t create() {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }

  ~CudaStream() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  cudaStream_t get() const noexcept {
    return stream_;
  }

  cudaError_t reset() {
    const cudaError_t status = cudaStreamDestroy(stream_);
    if (status == cudaSuccess) {
      stream_ = nullptr;
    }
    return status;
  }

 private:
  cudaStream_t stream_ = nullptr;
};

class CudaDeviceGuard final {
 public:
  CudaDeviceGuard() = default;

  ~CudaDeviceGuard() {
    if (restore_device_) {
      (void)cudaSetDevice(saved_device_);
    }
  }

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

  cudaError_t set_device(int device) {
    cudaError_t status = cudaGetDevice(&saved_device_);
    if (status != cudaSuccess) {
      return status;
    }
    restore_device_ = true;
    status = cudaSetDevice(device);
    if (status != cudaSuccess) {
      restore_device_ = false;
    }
    return status;
  }

 private:
  int saved_device_ = 0;
  bool restore_device_ = false;
};

void check_cuda(cudaError_t status, const char* operation) {
  ASSERT_EQ(status, cudaSuccess) << operation << ": "
                              << cudaGetErrorString(status);
}

}  // namespace

TEST(LokiSmoke, CpuFourierFfaLinksAndRuns) {
  const std::vector<float> signal = make_signal();
  const std::vector<float> variance(signal.size(), 1.0F);
  const auto config = make_frequency_search_config(true);

  const auto [scores, plan] = loki::algorithms::compute_ffa_scores(
      signal, variance, config, true, false);

  expect_finite_nonempty_scores(scores);
  EXPECT_GT(plan.get_fold_size_time(), 0U);
}

TEST(LokiSmoke, SharesCudaRuntimeWithGaffa) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  EXPECT_EQ(gaffa::vector_add({1.0F, 2.0F}, {3.0F, 4.0F}),
            (std::vector<float>{4.0F, 6.0F}));

  const std::vector<float> signal = make_signal();
  const std::vector<float> variance(signal.size(), 1.0F);
  const auto config = make_frequency_search_config(true);
  const auto [scores, plan] = loki::algorithms::compute_ffa_scores_cuda(
      signal, variance, config, 0, true);

  expect_finite_nonempty_scores(scores);
  EXPECT_GT(plan.get_fold_size_time(), 0U);
  EXPECT_EQ(gaffa::vector_add({5.0F, 6.0F}, {7.0F, 8.0F}),
            (std::vector<float>{12.0F, 14.0F}));
}

TEST(LokiSmoke, PublicDevicePffaPrimitiveRunsOnGaffaBuffers) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  CudaDeviceGuard device_guard;
  check_cuda(device_guard.set_device(0), "cudaSetDevice");
  const std::vector<float> signal = make_signal();
  const std::vector<float> variance(signal.size(), 1.0F);
  const auto config = make_frequency_search_config(false);
  loki::regions::FFARegionPlanner<float> regions(config, true);
  ASSERT_GT(regions.get_nregions(), 0U);

  const auto& stats = regions.get_stats();
  loki::memory::FFAWorkspaceCUDA<float> workspace(
      stats.get_max_buffer_size(), stats.get_max_coord_size(),
      stats.get_max_ffa_levels(), config.get_nparams());
  const auto& region_config = regions.get_cfgs().front();
  loki::plans::FFAPlan<float> plan(region_config);
  loki::algorithms::FFACUDA<float> ffa(workspace, region_config, 0);

  gaffa::CudaDeviceBuffer<float> signal_device(signal.size());
  gaffa::CudaDeviceBuffer<float> variance_device(variance.size());
  gaffa::CudaDeviceBuffer<float> fold_device(plan.get_buffer_size());
  CudaStream stream;
  check_cuda(stream.create(), "cudaStreamCreateWithFlags");
  check_cuda(cudaMemcpyAsync(signal_device.data(), signal.data(),
                             signal_device.bytes(), cudaMemcpyHostToDevice,
                             stream.get()),
             "signal H2D copy");
  check_cuda(cudaMemcpyAsync(variance_device.data(), variance.data(),
                             variance_device.bytes(), cudaMemcpyHostToDevice,
                             stream.get()),
             "variance H2D copy");

  ffa.execute(
      cuda::std::span<const float>(signal_device.data(), signal_device.size()),
      cuda::std::span<const float>(variance_device.data(),
                                   variance_device.size()),
      cuda::std::span<float>(fold_device.data(), fold_device.size()),
      stream.get());

  std::vector<float> fold(plan.get_fold_size());
  check_cuda(cudaMemcpyAsync(fold.data(), fold_device.data(),
                             fold.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream.get()),
             "fold D2H copy");
  check_cuda(cudaStreamSynchronize(stream.get()), "P-FFA stream synchronize");

  EXPECT_TRUE(std::ranges::all_of(fold, [](float value) {
    return std::isfinite(value);
  }));
  EXPECT_TRUE(std::ranges::any_of(fold, [](float value) {
    return std::abs(value) > 1.0e-6F;
  }));
  check_cuda(stream.reset(), "cudaStreamDestroy");
}
