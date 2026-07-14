#include "gaffa/vector_add.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <loki/algorithms/ffa.hpp>
#include <loki/common/types.hpp>
#include <loki/search/configs.hpp>

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
