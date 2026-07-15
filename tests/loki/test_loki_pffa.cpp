#include "gaffa/loki_pffa.h"
#include "gaffa/cuda_memory.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

gaffa::LokiTaylorSearchSpace frequency_only_space() {
  return gaffa::LokiTaylorSearchSpace{
      .frequency_hz = {.minimum = 10.0, .maximum = 20.0},
  };
}

TEST(LokiPffaPlan, StoresValidatedLogicalSearchDefinition) {
  const auto plan = gaffa::make_loki_pffa_plan(
      1U << 18U, 64.0e-6, frequency_only_space());

  EXPECT_EQ(plan.input_nsamples(), 1U << 18U);
  EXPECT_DOUBLE_EQ(plan.tsamp_seconds(), 64.0e-6);
  EXPECT_DOUBLE_EQ(plan.search_space().frequency_hz.minimum, 10.0);
  EXPECT_EQ(plan.options().phase_bins_min, 256U);
}

TEST(LokiPffaPlan, RejectsNonPowerOfTwoInput) {
  EXPECT_THROW((void)gaffa::make_loki_pffa_plan(
                   1000, 64.0e-6, frequency_only_space()),
               std::invalid_argument);
}

TEST(LokiPffaPlan, RejectsDiscontinuousTaylorModel) {
  auto space = frequency_only_space();
  space.jerk = gaffa::ClosedRange{.minimum = -1.0, .maximum = 1.0};
  EXPECT_THROW((void)gaffa::make_loki_pffa_plan(1U << 18U, 64.0e-6, space),
               std::invalid_argument);
}

TEST(LokiPffaPlan, RejectsZeroWidthFrequencyRange) {
  auto space = frequency_only_space();
  space.frequency_hz = {.minimum = 10.0, .maximum = 10.0};
  EXPECT_THROW((void)gaffa::make_loki_pffa_plan(1U << 18U, 64.0e-6, space),
               std::invalid_argument);
}

TEST(LokiPffaPlan, RejectsUnsupportedSnapSearch) {
  auto space = frequency_only_space();
  space.acceleration = gaffa::ClosedRange{.minimum = -1.0, .maximum = 1.0};
  space.jerk = gaffa::ClosedRange{.minimum = -1.0, .maximum = 1.0};
  space.snap = gaffa::ClosedRange{.minimum = -1.0, .maximum = 1.0};
  EXPECT_THROW((void)gaffa::make_loki_pffa_plan(1U << 18U, 64.0e-6, space),
               std::invalid_argument);
}

TEST(LokiPffaPlan, RejectsInvalidDetectionOptions) {
  auto options = gaffa::LokiPffaPlanOptions{};
  options.snr_threshold = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW((void)gaffa::make_loki_pffa_plan(
                   1U << 18U, 64.0e-6, frequency_only_space(), options),
               std::invalid_argument);
}

TEST(LokiPffaProgram, SearchesOneNormalisedDeviceSeries) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is not visible";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  constexpr std::size_t nsamples = 1U << 18U;
  constexpr double tsamp_seconds = 64.0e-6;
  std::vector<float> signal(nsamples);
  for (std::size_t sample = 0; sample < signal.size(); ++sample) {
    const double time = static_cast<double>(sample) * tsamp_seconds;
    signal[sample] = static_cast<float>(
        std::sin(2.0 * std::acos(-1.0) * 105.0 * time));
  }

  auto plan_options = gaffa::LokiPffaPlanOptions{};
  plan_options.phase_bins_min = 64;
  plan_options.phase_bins_max = 64;
  plan_options.snr_threshold = 3.0F;
  const auto plan = gaffa::make_loki_pffa_plan(
      nsamples, tsamp_seconds,
      gaffa::LokiTaylorSearchSpace{
          .frequency_hz = {.minimum = 100.0, .maximum = 110.0},
      },
      plan_options);
  gaffa::LokiPffaProgram program(plan);
  gaffa::CudaDeviceBuffer<float> signal_device(signal.size());
  ASSERT_EQ(cudaMemcpy(signal_device.data(), signal.data(), signal_device.bytes(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  const auto& signal_device_const = signal_device;
  const auto peaks = program.search(signal_device_const.as_span(0));
  ASSERT_FALSE(peaks.empty());
  EXPECT_TRUE(std::ranges::all_of(peaks, [](const auto& peak) {
    return std::isfinite(peak.snr) && peak.frequency_hz > 0.0 &&
           peak.phase_bins != 0 && peak.boxcar_width != 0 &&
           peak.duty_cycle > 0.0F;
  }));
}

}  // namespace
