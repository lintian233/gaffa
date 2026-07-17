#include "gaffa/loki_pffa.h"
#include "gaffa/cuda_memory.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kNsamples = 1U << 18U;
constexpr double kTsampSeconds = 64.0e-6;
constexpr double kReferenceTimeSeconds =
    0.5 * static_cast<double>(kNsamples) * kTsampSeconds;

gaffa::LokiTaylorSearchSpace frequency_only_space() {
  return gaffa::LokiTaylorSearchSpace{
      .frequency_hz = {.minimum = 10.0, .maximum = 20.0},
  };
}

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::vector<float> make_sinusoid() {
  std::vector<float> signal(kNsamples);
  for (std::size_t sample = 0; sample < signal.size(); ++sample) {
    const double time = static_cast<double>(sample) * kTsampSeconds;
    signal[sample] = static_cast<float>(
        std::sin(2.0 * std::acos(-1.0) * 105.0 * time));
  }
  return signal;
}

gaffa::LokiPffaPlanOptions test_plan_options() {
  auto options = gaffa::LokiPffaPlanOptions{};
  options.phase_bins_min = 64;
  options.phase_bins_max = 64;
  options.snr_threshold = 3.0F;
  return options;
}

std::vector<gaffa::PeriodicPeak> search_on_cuda(
    const gaffa::LokiTaylorSearchSpace& search_space) {
  if (!has_cuda_device()) {
    throw std::runtime_error("CUDA device is not visible");
  }
  const cudaError_t set_device_status = cudaSetDevice(0);
  if (set_device_status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaSetDevice: ") +
                             cudaGetErrorString(set_device_status));
  }

  const auto plan = gaffa::make_loki_pffa_plan(
      kNsamples, kTsampSeconds, search_space, test_plan_options());
  gaffa::LokiPffaProgram program(plan);
  const std::vector<float> signal = make_sinusoid();
  gaffa::CudaDeviceBuffer<float> signal_device(signal.size());
  const cudaError_t copy_status =
      cudaMemcpy(signal_device.data(), signal.data(), signal_device.bytes(),
                 cudaMemcpyHostToDevice);
  if (copy_status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpy signal: ") +
                             cudaGetErrorString(copy_status));
  }
  const auto& signal_device_const = signal_device;
  return program.search(signal_device_const.as_span(0));
}

void expect_common_peak_fields(std::span<const gaffa::PeriodicPeak> peaks,
                               gaffa::MotionOrder expected_order) {
  ASSERT_FALSE(peaks.empty());
  for (const auto& peak : peaks) {
    EXPECT_EQ(peak.motion.order, expected_order);
    EXPECT_DOUBLE_EQ(peak.motion.reference_time_seconds, kReferenceTimeSeconds);
    EXPECT_TRUE(std::isfinite(peak.snr));
    EXPECT_GT(peak.motion.frequency_hz, 0.0);
    EXPECT_FALSE(peak.phase_bin.has_value());
    EXPECT_NE(peak.phase_bins, 0U);
    EXPECT_NE(peak.boxcar_width_bins, 0U);
    EXPECT_GT(peak.duty_cycle, 0.0);
  }
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
  space.jerk_m_per_s3 =
      gaffa::ValueRange{.minimum = -1.0, .maximum = 1.0};
  EXPECT_THROW((void)gaffa::make_loki_pffa_plan(1U << 18U, 64.0e-6, space),
               std::invalid_argument);
}

TEST(LokiPffaPlan, RejectsZeroWidthFrequencyRange) {
  auto space = frequency_only_space();
  space.frequency_hz = {.minimum = 10.0, .maximum = 10.0};
  EXPECT_THROW((void)gaffa::make_loki_pffa_plan(1U << 18U, 64.0e-6, space),
               std::invalid_argument);
}

TEST(LokiPffaPlan, RejectsZeroWidthMotionRanges) {
  auto acceleration_space = frequency_only_space();
  acceleration_space.acceleration_m_per_s2 =
      gaffa::ValueRange{.minimum = 1.25, .maximum = 1.25};
  EXPECT_THROW((void)gaffa::make_loki_pffa_plan(1U << 18U, 64.0e-6,
                                                 acceleration_space),
               std::invalid_argument);

  auto jerk_space = frequency_only_space();
  jerk_space.acceleration_m_per_s2 =
      gaffa::ValueRange{.minimum = 1.0, .maximum = 1.5};
  jerk_space.jerk_m_per_s3 =
      gaffa::ValueRange{.minimum = -0.5, .maximum = -0.5};
  EXPECT_THROW((void)gaffa::make_loki_pffa_plan(1U << 18U, 64.0e-6,
                                                 jerk_space),
               std::invalid_argument);
}

TEST(LokiPffaPlan, RejectsUnsupportedSnapSearch) {
  auto space = frequency_only_space();
  space.acceleration_m_per_s2 =
      gaffa::ValueRange{.minimum = -1.0, .maximum = 1.0};
  space.jerk_m_per_s3 =
      gaffa::ValueRange{.minimum = -1.0, .maximum = 1.0};
  space.snap_m_per_s4 =
      gaffa::ValueRange{.minimum = -1.0, .maximum = 1.0};
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
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }
  const auto peaks = search_on_cuda({
      .frequency_hz = {.minimum = 100.0, .maximum = 110.0},
  });

  expect_common_peak_fields(peaks, gaffa::MotionOrder::Frequency);
  for (const auto& peak : peaks) {
    EXPECT_GE(peak.motion.frequency_hz, 100.0);
    EXPECT_LE(peak.motion.frequency_hz, 110.0);
    EXPECT_DOUBLE_EQ(peak.motion.acceleration_m_per_s2, 0.0);
    EXPECT_DOUBLE_EQ(peak.motion.jerk_m_per_s3, 0.0);
    EXPECT_DOUBLE_EQ(peak.motion.snap_m_per_s4, 0.0);
  }
}

TEST(LokiPffaProgram, MapsAccelerationCellToPeriodicMotion) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const auto peaks = search_on_cuda({
      .frequency_hz = {.minimum = 100.0, .maximum = 110.0},
      .acceleration_m_per_s2 =
          std::optional<gaffa::ValueRange>{gaffa::ValueRange{
              .minimum = 1.0,
              .maximum = 1.5,
          }},
  });

  expect_common_peak_fields(peaks, gaffa::MotionOrder::Acceleration);
  for (const auto& peak : peaks) {
    EXPECT_GE(peak.motion.frequency_hz, 100.0);
    EXPECT_LE(peak.motion.frequency_hz, 110.0);
    EXPECT_GE(peak.motion.acceleration_m_per_s2, 1.0);
    EXPECT_LE(peak.motion.acceleration_m_per_s2, 1.5);
    EXPECT_DOUBLE_EQ(peak.motion.jerk_m_per_s3, 0.0);
    EXPECT_DOUBLE_EQ(peak.motion.snap_m_per_s4, 0.0);
  }
}

TEST(LokiPffaProgram, MapsJerkCellToPeriodicMotion) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const auto peaks = search_on_cuda({
      .frequency_hz = {.minimum = 100.0, .maximum = 110.0},
      .acceleration_m_per_s2 =
          std::optional<gaffa::ValueRange>{gaffa::ValueRange{
              .minimum = 1.0,
              .maximum = 1.5,
          }},
      .jerk_m_per_s3 = std::optional<gaffa::ValueRange>{gaffa::ValueRange{
          .minimum = -0.5,
          .maximum = -0.25,
      }},
  });

  expect_common_peak_fields(peaks, gaffa::MotionOrder::Jerk);
  for (const auto& peak : peaks) {
    EXPECT_GE(peak.motion.frequency_hz, 100.0);
    EXPECT_LE(peak.motion.frequency_hz, 110.0);
    EXPECT_GE(peak.motion.acceleration_m_per_s2, 1.0);
    EXPECT_LE(peak.motion.acceleration_m_per_s2, 1.5);
    EXPECT_GE(peak.motion.jerk_m_per_s3, -0.5);
    EXPECT_LE(peak.motion.jerk_m_per_s3, -0.25);
    EXPECT_DOUBLE_EQ(peak.motion.snap_m_per_s4, 0.0);
  }
}

}  // namespace
