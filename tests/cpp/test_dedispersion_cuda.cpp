#include "gaffa/dedispersion.h"
#include "gaffa/dedispersion_cuda.h"
#include "gaffa/filterbank.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/vector_add.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

template <typename T>
gaffa::HostSampleView<T> make_view(const std::vector<T>& samples,
                                   std::size_t nsamples,
                                   std::size_t nchans) {
  return gaffa::HostSampleView<T>{samples.data(),
                                  gaffa::SampleShape{nsamples, 1, nchans}};
}

bool has_cuda_device() {
  return gaffa::cuda_device_count() != 0;
}

gaffa::SingleDmDedispersionPlan single_plan(double dm) {
  return gaffa::SingleDmDedispersionPlan{
      .dm = dm,
      .ref_frequency_mhz = 2000.0,
      .tsamp = 0.003111606,
      .chan_begin = 0,
      .chan_end = 4,
  };
}

gaffa::MultiDmDedispersionPlan multi_plan(std::size_t ndm) {
  return gaffa::MultiDmDedispersionPlan{
      .dm_low = 0.0,
      .dm_step = 1.0,
      .ndm = ndm,
      .ref_frequency_mhz = 2000.0,
      .tsamp = 0.003111606,
      .chan_begin = 0,
      .chan_end = 4,
  };
}

std::filesystem::path test_data_path(const std::string& filename) {
  return std::filesystem::path(GAFFA_TEST_DATA_DIR) / filename;
}

}  // namespace

TEST(DedispersionCuda, SingleDmUint8MatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint8_t> samples{1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const auto view = make_view(samples, 2, 4);

  const auto cpu = gaffa::dedisperse_single_dm_cpu(view, frequency,
                                                  single_plan(1.0));
  const auto cuda = gaffa::dedisperse_single_dm_cuda(view, frequency,
                                                    single_plan(1.0));

  EXPECT_EQ(cuda.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(cuda.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(cuda.data, cpu.data);
}

TEST(DedispersionCuda, SingleDmUint16MatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint16_t> samples{100, 200, 300, 400,
                                           500, 600, 700, 800};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const auto view = make_view(samples, 2, 4);

  const auto cpu = gaffa::dedisperse_single_dm_cpu(view, frequency,
                                                  single_plan(1.0));
  const auto cuda = gaffa::dedisperse_single_dm_cuda(view, frequency,
                                                    single_plan(1.0));

  EXPECT_EQ(cuda.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(cuda.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(cuda.data, cpu.data);
}

TEST(DedispersionCuda, SingleDmFloatMatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> samples{1.0F, 2.0F, 3.0F, 4.0F,
                                   5.0F, 6.0F, 7.0F, 8.0F};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const auto view = make_view(samples, 2, 4);

  const auto cpu = gaffa::dedisperse_single_dm_cpu(view, frequency,
                                                  single_plan(1.0));
  const auto cuda = gaffa::dedisperse_single_dm_cuda(view, frequency,
                                                    single_plan(1.0));

  EXPECT_EQ(cuda.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(cuda.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(cuda.data, cpu.data);
}

TEST(DedispersionCuda, SpectrumUint8MatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint8_t> samples{1,  2,  3,  4,  5,  6,  7,  8,
                                          9,  10, 11, 12, 13, 14, 15, 16};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(1.0);
  plan.chan_begin = 1;
  plan.chan_end = 4;
  const auto view = make_view(samples, 4, 4);

  const auto cpu = gaffa::dedisperse_spectrum_cpu(view, frequency, plan);
  const auto cuda = gaffa::dedisperse_spectrum_cuda(view, frequency, plan);

  EXPECT_EQ(cuda.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(cuda.shape.nifs, cpu.shape.nifs);
  EXPECT_EQ(cuda.shape.nchans, cpu.shape.nchans);
  EXPECT_EQ(cuda.dm, cpu.dm);
  EXPECT_EQ(cuda.tsamp, cpu.tsamp);
  EXPECT_EQ(cuda.chan_begin, cpu.chan_begin);
  EXPECT_EQ(cuda.chan_end, cpu.chan_end);
  EXPECT_EQ(cuda.data, cpu.data);
}

TEST(DedispersionCuda, SpectrumUint16AndFloatMatchCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint16_t> samples_u16{100, 200, 300, 400,
                                               500, 600, 700, 800,
                                               900, 1000, 1100, 1200,
                                               1300, 1400, 1500, 1600};
  const std::vector<float> samples_f32{1.0F,  2.0F,  3.0F,  4.0F,
                                       5.0F,  6.0F,  7.0F,  8.0F,
                                       9.0F,  10.0F, 11.0F, 12.0F,
                                       13.0F, 14.0F, 15.0F, 16.0F};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const auto view_u16 = make_view(samples_u16, 4, 4);
  const auto view_f32 = make_view(samples_f32, 4, 4);

  const auto cpu_u16 =
      gaffa::dedisperse_spectrum_cpu(view_u16, frequency, single_plan(1.0));
  const auto cuda_u16 =
      gaffa::dedisperse_spectrum_cuda(view_u16, frequency, single_plan(1.0));
  const auto cpu_f32 =
      gaffa::dedisperse_spectrum_cpu(view_f32, frequency, single_plan(1.0));
  const auto cuda_f32 =
      gaffa::dedisperse_spectrum_cuda(view_f32, frequency, single_plan(1.0));

  EXPECT_EQ(cuda_u16.shape.nsamples, cpu_u16.shape.nsamples);
  EXPECT_EQ(cuda_u16.shape.nchans, cpu_u16.shape.nchans);
  EXPECT_EQ(cuda_u16.data, cpu_u16.data);
  EXPECT_EQ(cuda_f32.shape.nsamples, cpu_f32.shape.nsamples);
  EXPECT_EQ(cuda_f32.shape.nchans, cpu_f32.shape.nchans);
  EXPECT_EQ(cuda_f32.data, cpu_f32.data);
}

TEST(DedispersionCuda, SpectrumDeviceResultCopiesToHost) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint8_t> samples{1,  2,  3,  4,  5,  6,  7,  8,
                                          9,  10, 11, 12, 13, 14, 15, 16};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const auto view = make_view(samples, 4, 4);

  const auto cpu =
      gaffa::dedisperse_spectrum_cpu(view, frequency, single_plan(1.0));
  const auto device =
      gaffa::dedisperse_spectrum_cuda_device(view, frequency, single_plan(1.0));
  const auto device_view = device.view();
  const auto device_span = device_view.as_span();
  const auto host = gaffa::copy_to_host(device);

  EXPECT_EQ(device.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(device.shape.nifs, cpu.shape.nifs);
  EXPECT_EQ(device.shape.nchans, cpu.shape.nchans);
  EXPECT_EQ(device.size(), cpu.data.size());
  EXPECT_EQ(device_view.data, device.data.data());
  EXPECT_EQ(device_view.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(device_view.shape.nifs, cpu.shape.nifs);
  EXPECT_EQ(device_view.shape.nchans, cpu.shape.nchans);
  EXPECT_EQ(device_view.size(), cpu.data.size());
  EXPECT_EQ(device_view.device_id, 0);
  EXPECT_EQ(device_span.data, device.data.data());
  EXPECT_EQ(device_span.size(), cpu.data.size());
  EXPECT_EQ(device_span.bytes(), cpu.data.size() * sizeof(std::uint8_t));
  EXPECT_EQ(device_span.device_id, 0);
  EXPECT_EQ(host.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(host.shape.nifs, cpu.shape.nifs);
  EXPECT_EQ(host.shape.nchans, cpu.shape.nchans);
  EXPECT_EQ(host.dm, cpu.dm);
  EXPECT_EQ(host.tsamp, cpu.tsamp);
  EXPECT_EQ(host.chan_begin, cpu.chan_begin);
  EXPECT_EQ(host.chan_end, cpu.chan_end);
  EXPECT_EQ(host.data, cpu.data);
}

TEST(DedispersionCuda, SpectrumChannelSumMatchesSingleDmCuda) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint8_t> samples{1,  2,  3,  4,  5,  6,  7,  8,
                                          9,  10, 11, 12, 13, 14, 15, 16};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const auto view = make_view(samples, 4, 4);

  const auto spectrum =
      gaffa::dedisperse_spectrum_cuda(view, frequency, single_plan(1.0));
  const auto single =
      gaffa::dedisperse_single_dm_cuda(view, frequency, single_plan(1.0));

  ASSERT_EQ(single.shape.nsamples, spectrum.shape.nsamples);
  for (std::size_t time = 0; time < spectrum.shape.nsamples; ++time) {
    std::uint32_t sum = 0;
    for (std::size_t channel = 0; channel < spectrum.shape.nchans; ++channel) {
      sum += spectrum.data[time * spectrum.shape.nchans + channel];
    }
    EXPECT_EQ(single.data[time], sum);
  }
}

TEST(DedispersionCuda, MultiDmUint8MatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint8_t> samples{1,  2,  3,  4,  5,  6,  7,  8,
                                          9,  10, 11, 12, 13, 14, 15, 16};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const auto view = make_view(samples, 4, 4);

  const auto cpu = gaffa::dedisperse_multi_dm_cpu(view, frequency,
                                                 multi_plan(3));
  const auto cuda = gaffa::dedisperse_multi_dm_cuda(view, frequency,
                                                   multi_plan(3));

  EXPECT_EQ(cuda.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(cuda.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(cuda.data, cpu.data);
}

TEST(DedispersionCuda, DeviceResultCopiesToHost) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint8_t> samples{1,  2,  3,  4,  5,  6,  7,  8,
                                          9,  10, 11, 12, 13, 14, 15, 16};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const auto view = make_view(samples, 4, 4);

  const auto cpu = gaffa::dedisperse_multi_dm_cpu(view, frequency,
                                                 multi_plan(3));
  const auto device = gaffa::dedisperse_multi_dm_cuda_device(
      view, frequency, multi_plan(3));
  const auto device_view = device.view();
  const auto device_span = device_view.as_span();
  const auto host = gaffa::copy_to_host(device);

  EXPECT_EQ(device.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(device.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(device.size(), cpu.data.size());
  EXPECT_EQ(device_view.data, device.data.data());
  EXPECT_EQ(device_view.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(device_view.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(device_view.size(), cpu.data.size());
  EXPECT_EQ(device_view.device_id, 0);
  EXPECT_EQ(device_span.data, device.data.data());
  EXPECT_EQ(device_span.size(), cpu.data.size());
  EXPECT_EQ(device_span.bytes(), cpu.data.size() * sizeof(std::uint32_t));
  EXPECT_EQ(device_span.device_id, 0);
  EXPECT_EQ(host.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(host.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(host.data, cpu.data);
}

TEST(DedispersionCuda, MultiDmUint16MatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint16_t> samples{100, 200, 300, 400,
                                           500, 600, 700, 800,
                                           900, 1000, 1100, 1200,
                                           1300, 1400, 1500, 1600};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const auto view = make_view(samples, 4, 4);

  const auto cpu = gaffa::dedisperse_multi_dm_cpu(view, frequency,
                                                 multi_plan(3));
  const auto cuda = gaffa::dedisperse_multi_dm_cuda(view, frequency,
                                                   multi_plan(3));

  EXPECT_EQ(cuda.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(cuda.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(cuda.data, cpu.data);
}

TEST(DedispersionCuda, MultiDmFloatMatchesCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<float> samples{1.0F, 2.0F, 3.0F, 4.0F,
                                   5.0F, 6.0F, 7.0F, 8.0F,
                                   9.0F, 10.0F, 11.0F, 12.0F,
                                   13.0F, 14.0F, 15.0F, 16.0F};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const auto view = make_view(samples, 4, 4);

  const auto cpu = gaffa::dedisperse_multi_dm_cpu(view, frequency,
                                                 multi_plan(3));
  const auto cuda = gaffa::dedisperse_multi_dm_cuda(view, frequency,
                                                   multi_plan(3));

  EXPECT_EQ(cuda.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(cuda.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(cuda.data, cpu.data);
}

TEST(DedispersionCuda, SubbandDegenerateMatchesMultiDmCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint8_t> samples{1,  2,  3,  4,  5,  6,  7,  8,
                                          9,  10, 11, 12, 13, 14, 15, 16};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  const gaffa::SubbandDedispersionOptions subband_options{
      .subband_channels = 1,
      .ndm_per_nominal = 1,
  };
  const auto view = make_view(samples, 4, 4);

  const auto cpu = gaffa::dedisperse_multi_dm_cpu(view, frequency,
                                                 multi_plan(3));
  const auto cuda = gaffa::dedisperse_subband_cuda(
      view, frequency, multi_plan(3), subband_options);

  EXPECT_EQ(cuda.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(cuda.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(cuda.data, cpu.data);
}

TEST(DedispersionCuda, SubbandNormalMatchesCpuSubband) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::vector<std::uint8_t> samples{1, 2, 3, 4, 5, 6, 7, 8,
                                          9, 10, 11, 12, 13, 14, 15, 16};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  gaffa::MultiDmDedispersionPlan plan = multi_plan(4);
  plan.ndm = 4;
  const gaffa::SubbandDedispersionOptions subband_options{
      .subband_channels = 2,
      .ndm_per_nominal = 2,
  };
  const gaffa::CudaDedispersionOptions cuda_options{
      .device_id = 0,
      .threads_per_block = 128,
      .time_tile_samples = 2,
  };
  const auto view = make_view(samples, 4, 4);

  const auto cpu = gaffa::dedisperse_subband_cpu(view, frequency, plan,
                                                subband_options);
  const auto cuda = gaffa::dedisperse_subband_cuda(
      view, frequency, plan, subband_options, cuda_options);

  EXPECT_EQ(cuda.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(cuda.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(cuda.data, cpu.data);
}

TEST(DedispersionCuda, BaseTestFixtureMatchesCpuModes) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  const std::filesystem::path path = test_data_path("basetest.fil");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "basetest.fil is not available";
  }

  const gaffa::FilterbankData filterbank = gaffa::read_filterbank(path);
  ASSERT_EQ(filterbank.header.nifs, 1);
  ASSERT_EQ(filterbank.header.nbits, 8);

  const auto samples = gaffa::sample_view<std::uint8_t>(filterbank);
  const gaffa::MultiDmDedispersionPlan plan{
      .dm_low = 0.0,
      .dm_step = 1.0,
      .ndm = 50,
      .ref_frequency_mhz = filterbank.header.frequency_table.back(),
      .tsamp = filterbank.header.tsamp,
      .chan_begin = 0,
      .chan_end = static_cast<std::size_t>(filterbank.header.nchans),
  };
  const gaffa::SubbandDedispersionOptions degenerate_subband{
      .subband_channels = 1,
      .ndm_per_nominal = 1,
  };

  const auto cpu = gaffa::dedisperse_multi_dm_cpu(
      samples, filterbank.header.frequency_table, plan);
  const auto cuda = gaffa::dedisperse_subband_cuda(
      samples, filterbank.header.frequency_table, plan, degenerate_subband);

  EXPECT_EQ(cuda.shape.ndm, cpu.shape.ndm);
  EXPECT_EQ(cuda.shape.nsamples, cpu.shape.nsamples);
  EXPECT_EQ(cuda.data, cpu.data);
}
