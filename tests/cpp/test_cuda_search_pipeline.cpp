#include "gaffa/dedispersion.h"
#include "gaffa/dedispersion_cuda.h"
#include "gaffa/ffa_cuda.h"
#include "gaffa/ffa_search.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/preprocessing.h"
#include "gaffa/preprocessing_cuda.h"
#include "gaffa/time_series_cuda.h"
#include "gaffa/vector_add.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

bool has_cuda_device() {
  return gaffa::cuda_device_count() != 0;
}

gaffa::FfaSearchPlan pipeline_ffa_plan(std::size_t nsamples) {
  constexpr std::size_t bins = 8;
  return gaffa::FfaSearchPlan{
      .tasks = {gaffa::FfaSearchTask{
          .downsample_factor = 1.0,
          .effective_tsamp = 0.001,
          .input_nsamples = nsamples,
          .prepared_nsamples = nsamples,
          .bins = bins,
          .rows = nsamples / bins,
          .rows_eval = nsamples / bins,
          .period_begin = 0.008,
          .period_end = 0.009,
      }},
      .width_trials = {1, 2},
  };
}

bool by_peak_location(const gaffa::FfaBatchPeak& lhs,
                      const gaffa::FfaBatchPeak& rhs) {
  if (lhs.series_index != rhs.series_index) {
    return lhs.series_index < rhs.series_index;
  }
  if (lhs.peak.shift != rhs.peak.shift) {
    return lhs.peak.shift < rhs.peak.shift;
  }
  if (lhs.peak.width_index != rhs.peak.width_index) {
    return lhs.peak.width_index < rhs.peak.width_index;
  }
  return lhs.peak.phase < rhs.peak.phase;
}

}  // namespace

TEST(CudaSearchPipeline, DeviceDedispersionPreprocessAndFfaMatchCpu) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  constexpr std::size_t nsamples = 64;
  constexpr std::size_t nchans = 4;
  constexpr std::size_t ndm = 2;
  std::vector<std::uint8_t> samples(nsamples * nchans);
  for (std::size_t index = 0; index < samples.size(); ++index) {
    samples[index] = static_cast<std::uint8_t>((index * 11 + 7) % 251);
  }
  const auto input = gaffa::make_host_sample_view<std::uint8_t>(
      std::span<const std::uint8_t>(samples),
      gaffa::SampleShape{nsamples, 1, nchans});
  const std::vector<double> frequency_mhz(nchans, 1400.0);
  const gaffa::MultiDmDedispersionPlan dedispersion_plan{
      .dm_low = 0.0,
      .dm_step = 1.0,
      .ndm = ndm,
      .ref_frequency_mhz = 1400.0,
      .tsamp = 0.001,
      .chan_begin = 0,
      .chan_end = nchans,
  };
  const gaffa::SubbandDedispersionOptions subband_options{
      .subband_channels = 2,
      .ndm_per_nominal = 2,
  };
  const gaffa::PreprocessPlan preprocess_plan{
      .steps = {gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise}},
  };
  const auto ffa_plan = pipeline_ffa_plan(nsamples);
  const gaffa::FfaSearchOptions search_options{.snr_threshold = -1000000.0F};

  const auto cpu_dedispersed = gaffa::dedisperse_subband_cpu(
      input, frequency_mhz, dedispersion_plan, subband_options);
  std::vector<gaffa::FfaBatchPeak> expected;
  for (std::size_t dm_index = 0; dm_index < ndm; ++dm_index) {
    std::vector<float> series(nsamples);
    for (std::size_t sample = 0; sample < nsamples; ++sample) {
      series[sample] = static_cast<float>(
          cpu_dedispersed.data[dm_index * nsamples + sample]);
    }
    const auto preprocessed = gaffa::preprocess_time_series_cpu(
        gaffa::TimeSeries{.data = std::move(series), .tsamp = 0.001},
        preprocess_plan);
    const auto cpu_peaks = gaffa::search_ffa_cpu(
        preprocessed.data, ffa_plan, search_options);
    for (const auto& peak : cpu_peaks.peaks) {
      expected.push_back({.series_index = dm_index, .peak = peak});
    }
  }

  auto gpu_dedispersed = gaffa::dedisperse_subband_cuda_device(
      input, frequency_mhz, dedispersion_plan, subband_options);
  ASSERT_EQ(gpu_dedispersed.shape.ndm, ndm);
  ASSERT_EQ(gpu_dedispersed.shape.nsamples, nsamples);
  gaffa::CudaDeviceBuffer<float> device_series(gpu_dedispersed.size());
  gaffa::convert_time_series_batch_to_float_cuda(
      {.data = gpu_dedispersed.data.data(),
       .count = gpu_dedispersed.size(),
       .device_id = gpu_dedispersed.device_id},
      ndm, nsamples, device_series.as_span(gpu_dedispersed.device_id));

  gaffa::CudaPreprocessProgram preprocess_program(
      preprocess_plan, {.device_id = gpu_dedispersed.device_id},
      {.series_tile_size = ndm, .max_nsamples = nsamples});
  const gaffa::MutableCudaTimeSeriesBatchView mutable_batch{
      .data = device_series.data(),
      .nseries = ndm,
      .nsamples = nsamples,
      .device_id = gpu_dedispersed.device_id,
  };
  gaffa::preprocess_time_series_batch_inplace_cuda(preprocess_program,
                                                    mutable_batch);
  preprocess_program.synchronize();

  gaffa::CudaFfaProgram ffa_program(
      ffa_plan, {.device_id = gpu_dedispersed.device_id},
      {.series_tile_size = ndm});
  auto actual = gaffa::run_ffa_batch_cuda(
      ffa_program, mutable_batch.as_const(), search_options).peaks;
  std::sort(expected.begin(), expected.end(), by_peak_location);
  std::sort(actual.begin(), actual.end(), by_peak_location);

  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(actual[index].series_index, expected[index].series_index);
    EXPECT_EQ(actual[index].peak.shift, expected[index].peak.shift);
    EXPECT_EQ(actual[index].peak.phase, expected[index].peak.phase);
    EXPECT_EQ(actual[index].peak.width_index, expected[index].peak.width_index);
    EXPECT_FLOAT_EQ(actual[index].peak.snr, expected[index].peak.snr);
    EXPECT_DOUBLE_EQ(actual[index].peak.period, expected[index].peak.period);
    EXPECT_DOUBLE_EQ(actual[index].peak.frequency, expected[index].peak.frequency);
  }
}
