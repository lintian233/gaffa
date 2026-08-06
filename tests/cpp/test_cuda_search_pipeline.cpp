#include "gaffa/dedispersion.h"
#include "gaffa/dedispersion_cuda.h"
#include "gaffa/dm_search_cuda.h"
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
      .observation = {.nsamples = nsamples, .tsamp_seconds = 0.001},
      .tasks = {gaffa::FfaSearchTask{
          .downsample_factor = 1.0,
          .effective_tsamp = 0.001,
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

bool by_peak_location(const gaffa::DmPeak& lhs,
                      const gaffa::DmPeak& rhs) {
  if (lhs.dm_index != rhs.dm_index) {
    return lhs.dm_index < rhs.dm_index;
  }
  if (lhs.peak.phase_bin != rhs.peak.phase_bin) {
    return lhs.peak.phase_bin < rhs.peak.phase_bin;
  }
  if (lhs.peak.boxcar_width_bins != rhs.peak.boxcar_width_bins) {
    return lhs.peak.boxcar_width_bins < rhs.peak.boxcar_width_bins;
  }
  return lhs.peak.motion.frequency_hz < rhs.peak.motion.frequency_hz;
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
  const std::vector<double> dms{0.0, 1.0};

  const auto cpu_dedispersed = gaffa::dedisperse_subband_cpu(
      input, frequency_mhz, dedispersion_plan, subband_options);
  gaffa::DmPeaks expected;
  for (std::size_t dm_index = 0; dm_index < ndm; ++dm_index) {
    std::vector<float> series(nsamples);
    for (std::size_t sample = 0; sample < nsamples; ++sample) {
      series[sample] = static_cast<float>(
          cpu_dedispersed.data[dm_index * nsamples + sample]);
    }
    const auto preprocessed = gaffa::preprocess_time_series_cpu(
        gaffa::TimeSeries{.data = std::move(series), .tsamp = 0.001},
        preprocess_plan);
    const auto cpu_peaks = gaffa::search_ffa_raw_cpu(
        preprocessed.data, ffa_plan, search_options);
    const auto periodic =
        gaffa::periodic_peaks_from_ffa(cpu_peaks.peaks, ffa_plan.observation);
    gaffa::DmPeaks dm_peaks =
        gaffa::attach_dm_peaks(periodic, dms[dm_index], dm_index + 7);
    expected.insert(expected.end(), dm_peaks.begin(), dm_peaks.end());
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
  gaffa::CudaFfaProgram ffa_program(
      ffa_plan, {.device_id = gpu_dedispersed.device_id},
      {.series_tile_size = ndm});
  auto actual = gaffa::search_dm_ffa_cuda(
      preprocess_program, ffa_program, mutable_batch,
      {.values = dms, .index_offset = 7}, search_options);
  std::sort(expected.begin(), expected.end(), by_peak_location);
  std::sort(actual.begin(), actual.end(), by_peak_location);

  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(actual[index].dm_index, expected[index].dm_index);
    EXPECT_DOUBLE_EQ(actual[index].dm, expected[index].dm);
    EXPECT_EQ(actual[index].peak.phase_bin, expected[index].peak.phase_bin);
    EXPECT_EQ(actual[index].peak.boxcar_width_bins,
              expected[index].peak.boxcar_width_bins);
    EXPECT_FLOAT_EQ(actual[index].peak.snr, expected[index].peak.snr);
    EXPECT_DOUBLE_EQ(actual[index].peak.period_seconds(),
                     expected[index].peak.period_seconds());
    EXPECT_DOUBLE_EQ(actual[index].peak.motion.frequency_hz,
                     expected[index].peak.motion.frequency_hz);
  }
}
