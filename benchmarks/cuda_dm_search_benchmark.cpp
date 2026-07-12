#include "gaffa/candidate.h"
#include "gaffa/dedispersion_cuda.h"
#include "gaffa/ffa_cuda.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/filterbank.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/harmonic.h"
#include "gaffa/preprocessing.h"
#include "gaffa/preprocessing_cuda.h"
#include "gaffa/time_series_cuda.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

struct Args {
  std::filesystem::path path;
  std::size_t ndm = 32;
  double dm_low = 1100.2;
  double dm_step = 1.0;
  double period_min = 0.1;
  double period_max = 1.0;
  float snr_threshold = 6.0F;
  std::size_t bins_min = 200;
  std::size_t bins_max = 256;
  double running_median_seconds = 5.0;
  std::size_t subband_channels = 32;
  std::size_t ndm_per_nominal = 32;
};

struct Timings {
  double dedisperse = 0.0;
  double convert = 0.0;
  double preprocess = 0.0;
  double ffa = 0.0;
  double candidate = 0.0;
  double harmonic = 0.0;
};

template <typename Function>
double time_once(Function&& function) {
  const auto begin = std::chrono::steady_clock::now();
  function();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
      .count();
}

std::size_t parse_size(const char* value) { return std::stoull(value); }
double parse_double(const char* value) { return std::stod(value); }
float parse_float(const char* value) { return std::stof(value); }

Args parse_args(int argc, char** argv) {
  if (argc < 2 || argc > 13) {
    throw std::invalid_argument(
        "usage: cuda_dm_search_benchmark <file> [ndm dm_low dm_step "
        "period_min period_max snr bins_min bins_max median_seconds "
        "subband_channels ndm_per_nominal]");
  }
  Args args{.path = argv[1]};
  if (argc > 2) args.ndm = parse_size(argv[2]);
  if (argc > 3) args.dm_low = parse_double(argv[3]);
  if (argc > 4) args.dm_step = parse_double(argv[4]);
  if (argc > 5) args.period_min = parse_double(argv[5]);
  if (argc > 6) args.period_max = parse_double(argv[6]);
  if (argc > 7) args.snr_threshold = parse_float(argv[7]);
  if (argc > 8) args.bins_min = parse_size(argv[8]);
  if (argc > 9) args.bins_max = parse_size(argv[9]);
  if (argc > 10) args.running_median_seconds = parse_double(argv[10]);
  if (argc > 11) args.subband_channels = parse_size(argv[11]);
  if (argc > 12) args.ndm_per_nominal = parse_size(argv[12]);
  if (args.ndm == 0 || args.dm_step <= 0.0 || args.period_min <= 0.0 ||
      args.period_max <= args.period_min || args.bins_min <= 1 ||
      args.bins_max < args.bins_min || args.subband_channels == 0 ||
      args.ndm_per_nominal == 0 || !std::isfinite(args.dm_low) ||
      !std::isfinite(args.dm_step) || !std::isfinite(args.snr_threshold)) {
    throw std::invalid_argument("invalid CUDA DM-search benchmark arguments");
  }
  return args;
}

template <typename T>
std::vector<gaffa::DmPeak> run_typed(const gaffa::FilterbankData& filterbank,
                                     const Args& args,
                                     Timings& timings,
                                     std::size_t& dedispersed_bytes,
                                     std::size_t& float_bytes,
                                     std::size_t& preprocess_workspace,
                                     std::size_t& ffa_workspace) {
  const auto plan = gaffa::MultiDmDedispersionPlan{
      .dm_low = args.dm_low, .dm_step = args.dm_step, .ndm = args.ndm,
      .ref_frequency_mhz = filterbank.header.frequency_table.back(),
      .tsamp = filterbank.header.tsamp, .chan_begin = 0,
      .chan_end = static_cast<std::size_t>(filterbank.header.nchans)};
  const auto subband = gaffa::SubbandDedispersionOptions{
      .subband_channels = args.subband_channels,
      .ndm_per_nominal = args.ndm_per_nominal};
  const auto preprocess_plan = gaffa::make_riptide_preprocess_plan(
      filterbank.header.tsamp,
      {.running_median_width_seconds = args.running_median_seconds,
       .normalise = true});
  const auto input = gaffa::sample_view<T>(filterbank);
  auto dedispersed = gaffa::CudaDedispersedResult<
      std::conditional_t<std::is_same_v<T, float>, float, std::uint32_t>>{};
  timings.dedisperse = time_once([&] {
    dedispersed = gaffa::dedisperse_subband_cuda_device(
        input, filterbank.header.frequency_table, plan, subband);
  });
  const auto ffa_plan = gaffa::make_riptide_ffa_plan(
      dedispersed.shape.nsamples, filterbank.header.tsamp,
      {.period_min = args.period_min, .period_max = args.period_max,
       .bins_min = args.bins_min, .bins_max = args.bins_max});
  gaffa::CudaPreprocessProgram preprocess_program(
      preprocess_plan, {.device_id = dedispersed.device_id},
      {.series_tile_size = dedispersed.shape.ndm,
       .max_nsamples = dedispersed.shape.nsamples});
  gaffa::CudaFfaProgram ffa_program(
      ffa_plan, {.device_id = dedispersed.device_id},
      {.series_tile_size = dedispersed.shape.ndm});
  gaffa::CudaDeviceBuffer<float> device_series(dedispersed.size());
  timings.convert = time_once([&] {
    if constexpr (std::is_same_v<T, float>) {
      if (cudaMemcpy(device_series.data(), dedispersed.data.data(),
                     dedispersed.bytes(), cudaMemcpyDeviceToDevice) != cudaSuccess) {
        throw std::runtime_error("CUDA float batch D2D copy failed");
      }
    } else {
      gaffa::convert_time_series_batch_to_float_cuda(
          {.data = dedispersed.data.data(), .count = dedispersed.size(),
           .device_id = dedispersed.device_id},
          dedispersed.shape.ndm, dedispersed.shape.nsamples,
          device_series.as_span(dedispersed.device_id));
      if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("CUDA uint32-to-float conversion failed");
      }
    }
  });
  const gaffa::MutableCudaTimeSeriesBatchView mutable_batch{
      .data = device_series.data(), .nseries = dedispersed.shape.ndm,
      .nsamples = dedispersed.shape.nsamples, .device_id = dedispersed.device_id};
  timings.preprocess = time_once([&] {
    gaffa::preprocess_time_series_batch_inplace_cuda(preprocess_program,
                                                      mutable_batch);
    preprocess_program.synchronize();
  });
  gaffa::FfaBatchSearchResult ffa_result;
  timings.ffa = time_once([&] {
    ffa_result = gaffa::run_ffa_batch_cuda(
        ffa_program, mutable_batch.as_const(), {.snr_threshold = args.snr_threshold});
  });
  dedispersed_bytes = dedispersed.bytes();
  float_bytes = device_series.bytes();
  preprocess_workspace = preprocess_program.workspace_shape().total_bytes;
  ffa_workspace = ffa_program.workspace_shape().total_bytes;
  std::vector<gaffa::DmPeak> peaks;
  peaks.reserve(ffa_result.peaks.size());
  for (const auto& item : ffa_result.peaks) {
    peaks.push_back({.dm = args.dm_low + item.series_index * args.dm_step,
                     .dm_index = item.series_index, .peak = item.peak});
  }
  return peaks;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    gaffa::FilterbankData filterbank;
    const double read_seconds = time_once([&] { filterbank = gaffa::read_filterbank(args.path); });
    Timings timings;
    std::size_t dedispersed_bytes = 0, float_bytes = 0, preprocess_workspace = 0,
                ffa_workspace = 0;
    const auto peaks = std::visit(
        [&](const auto& values) {
          using T = typename std::decay_t<decltype(values)>::value_type;
          return run_typed<T>(filterbank, args, timings, dedispersed_bytes,
                              float_bytes, preprocess_workspace, ffa_workspace);
        }, filterbank.samples);
    std::vector<gaffa::Candidate> candidates;
    timings.candidate = time_once([&] {
      candidates = gaffa::select_candidates_cpu(
          peaks, filterbank.header.tsamp * filterbank.header.nsamples,
          {.frequency_cluster_radius = 0.1, .dm_cluster_radius = 50,
           .cluster_across_widths = true});
    });
    std::vector<gaffa::HarmonicCandidate> harmonics;
    timings.harmonic = time_once([&] {
      harmonics = gaffa::flag_harmonics_cpu(
          candidates, {.observation_seconds = filterbank.header.tsamp * filterbank.header.nsamples,
                       .frequency_low_mhz = filterbank.header.frequency_table.front(),
                       .frequency_high_mhz = filterbank.header.frequency_table.back()});
    });
    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "cuda_dm_search_begin file=\"" << args.path.string()
              << "\" ndm=" << args.ndm << " dm_low=" << args.dm_low
              << " dm_step=" << args.dm_step << " period_min=" << args.period_min
              << " period_max=" << args.period_max << " snr_threshold="
              << args.snr_threshold << " bins_min=" << args.bins_min
              << " bins_max=" << args.bins_max << '\n';
    std::cout << "timing read_seconds=" << read_seconds
              << " dedisperse_seconds=" << timings.dedisperse
              << " convert_seconds=" << timings.convert
              << " preprocess_seconds=" << timings.preprocess
              << " ffa_seconds=" << timings.ffa
              << " candidate_seconds=" << timings.candidate
              << " harmonic_seconds=" << timings.harmonic << '\n';
    std::cout << "memory dedispersed_bytes=" << dedispersed_bytes
              << " float_batch_bytes=" << float_bytes
              << " preprocess_workspace_bytes=" << preprocess_workspace
              << " ffa_workspace_bytes=" << ffa_workspace
              << " free_device_bytes=" << free_bytes
              << " total_device_bytes=" << total_bytes << '\n';
    std::cout << "result peaks=" << peaks.size() << " candidates=" << candidates.size()
              << " harmonic_records=" << harmonics.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
