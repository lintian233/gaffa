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
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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
  std::size_t max_peaks = 0;
  std::size_t print_peaks = 64;
  std::size_t max_candidates = 0;
};

struct Timings {
  double dedisperse = 0.0;
  double convert = 0.0;
  double preprocess = 0.0;
  double ffa = 0.0;
  double candidate = 0.0;
  double harmonic = 0.0;
  double read = 0.0;
  double total = 0.0;
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
  if (argc < 2 || argc > 16) {
    throw std::invalid_argument(
        "usage: cuda_dm_search_benchmark <file> [ndm dm_low dm_step "
        "period_min period_max snr bins_min bins_max median_seconds "
        "subband_channels ndm_per_nominal max_peaks print_peaks "
        "max_candidates]");
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
  if (argc > 13) args.max_peaks = parse_size(argv[13]);
  if (argc > 14) args.print_peaks = parse_size(argv[14]);
  if (argc > 15) args.max_candidates = parse_size(argv[15]);
  if (args.ndm == 0 || args.dm_step <= 0.0 || args.period_min <= 0.0 ||
      args.period_max <= args.period_min || args.bins_min <= 1 ||
      args.bins_max < args.bins_min || args.subband_channels == 0 ||
      args.ndm_per_nominal == 0 || !std::isfinite(args.dm_low) ||
      !std::isfinite(args.dm_step) || !std::isfinite(args.snr_threshold)) {
    throw std::invalid_argument("invalid CUDA DM-search benchmark arguments");
  }
  return args;
}

gaffa::CandidateSelectionOptions candidate_options(const Args& args) {
  return gaffa::CandidateSelectionOptions{
      .frequency_cluster_radius = 0.1,
      .dm_cluster_radius = 50,
      .cluster_across_widths = true,
      .max_candidates = 0,
  };
}

gaffa::HarmonicOptions harmonic_options() {
  return gaffa::HarmonicOptions{
      .max_harmonic = 32,
      .denominator_max = 100,
      .frequency_tolerance_bins = 1.5,
      .phase_distance_max = 1.0,
      .dm_distance_max = 3.0,
      .use_snr_consistency = false,
      .snr_distance_max = 3.0,
  };
}

gaffa::HarmonicContext harmonic_context(
    const gaffa::FilterbankHeader& header) {
  const auto [low, high] = std::minmax_element(header.frequency_table.begin(),
                                                header.frequency_table.end());
  return gaffa::HarmonicContext{
      .observation_seconds =
          header.tsamp * static_cast<double>(header.nsamples),
      .frequency_low_mhz = *low,
      .frequency_high_mhz = *high,
  };
}

void print_candidate(std::string_view label,
                     std::size_t rank,
                     const gaffa::Candidate& candidate) {
  const auto& best = candidate.best;
  const auto& peak = best.peak;
  std::cout << label
            << " rank=" << rank
            << " dm=" << best.dm
            << " dm_index=" << best.dm_index
            << " snr=" << peak.snr
            << " period=" << peak.period_seconds()
            << " frequency=" << peak.motion.frequency_hz
            << " width=" << peak.boxcar_width_bins
            << " duty_cycle=" << peak.duty_cycle
            << " peak_count=" << candidate.peak_count
            << " dm_index_min=" << candidate.extent.dm_index_min
            << " dm_index_max=" << candidate.extent.dm_index_max
            << " frequency_min=" << candidate.extent.frequency_hz.minimum
            << " frequency_max=" << candidate.extent.frequency_hz.maximum << '\n';
}

void print_harmonic_candidate(std::size_t rank,
                              const gaffa::HarmonicCandidate& candidate) {
  const auto& harmonic = candidate.harmonic;
  const auto& best = candidate.candidate.best;
  const auto& peak = best.peak;
  std::cout << "harmonic_candidate"
            << " rank=" << rank
            << " parent_rank=" << harmonic.parent_index
            << " ratio=" << harmonic.numerator << '/' << harmonic.denominator
            << " frequency_error_bins=" << harmonic.frequency_error_bins
            << " phase_distance=" << harmonic.phase_distance
            << " dm_distance=" << harmonic.dm_distance
            << " expected_snr=" << harmonic.expected_snr
            << " snr_distance=" << harmonic.snr_distance
            << " dm=" << best.dm
            << " dm_index=" << best.dm_index
            << " snr=" << peak.snr
            << " period=" << peak.period_seconds()
            << " frequency=" << peak.motion.frequency_hz
            << " width=" << peak.boxcar_width_bins
            << " duty_cycle=" << peak.duty_cycle
            << " peak_count=" << candidate.candidate.peak_count << '\n';
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
        ffa_program, mutable_batch.as_const(),
        {.snr_threshold = args.snr_threshold, .max_peaks = args.max_peaks});
  });
  dedispersed_bytes = dedispersed.bytes();
  float_bytes = device_series.bytes();
  preprocess_workspace = preprocess_program.workspace_shape().total_bytes;
  ffa_workspace = ffa_program.workspace_shape().total_bytes;
  std::vector<gaffa::DmPeak> peaks;
  peaks.reserve(ffa_result.peaks.size());
  for (const auto& item : ffa_result.peaks) {
    peaks.push_back({.dm = args.dm_low + item.series_index * args.dm_step,
                     .dm_index = item.series_index,
                     .peak = gaffa::periodic_peak_from_ffa(item.peak)});
  }
  return peaks;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    const auto total_begin = std::chrono::steady_clock::now();
    const Args args = parse_args(argc, argv);
    gaffa::FilterbankData filterbank;
    Timings timings;
    timings.read = time_once([&] { filterbank = gaffa::read_filterbank(args.path); });
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
          candidate_options(args));
    });
    std::vector<gaffa::HarmonicCandidate> flagged_candidates;
    std::vector<gaffa::Candidate> filtered_candidates;
    timings.harmonic = time_once([&] {
      flagged_candidates = gaffa::flag_harmonics_cpu(
          candidates, harmonic_context(filterbank.header), harmonic_options());
      filtered_candidates =
          gaffa::remove_harmonics_cpu(flagged_candidates, args.max_candidates);
    });
    timings.total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_begin).count();
    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "cuda_dm_search_begin file=\"" << args.path.string()
              << "\" ndm=" << args.ndm << " dm_low=" << args.dm_low
              << " dm_step=" << args.dm_step << " period_min=" << args.period_min
              << " period_max=" << args.period_max << " snr_threshold="
              << args.snr_threshold << " bins_min=" << args.bins_min
              << " bins_max=" << args.bins_max
              << " max_peaks=" << args.max_peaks
              << " max_candidates=" << args.max_candidates
              << " candidate_frequency_cluster_radius="
              << candidate_options(args).frequency_cluster_radius
              << " candidate_dm_cluster_radius="
              << candidate_options(args).dm_cluster_radius
              << " candidate_cluster_across_widths="
              << candidate_options(args).cluster_across_widths
              << " candidate_max_candidates="
              << candidate_options(args).max_candidates
              << " harmonic_max_harmonic=" << harmonic_options().max_harmonic
              << " harmonic_denominator_max="
              << harmonic_options().denominator_max
              << " harmonic_frequency_tolerance_bins="
              << harmonic_options().frequency_tolerance_bins
              << " harmonic_phase_distance_max="
              << harmonic_options().phase_distance_max
              << " harmonic_dm_distance_max="
              << harmonic_options().dm_distance_max
              << " harmonic_use_snr_consistency="
              << harmonic_options().use_snr_consistency
              << " print_peaks=" << args.print_peaks << '\n';
    std::cout << "timing read_seconds=" << timings.read
              << " dedisperse_seconds=" << timings.dedisperse
              << " convert_seconds=" << timings.convert
              << " preprocess_seconds=" << timings.preprocess
              << " ffa_seconds=" << timings.ffa
              << " candidate_seconds=" << timings.candidate
              << " harmonic_seconds=" << timings.harmonic
              << " total_seconds=" << timings.total << '\n';
    std::cout << "memory dedispersed_bytes=" << dedispersed_bytes
              << " float_batch_bytes=" << float_bytes
              << " preprocess_workspace_bytes=" << preprocess_workspace
              << " ffa_workspace_bytes=" << ffa_workspace
              << " free_device_bytes=" << free_bytes
              << " total_device_bytes=" << total_bytes << '\n';
    const std::size_t harmonic_count = std::count_if(
        flagged_candidates.begin(), flagged_candidates.end(),
        [](const gaffa::HarmonicCandidate& candidate) {
          return candidate.harmonic.is_harmonic;
        });
    std::cout << "result peaks=" << peaks.size()
              << " raw_candidates=" << candidates.size()
              << " harmonic_candidates=" << harmonic_count
              << " candidates=" << filtered_candidates.size() << '\n';

    const std::size_t raw_print_count =
        std::min(args.print_peaks, candidates.size());
    for (std::size_t rank = 0; rank < raw_print_count; ++rank) {
      print_candidate("raw_candidate", rank, candidates[rank]);
    }
    if (raw_print_count < candidates.size()) {
      std::cout << "result raw_candidates_omitted="
                << candidates.size() - raw_print_count << '\n';
    }
    std::size_t harmonic_printed = 0;
    for (std::size_t rank = 0;
         rank < flagged_candidates.size() && harmonic_printed < args.print_peaks;
         ++rank) {
      if (!flagged_candidates[rank].harmonic.is_harmonic) {
        continue;
      }
      print_harmonic_candidate(rank, flagged_candidates[rank]);
      ++harmonic_printed;
    }
    const std::size_t final_print_count =
        std::min(args.print_peaks, filtered_candidates.size());
    for (std::size_t rank = 0; rank < final_print_count; ++rank) {
      print_candidate("candidate", rank, filtered_candidates[rank]);
    }
    if (final_print_count < filtered_candidates.size()) {
      std::cout << "result candidates_omitted="
                << filtered_candidates.size() - final_print_count << '\n';
    }
    std::cout << "cuda_dm_search_end\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
