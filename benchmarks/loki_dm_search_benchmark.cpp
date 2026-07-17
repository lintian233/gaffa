#include "gaffa/candidate.h"
#include "gaffa/dedispersion_cuda.h"
#include "gaffa/filterbank.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/harmonic.h"
#include "gaffa/loki_pffa.h"
#include "gaffa/preprocessing.h"
#include "gaffa/preprocessing_cuda.h"
#include "gaffa/time_series_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
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
  std::size_t phase_bins_min = 256;
  std::size_t phase_bins_max = 256;
  double duty_cycle_max = 0.20;
  double running_median_seconds = 5.0;
  std::size_t subband_channels = 32;
  std::size_t ndm_per_nominal = 32;
  std::size_t print_peaks = 64;
  std::string window_mode = "truncate";
};

struct Timings {
  double read = 0.0;
  double dedisperse = 0.0;
  double convert = 0.0;
  double window = 0.0;
  double preprocess = 0.0;
  double loki_search = 0.0;
  double grouping = 0.0;
  double clustering = 0.0;
  double harmonic = 0.0;
  double total = 0.0;
};

enum class WindowMode {
  Truncate,
  ZeroPad,
};

// Describes a benchmark-local adaptation from an arbitrary valid dedispersed
// length to the power-of-two length required by Loki.
struct PrefixWindow {
  std::size_t source_nsamples = 0;
  std::size_t search_nsamples = 0;
  WindowMode mode = WindowMode::Truncate;

  [[nodiscard]] std::size_t discarded_tail_samples() const noexcept {
    return mode == WindowMode::Truncate
               ? source_nsamples - search_nsamples
               : 0;
  }

  [[nodiscard]] std::size_t padded_tail_samples() const noexcept {
    return mode == WindowMode::ZeroPad
               ? search_nsamples - source_nsamples
               : 0;
  }
};

using LokiDmPeak = gaffa::DmPeak;

template <typename Function>
double time_once(Function&& function) {
  const auto begin = std::chrono::steady_clock::now();
  function();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
      .count();
}

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::size_t parse_size(const char* value) { return std::stoull(value); }
double parse_double(const char* value) { return std::stod(value); }
float parse_float(const char* value) { return std::stof(value); }

Args parse_args(int argc, char** argv) {
  if (argc < 2 || argc > 17) {
    throw std::invalid_argument(
        "usage: loki_dm_search_benchmark <file> [ndm dm_low dm_step "
        "period_min period_max snr phase_bins_min phase_bins_max "
        "duty_cycle_max median_seconds subband_channels ndm_per_nominal "
        "print_peaks window_mode]");
  }
  Args args{.path = argv[1]};
  if (argc > 2) args.ndm = parse_size(argv[2]);
  if (argc > 3) args.dm_low = parse_double(argv[3]);
  if (argc > 4) args.dm_step = parse_double(argv[4]);
  if (argc > 5) args.period_min = parse_double(argv[5]);
  if (argc > 6) args.period_max = parse_double(argv[6]);
  if (argc > 7) args.snr_threshold = parse_float(argv[7]);
  if (argc > 8) args.phase_bins_min = parse_size(argv[8]);
  if (argc > 9) args.phase_bins_max = parse_size(argv[9]);
  if (argc > 10) args.duty_cycle_max = parse_double(argv[10]);
  if (argc > 11) args.running_median_seconds = parse_double(argv[11]);
  if (argc > 12) args.subband_channels = parse_size(argv[12]);
  if (argc > 13) args.ndm_per_nominal = parse_size(argv[13]);
  if (argc > 14) args.print_peaks = parse_size(argv[14]);
  if (argc > 15) args.window_mode = argv[15];

  if (args.ndm == 0 || !(args.dm_step > 0.0) || !(args.period_min > 0.0) ||
      !(args.period_max > args.period_min) || args.phase_bins_min < 2 ||
      args.phase_bins_max < args.phase_bins_min ||
      !(args.duty_cycle_max > 0.0 && args.duty_cycle_max < 1.0) ||
      args.subband_channels == 0 || args.ndm_per_nominal == 0 ||
      !std::isfinite(args.dm_low) || !std::isfinite(args.dm_step) ||
      !std::isfinite(args.snr_threshold) ||
      (args.window_mode != "truncate" && args.window_mode != "zero-pad")) {
    throw std::invalid_argument("invalid Loki DM-search benchmark arguments");
  }
  return args;
}

WindowMode parse_window_mode(const Args& args) {
  return args.window_mode == "truncate" ? WindowMode::Truncate
                                         : WindowMode::ZeroPad;
}

PrefixWindow make_prefix_window(std::size_t source_nsamples, WindowMode mode) {
  if (source_nsamples == 0) {
    throw std::invalid_argument("Loki benchmark requires non-empty time series");
  }
  if (mode == WindowMode::ZeroPad &&
      source_nsamples > std::numeric_limits<std::size_t>::max() / 2) {
    throw std::overflow_error("Loki benchmark zero-pad length overflows size_t");
  }
  return PrefixWindow{
      .source_nsamples = source_nsamples,
      .search_nsamples = mode == WindowMode::Truncate
                             ? std::bit_floor(source_nsamples)
                             : std::bit_ceil(source_nsamples),
      .mode = mode,
  };
}

gaffa::CandidateClusteringOptions candidate_options() {
  return gaffa::CandidateClusteringOptions{
      .max_phase_distance_cycles = 0.1,
      .max_dm_index_distance = 50,
      .cluster_across_widths = true,
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
    const gaffa::FilterbankHeader& header, double observation_seconds) {
  const auto [low, high] = std::minmax_element(header.frequency_table.begin(),
                                                header.frequency_table.end());
  return gaffa::HarmonicContext{
      .observation_seconds = observation_seconds,
      .frequency_low_mhz = *low,
      .frequency_high_mhz = *high,
  };
}

bool better_loki_peak(const LokiDmPeak& lhs, const LokiDmPeak& rhs) {
  if (lhs.peak.snr != rhs.peak.snr) {
    return lhs.peak.snr > rhs.peak.snr;
  }
  if (lhs.peak.motion.frequency_hz != rhs.peak.motion.frequency_hz) {
    return lhs.peak.motion.frequency_hz < rhs.peak.motion.frequency_hz;
  }
  return lhs.dm_index < rhs.dm_index;
}

void print_loki_peak(std::size_t rank, const LokiDmPeak& peak) {
  std::cout << "raw_loki_peak"
            << " rank=" << rank
            << " dm=" << peak.dm
            << " dm_index=" << peak.dm_index
            << " snr=" << peak.peak.snr
            << " period=" << peak.peak.period_seconds()
            << " frequency=" << peak.peak.motion.frequency_hz
            << " phase_bins=" << peak.peak.phase_bins
            << " width=" << peak.peak.boxcar_width_bins
            << " duty_cycle=" << peak.peak.duty_cycle;
  if (peak.peak.motion.order >= gaffa::MotionOrder::Acceleration) {
    std::cout << " acceleration_m_per_s2="
              << peak.peak.motion.acceleration_m_per_s2;
  }
  if (peak.peak.motion.order >= gaffa::MotionOrder::Jerk) {
    std::cout << " jerk_m_per_s3=" << peak.peak.motion.jerk_m_per_s3;
  }
  if (peak.peak.motion.order >= gaffa::MotionOrder::Snap) {
    std::cout << " snap_m_per_s4=" << peak.peak.motion.snap_m_per_s4;
  }
  std::cout << '\n';
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
            << " peak_count=" << candidate.member_count
            << " dm_index_min=" << candidate.extent.dm_index_min
            << " dm_index_max=" << candidate.extent.dm_index_max
            << " frequency_min=" << candidate.extent.frequency_hz.minimum
            << " frequency_max=" << candidate.extent.frequency_hz.maximum << '\n';
}

void print_harmonic_candidate(std::size_t rank,
                              const gaffa::Candidate& candidate,
                              const gaffa::HarmonicRelation& harmonic) {
  const auto& best = candidate.best;
  const auto& peak = best.peak;
  std::cout << "harmonic_candidate"
            << " rank=" << rank
            << " parent_rank=" << harmonic.parent_index
            << " ratio=" << harmonic.numerator << '/' << harmonic.denominator
            << " frequency_error_bins=" << harmonic.frequency_error_bins
            << " maximum_phase_drift_cycles="
            << harmonic.maximum_phase_drift_cycles
            << " phase_drift_time_seconds="
            << harmonic.phase_drift_time_seconds
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
            << " peak_count=" << candidate.member_count << '\n';
}

template <typename T>
gaffa::CudaDeviceBuffer<float> make_full_float_batch(
    const gaffa::CudaDedispersedResult<T>& dedispersed) {
  gaffa::CudaDeviceBuffer<float> full(dedispersed.size());
  check_cuda(cudaSetDevice(dedispersed.device_id), "cudaSetDevice");

  if constexpr (std::is_same_v<T, float>) {
    check_cuda(cudaMemcpyAsync(full.data(), dedispersed.data.data(),
                               dedispersed.bytes(), cudaMemcpyDeviceToDevice,
                               nullptr),
               "cudaMemcpyAsync Loki float batch");
  } else {
    // Keep the generic uint32-to-float primitive in the library untouched.
    // This full-resolution temporary exists only in the benchmark adapter.
    gaffa::convert_time_series_batch_to_float_cuda(
        {.data = dedispersed.data.data(), .count = dedispersed.size(),
         .device_id = dedispersed.device_id},
        dedispersed.shape.ndm,
        dedispersed.shape.nsamples,
        full.as_span(dedispersed.device_id));
  }
  check_cuda(cudaStreamSynchronize(nullptr),
             "cudaStreamSynchronize Loki float batch");
  return full;
}

void copy_rows_to_window(const gaffa::CudaDeviceBuffer<float>& source,
                         std::size_t source_nsamples,
                         gaffa::CudaDeviceBuffer<float>& destination,
                         std::size_t destination_nsamples,
                         std::size_t copied_nsamples,
                         std::size_t nseries) {
  check_cuda(cudaMemcpy2DAsync(
                 destination.data(), destination_nsamples * sizeof(float),
                 source.data(), source_nsamples * sizeof(float),
                 copied_nsamples * sizeof(float), nseries,
                 cudaMemcpyDeviceToDevice, nullptr),
             "cudaMemcpy2DAsync Loki search window");
}

template <typename T>
std::vector<LokiDmPeak> run_typed(const gaffa::FilterbankData& filterbank,
                                  const Args& args,
                                  Timings& timings,
                                  PrefixWindow& window,
                                  std::size_t& dedispersed_bytes,
                                  std::size_t& source_float_bytes,
                                  std::size_t& search_float_bytes,
                                  std::size_t& preprocess_workspace_bytes) {
  const auto dedispersion_plan = gaffa::MultiDmDedispersionPlan{
      .dm_low = args.dm_low,
      .dm_step = args.dm_step,
      .ndm = args.ndm,
      .ref_frequency_mhz = filterbank.header.frequency_table.back(),
      .tsamp = filterbank.header.tsamp,
      .chan_begin = 0,
      .chan_end = static_cast<std::size_t>(filterbank.header.nchans),
  };
  const auto subband_options = gaffa::SubbandDedispersionOptions{
      .subband_channels = args.subband_channels,
      .ndm_per_nominal = args.ndm_per_nominal,
  };
  const auto preprocess_plan = gaffa::make_riptide_preprocess_plan(
      filterbank.header.tsamp,
      {.running_median_width_seconds = args.running_median_seconds,
       .normalise = true});

  const auto input = gaffa::sample_view<T>(filterbank);
  auto dedispersed = gaffa::CudaDedispersedResult<
      std::conditional_t<std::is_same_v<T, float>, float, std::uint32_t>>{};
  timings.dedisperse = time_once([&] {
    dedispersed = gaffa::dedisperse_subband_cuda_device(
        input, filterbank.header.frequency_table, dedispersion_plan,
        subband_options);
  });

  window = make_prefix_window(dedispersed.shape.nsamples, parse_window_mode(args));
  dedispersed_bytes = dedispersed.bytes();
  const int device_id = dedispersed.device_id;
  gaffa::CudaDeviceBuffer<float> source_float;
  timings.convert = time_once([&] {
    source_float = make_full_float_batch(dedispersed);
  });
  source_float_bytes = source_float.bytes();
  dedispersed = {};

  gaffa::CudaDeviceBuffer<float> search_float(
      args.ndm * window.search_nsamples);
  if (window.mode == WindowMode::Truncate) {
    timings.window = time_once([&] {
      copy_rows_to_window(source_float, window.source_nsamples, search_float,
                          window.search_nsamples, window.search_nsamples,
                          args.ndm);
      check_cuda(cudaStreamSynchronize(nullptr),
                 "cudaStreamSynchronize Loki truncated window");
    });
    source_float = {};
  }

  gaffa::CudaPreprocessProgram preprocess_program(
      preprocess_plan, {.device_id = device_id},
      {.series_tile_size = args.ndm,
       .max_nsamples = window.mode == WindowMode::Truncate
                           ? window.search_nsamples
                           : window.source_nsamples});
  const gaffa::MutableCudaTimeSeriesBatchView preprocess_batch{
      .data = window.mode == WindowMode::Truncate ? search_float.data()
                                                   : source_float.data(),
      .nseries = args.ndm,
      .nsamples = window.mode == WindowMode::Truncate
                      ? window.search_nsamples
                      : window.source_nsamples,
      .device_id = device_id,
  };
  timings.preprocess = time_once([&] {
    gaffa::preprocess_time_series_batch_inplace_cuda(preprocess_program,
                                                      preprocess_batch);
    preprocess_program.synchronize();
  });
  preprocess_workspace_bytes = preprocess_program.workspace_shape().total_bytes;
  if (window.mode == WindowMode::ZeroPad) {
    timings.window = time_once([&] {
      if (window.padded_tail_samples() != 0) {
        check_cuda(cudaMemsetAsync(search_float.data(), 0, search_float.bytes(),
                                  nullptr),
                   "cudaMemsetAsync Loki zero-padded window");
      }
      copy_rows_to_window(source_float, window.source_nsamples, search_float,
                          window.search_nsamples, window.source_nsamples,
                          args.ndm);
      check_cuda(cudaStreamSynchronize(nullptr),
                 "cudaStreamSynchronize Loki zero-padded window");
    });
    source_float = {};
  }
  search_float_bytes = search_float.bytes();

  const auto loki_plan = gaffa::make_loki_pffa_plan(
      window.search_nsamples, filterbank.header.tsamp,
      {.frequency_hz = {.minimum = 1.0 / args.period_max,
                        .maximum = 1.0 / args.period_min}},
      {.phase_bins_min = args.phase_bins_min,
       .phase_bins_max = args.phase_bins_max,
       .duty_cycle_max = args.duty_cycle_max,
       .snr_threshold = args.snr_threshold});
  gaffa::LokiPffaProgram program(
      loki_plan, {.device_id = device_id});

  std::vector<LokiDmPeak> peaks;
  timings.loki_search = time_once([&] {
    for (std::size_t dm_index = 0; dm_index < args.ndm; ++dm_index) {
      const auto loki_peaks = program.search({
          .data = search_float.data() + dm_index * window.search_nsamples,
          .count = window.search_nsamples,
          .device_id = device_id,
      });
      peaks.reserve(peaks.size() + loki_peaks.size());
      for (const auto& peak : loki_peaks) {
        peaks.push_back({.dm = args.dm_low +
                                 static_cast<double>(dm_index) * args.dm_step,
                         .dm_index = dm_index,
                         .peak = peak});
      }
    }
  });
  return peaks;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // Benchmark output is also used for candidate comparisons. Preserve enough
    // significant digits to round-trip all double-valued search coordinates.
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    const auto total_begin = std::chrono::steady_clock::now();
    const Args args = parse_args(argc, argv);
    gaffa::FilterbankData filterbank;
    Timings timings;
    timings.read = time_once([&] { filterbank = gaffa::read_filterbank(args.path); });

    PrefixWindow window;
    std::size_t dedispersed_bytes = 0;
    std::size_t source_float_bytes = 0;
    std::size_t search_float_bytes = 0;
    std::size_t preprocess_workspace_bytes = 0;
    const auto raw_peaks = std::visit(
        [&](const auto& values) {
          using T = typename std::decay_t<decltype(values)>::value_type;
          return run_typed<T>(filterbank, args, timings, window,
                              dedispersed_bytes, source_float_bytes,
                              search_float_bytes,
                              preprocess_workspace_bytes);
        },
        filterbank.samples);

    std::vector<gaffa::DmPeakGroups> peak_groups;
    const double searched_duration_seconds =
        static_cast<double>(window.search_nsamples) * filterbank.header.tsamp;
    gaffa::CandidateSet candidates;
    timings.grouping = time_once([&] {
      peak_groups =
          gaffa::group_dm_peak_batch_cpu(raw_peaks, searched_duration_seconds);
    });
    timings.clustering = time_once([&] {
      candidates = gaffa::cluster_dm_peak_groups_cpu(
          peak_groups, searched_duration_seconds,
          candidate_options());
    });
    std::vector<gaffa::HarmonicRelation> harmonic_candidates;
    std::vector<std::size_t> filtered_candidates;
    timings.harmonic = time_once([&] {
      harmonic_candidates = gaffa::flag_harmonics_cpu(
          candidates,
          harmonic_context(filterbank.header, searched_duration_seconds),
          harmonic_options());
      filtered_candidates =
          gaffa::remove_harmonics_cpu(candidates, harmonic_candidates);
    });
    timings.total = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - total_begin)
                        .count();

    std::vector<LokiDmPeak> sorted_raw = raw_peaks;
    std::sort(sorted_raw.begin(), sorted_raw.end(), better_loki_peak);
    const std::size_t harmonic_count = std::count_if(
        harmonic_candidates.begin(), harmonic_candidates.end(),
        [](const gaffa::HarmonicRelation& relation) {
          return relation.is_harmonic;
        });
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    std::cout << "loki_dm_search_begin file=\"" << args.path.string()
              << "\" ndm=" << args.ndm
              << " dm_low=" << args.dm_low
              << " dm_step=" << args.dm_step
              << " period_min=" << args.period_min
              << " period_max=" << args.period_max
              << " frequency_min=" << 1.0 / args.period_max
              << " frequency_max=" << 1.0 / args.period_min
              << " snr_threshold=" << args.snr_threshold
              << " phase_bins_min=" << args.phase_bins_min
              << " phase_bins_max=" << args.phase_bins_max
              << " duty_cycle_max=" << args.duty_cycle_max
              << " taylor_model=backend_plan"
              << " candidate_projection=taylor_phase"
              << " window_mode=" << args.window_mode
              << " print_peaks=" << args.print_peaks << '\n';
    std::cout << "window source_nsamples=" << window.source_nsamples
              << " loki_search_nsamples=" << window.search_nsamples
              << " discarded_tail_samples=" << window.discarded_tail_samples()
              << " padded_tail_samples=" << window.padded_tail_samples()
              << " discarded_tail_fraction="
              << static_cast<double>(window.discarded_tail_samples()) /
                     static_cast<double>(window.source_nsamples)
              << " padded_tail_fraction="
              << static_cast<double>(window.padded_tail_samples()) /
                     static_cast<double>(window.search_nsamples)
              << '\n';
    std::cout << "timing read_seconds=" << timings.read
              << " dedisperse_seconds=" << timings.dedisperse
              << " convert_seconds=" << timings.convert
              << " window_seconds=" << timings.window
              << " preprocess_seconds=" << timings.preprocess
              << " loki_search_seconds=" << timings.loki_search
              << " grouping_seconds=" << timings.grouping
              << " clustering_seconds=" << timings.clustering
              << " harmonic_seconds=" << timings.harmonic
              << " total_seconds=" << timings.total << '\n';
    std::cout << "memory dedispersed_bytes=" << dedispersed_bytes
              << " source_float_temp_bytes=" << source_float_bytes
              << " search_float_bytes=" << search_float_bytes
              << " preprocess_workspace_bytes=" << preprocess_workspace_bytes
              << " free_device_bytes=" << free_bytes
              << " total_device_bytes=" << total_bytes << '\n';
    std::size_t local_group_count = 0;
    for (const auto& groups : peak_groups) {
      local_group_count += groups.groups.size();
    }
    std::cout << "result raw_loki_peaks=" << raw_peaks.size()
              << " local_groups=" << local_group_count
              << " clustered_candidates=" << candidates.candidates.size()
              << " harmonic_candidates=" << harmonic_count
              << " candidates=" << filtered_candidates.size() << '\n';

    const std::size_t raw_print_count =
        std::min(args.print_peaks, sorted_raw.size());
    for (std::size_t rank = 0; rank < raw_print_count; ++rank) {
      print_loki_peak(rank, sorted_raw[rank]);
    }
    if (raw_print_count < sorted_raw.size()) {
      std::cout << "result raw_loki_peaks_omitted="
                << sorted_raw.size() - raw_print_count << '\n';
    }
    const std::size_t candidate_print_count =
        std::min(args.print_peaks, candidates.candidates.size());
    for (std::size_t rank = 0; rank < candidate_print_count; ++rank) {
      print_candidate("clustered_candidate", rank,
                      candidates.candidates[rank]);
    }
    if (candidate_print_count < candidates.candidates.size()) {
      std::cout << "result clustered_candidates_omitted="
                << candidates.candidates.size() - candidate_print_count
                << '\n';
    }
    std::size_t harmonic_printed = 0;
    for (std::size_t rank = 0;
         rank < harmonic_candidates.size() && harmonic_printed < args.print_peaks;
         ++rank) {
      if (harmonic_candidates[rank].is_harmonic) {
        print_harmonic_candidate(rank, candidates.candidates[rank],
                                 harmonic_candidates[rank]);
        ++harmonic_printed;
      }
    }
    const std::size_t final_print_count =
        std::min(args.print_peaks, filtered_candidates.size());
    for (std::size_t rank = 0; rank < final_print_count; ++rank) {
      print_candidate("candidate", rank,
                      candidates.candidates[filtered_candidates[rank]]);
    }
    if (final_print_count < filtered_candidates.size()) {
      std::cout << "result candidates_omitted="
                << filtered_candidates.size() - final_print_count << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
