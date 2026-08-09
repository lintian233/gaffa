#include "search.h"

#include "cuda_worker.h"

#include "gaffa/candidate_analysis.h"
#include "gaffa/dedispersion.h"
#include "gaffa/dedispersion_cuda.h"
#include "gaffa/dm_search.h"
#include "gaffa/filterbank.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/harmonic.h"
#include "gaffa/preprocessing.h"

#include "progress.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace gaffa_search {
namespace {

struct DmRangeRuntime {
  std::size_t id = 0;
  DmRangeConfig config;
  std::size_t global_dm_index_begin = 0;
  std::vector<double> dm_values;
};

struct SearchRunResult {
  SearchRunInfo info;
  gaffa::DmPeaks peaks;
  bool complete = true;
  std::vector<std::string> warnings;
};

struct RawSearchResult {
  gaffa::DmPeaks peaks;
  std::vector<SearchRunInfo> runs;
  bool complete = true;
  std::vector<std::string> warnings;
};

struct Tile {
  std::size_t local_begin = 0;
  std::size_t count = 0;
};

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

std::vector<DmRangeRuntime> make_dm_ranges(const Config& config) {
  std::vector<DmRangeRuntime> result;
  result.reserve(config.dm_ranges.size());
  std::size_t global_index_begin = 0;
  for (std::size_t range_index = 0; range_index < config.dm_ranges.size();
       ++range_index) {
    const DmRangeConfig& range = config.dm_ranges[range_index];
    std::vector<double> values(range.ndm);
    for (std::size_t index = 0; index < range.ndm; ++index) {
      values[index] = range.dm_low +
                      static_cast<double>(index) * range.dm_step;
    }
    result.push_back(DmRangeRuntime{
        .id = range_index,
        .config = range,
        .global_dm_index_begin = global_index_begin,
        .dm_values = std::move(values),
    });
    global_index_begin += range.ndm;
  }
  return result;
}

gaffa::MultiDmDedispersionPlan make_dedispersion_plan(
    const gaffa::FilterbankHeader& header, const DmRangeConfig& range) {
  return gaffa::MultiDmDedispersionPlan{
      .dm_low = range.dm_low,
      .dm_step = range.dm_step,
      .ndm = range.ndm,
      .ref_frequency_mhz = header.frequency_table.back(),
      .tsamp = header.tsamp,
      .chan_begin = 0,
      .chan_end = static_cast<std::size_t>(header.nchans),
  };
}

gaffa::SubbandDedispersionOptions make_subband_options(
    const Config& config) {
  return gaffa::SubbandDedispersionOptions{
      .subband_channels = config.subband_channels,
      .ndm_per_nominal = config.ndm_per_nominal,
  };
}

gaffa::CudaDedispersionOptions make_cuda_options(const Config& config) {
  return gaffa::CudaDedispersionOptions{
      .device_id = config.dedispersion_device,
      .threads_per_block = 256,
  };
}

gaffa::PreprocessPlan make_preprocess_plan(const Config& config,
                                           double tsamp) {
  if (config.preprocess == "none") {
    return {};
  }
  if (config.preprocess == "normalise") {
    return gaffa::PreprocessPlan{
        .steps = {gaffa::PreprocessStep{
            .kind = gaffa::PreprocessStepKind::Normalise,
        }},
    };
  }
  return gaffa::make_riptide_preprocess_plan(
      tsamp,
      gaffa::RiptidePreprocessOptions{
          .running_median_width_seconds = config.running_median_seconds,
          .normalise = true,
      });
}

gaffa::FfaSearchPlan make_native_plan(const SearchRangeConfig& search,
                                      std::size_t nsamples, double tsamp) {
  const double min_period = tsamp * static_cast<double>(search.bins_min);
  if (search.period_min < min_period) {
    throw std::invalid_argument(
        "search period_min is smaller than tsamp * bins_min");
  }
  return gaffa::make_riptide_ffa_plan(
      nsamples, tsamp,
      gaffa::RiptideFfaPlanOptions{
          .period_min = search.period_min,
          .period_max = search.period_max,
          .bins_min = search.bins_min,
          .bins_max = search.bins_max,
      });
}

std::size_t next_power_of_two(std::size_t value) {
  std::size_t result = 1;
  while (result < value) {
    if (result > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("search window length overflow");
    }
    result *= 2;
  }
  return result;
}

std::size_t floor_power_of_two(std::size_t value) {
  if (value == 0) {
    throw std::invalid_argument("search input length must be non-zero");
  }
  std::size_t result = 1;
  while (result <= value / 2) {
    result *= 2;
  }
  return result;
}

std::size_t prepared_length(const SearchRangeConfig& search,
                            std::size_t source_nsamples) {
  if (search.backend == Backend::LokiCuda &&
      search.window_mode == WindowMode::Truncate) {
    return floor_power_of_two(source_nsamples);
  }
  if (search.window_mode == WindowMode::ZeroPad) {
    return next_power_of_two(source_nsamples);
  }
  return source_nsamples;
}

gaffa::CandidateOptions make_candidate_options(const Config& config) {
  gaffa::CandidateOptions options;
  options.clustering.max_dm_distance = config.candidate_dm_radius;
  options.selection.snr_min = 0.0F;
  options.selection.max_candidates = config.max_candidates;
  return options;
}

gaffa::HarmonicContext make_harmonic_context(
    const gaffa::FilterbankHeader& header, double observation_seconds) {
  const auto [low, high] = std::minmax_element(
      header.frequency_table.begin(), header.frequency_table.end());
  return gaffa::HarmonicContext{
      .observation_seconds = observation_seconds,
      .frequency_low_mhz = *low,
      .frequency_high_mhz = *high,
  };
}

std::vector<Tile> make_tiles(std::size_t ndm, std::size_t tile_size) {
  std::vector<Tile> tiles;
  for (std::size_t begin = 0; begin < ndm;) {
    const std::size_t count = std::min(tile_size, ndm - begin);
    tiles.push_back(Tile{.local_begin = begin, .count = count});
    begin += count;
  }
  return tiles;
}

void check_device_ids(const std::vector<int>& devices) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGetDeviceCount: ") +
                             cudaGetErrorString(status));
  }
  for (const int device : devices) {
    if (device < 0 || device >= device_count) {
      throw std::invalid_argument("CUDA device id is out of range");
    }
  }
}

template <typename T>
using DmResultValue =
    std::conditional_t<std::is_same_v<T, float>, float, std::uint32_t>;

template <typename T>
gaffa::DedispersedResult<DmResultValue<T>> dedisperse(
    const gaffa::FilterbankData& filterbank, const Config& config,
    const DmRangeConfig& range) {
  const auto input = gaffa::sample_view<T>(filterbank);
  const auto plan = make_dedispersion_plan(filterbank.header, range);
  const auto subband = make_subband_options(config);
  if (config.dedispersion_backend == DedispersionBackend::CpuSubband) {
    return gaffa::dedisperse_subband_cpu(
        input, filterbank.header.frequency_table, plan, subband);
  }
  return gaffa::dedisperse_subband_cuda(
      input, filterbank.header.frequency_table, plan, subband,
      make_cuda_options(config));
}

void append_checked(gaffa::DmPeaks& destination, gaffa::DmPeaks source,
                    const Config& config) {
  if (config.max_total_raw_peaks != 0 &&
      (destination.size() > config.max_total_raw_peaks ||
       source.size() > config.max_total_raw_peaks - destination.size())) {
    throw std::runtime_error(
        "maximum total raw peak limit exceeded");
  }
  destination.insert(destination.end(),
                     std::make_move_iterator(source.begin()),
                     std::make_move_iterator(source.end()));
}

template <typename ValueT>
gaffa::DmPeaks run_native_cpu(
    const gaffa::DedispersedResult<ValueT>& dedispersed,
    const DmRangeRuntime& dm_range, const SearchRangeConfig& search,
    const Config& config, double tsamp,
    std::optional<std::size_t> peak_limit) {
  if (search.window_mode != WindowMode::Truncate) {
    throw std::invalid_argument(
        "native-cpu currently supports only truncate window mode");
  }
  const auto plan = make_native_plan(search, dedispersed.shape.nsamples, tsamp);
  const auto preprocess = make_preprocess_plan(config, tsamp);
  auto peaks = gaffa::search_dm_ffa_cpu(
      dedispersed, gaffa::DmTrialView{
                      .values = dm_range.dm_values,
                      .index_offset = dm_range.global_dm_index_begin,
                  },
      plan,
      gaffa::DmFfaOptions{
          .preprocess = preprocess,
          .search = {
              .snr_threshold = config.snr_threshold,
              .max_peaks = config.max_peaks,
          },
      });
  if (peak_limit && peaks.size() > *peak_limit) {
    throw std::runtime_error("maximum total raw peak limit exceeded");
  }
  return peaks;
}

template <typename ValueT>
CudaSearchResult run_cuda_phase(
    const gaffa::DedispersedResult<ValueT>& dedispersed,
    const DmRangeRuntime& dm_range, const SearchRangeConfig& search,
    const Config& config, double tsamp, std::vector<int> devices,
    std::optional<std::size_t> peak_limit, ProgressTracker* progress) {
  check_device_ids(devices);
  const auto tiles = make_tiles(dedispersed.shape.ndm, config.dm_tile_size);
  std::vector<std::unique_ptr<CudaWorker>> workers;
  workers.reserve(devices.size());
  for (const int device : devices) {
    workers.push_back(std::make_unique<CudaWorker>(device,
                                                   config.dm_tile_size));
    if (search.backend == Backend::NativeCuda) {
      workers.back()->prepare_native(
          search, dedispersed.shape.nsamples, tsamp, config.snr_threshold,
          config.max_peaks, config.preprocess, config.running_median_seconds,
          config.resources.native_cuda.max_peak_memory_bytes,
          search.reduction);
    } else {
#ifdef GAFFA_SEARCH_ENABLE_LOKI
      workers.back()->prepare_loki(
          search, dedispersed.shape.nsamples, tsamp, config.snr_threshold,
          config.max_peaks, config.preprocess, config.running_median_seconds,
          search.reduction);
#else
      throw std::runtime_error(
          "loki-cuda was requested but GAFFA_ENABLE_LOKI is disabled");
#endif
    }
  }

  std::vector<CudaSearchResult> worker_results(workers.size());
  std::vector<std::exception_ptr> worker_errors(workers.size());
  std::atomic<std::size_t> next_tile = 0;
  std::atomic<bool> cancelled = false;
  std::atomic<std::size_t> produced = 0;
  std::vector<std::thread> threads;
  threads.reserve(workers.size());

  for (std::size_t worker_index = 0; worker_index < workers.size();
       ++worker_index) {
    threads.emplace_back([&, worker_index] {
      try {
        while (!cancelled.load(std::memory_order_acquire)) {
          const std::size_t tile_index =
              next_tile.fetch_add(1, std::memory_order_relaxed);
          if (tile_index >= tiles.size()) {
            break;
          }
          const Tile tile = tiles[tile_index];
          const std::size_t offset = checked_multiply(
              tile.local_begin, dedispersed.shape.nsamples,
              "DM tile offset overflow");
          const std::size_t count = checked_multiply(
              tile.count, dedispersed.shape.nsamples,
              "DM tile size overflow");
          const auto source = std::span<const ValueT>(
              dedispersed.data.data() + offset, count);
          const auto dms = std::span<const double>(
              dm_range.dm_values.data() + tile.local_begin, tile.count);
          CudaSearchResult search_result;
          if constexpr (std::is_same_v<ValueT, std::uint32_t>) {
            if (search.backend == Backend::NativeCuda) {
              search_result = workers[worker_index]->run_native(
                  source, tile.count, dedispersed.shape.nsamples, dms,
                  dm_range.global_dm_index_begin + tile.local_begin);
            } else {
#ifdef GAFFA_SEARCH_ENABLE_LOKI
              search_result = workers[worker_index]->run_loki(
                  source, tile.count, dedispersed.shape.nsamples, dms,
                  dm_range.global_dm_index_begin + tile.local_begin);
#endif
            }
          } else {
            if (search.backend == Backend::NativeCuda) {
              search_result = workers[worker_index]->run_native(
                  source, tile.count, dedispersed.shape.nsamples, dms,
                  dm_range.global_dm_index_begin + tile.local_begin);
            } else {
#ifdef GAFFA_SEARCH_ENABLE_LOKI
              search_result = workers[worker_index]->run_loki(
                  source, tile.count, dedispersed.shape.nsamples, dms,
                  dm_range.global_dm_index_begin + tile.local_begin);
#endif
            }
          }
          gaffa::DmPeaks& peaks = search_result.peaks;
          const std::size_t count_after =
              produced.fetch_add(peaks.size(), std::memory_order_relaxed) +
              peaks.size();
          if (peak_limit && count_after > *peak_limit) {
            cancelled.store(true, std::memory_order_release);
            throw std::runtime_error(
                "maximum total raw peak limit exceeded");
          }
          auto& local = worker_results[worker_index];
          local.complete &= search_result.complete;
          local.warnings.insert(local.warnings.end(),
                                search_result.warnings.begin(),
                                search_result.warnings.end());
          local.peaks.insert(local.peaks.end(),
                       std::make_move_iterator(peaks.begin()),
                       std::make_move_iterator(peaks.end()));
          if (progress != nullptr) {
            progress->complete_search_units();
          }
        }
      } catch (...) {
        worker_errors[worker_index] = std::current_exception();
        cancelled.store(true, std::memory_order_release);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  std::exception_ptr first_error;
  for (const auto& error : worker_errors) {
    if (error != nullptr) {
      first_error = error;
      break;
    }
  }
  for (auto& worker : workers) {
    try {
      worker->reset();
    } catch (...) {
      if (first_error == nullptr) {
        first_error = std::current_exception();
      }
    }
  }
  if (first_error != nullptr) {
    std::rethrow_exception(first_error);
  }
  gaffa::DmPeaks output;
  CudaSearchResult result;
  for (auto& local : worker_results) {
    result.complete &= local.complete;
    result.warnings.insert(result.warnings.end(), local.warnings.begin(),
                           local.warnings.end());
    output.insert(output.end(), std::make_move_iterator(local.peaks.begin()),
                  std::make_move_iterator(local.peaks.end()));
  }
  result.peaks = std::move(output);
  return result;
}

bool dm_peak_less(const gaffa::DmPeak& lhs, const gaffa::DmPeak& rhs) {
  if (lhs.dm_index != rhs.dm_index) {
    return lhs.dm_index < rhs.dm_index;
  }
  if (lhs.peak.motion.frequency_hz != rhs.peak.motion.frequency_hz) {
    return lhs.peak.motion.frequency_hz < rhs.peak.motion.frequency_hz;
  }
  if (lhs.peak.motion.acceleration_m_per_s2 !=
      rhs.peak.motion.acceleration_m_per_s2) {
    return lhs.peak.motion.acceleration_m_per_s2 <
           rhs.peak.motion.acceleration_m_per_s2;
  }
  if (lhs.peak.motion.jerk_m_per_s3 != rhs.peak.motion.jerk_m_per_s3) {
    return lhs.peak.motion.jerk_m_per_s3 < rhs.peak.motion.jerk_m_per_s3;
  }
  if (lhs.peak.motion.snap_m_per_s4 != rhs.peak.motion.snap_m_per_s4) {
    return lhs.peak.motion.snap_m_per_s4 < rhs.peak.motion.snap_m_per_s4;
  }
  if (lhs.peak.phase_bins != rhs.peak.phase_bins) {
    return lhs.peak.phase_bins < rhs.peak.phase_bins;
  }
  if (lhs.peak.boxcar_width_bins != rhs.peak.boxcar_width_bins) {
    return lhs.peak.boxcar_width_bins < rhs.peak.boxcar_width_bins;
  }
  if (lhs.peak.phase_bin != rhs.peak.phase_bin) {
    return lhs.peak.phase_bin < rhs.peak.phase_bin;
  }
  if (lhs.peak.snr != rhs.peak.snr) {
    return lhs.peak.snr > rhs.peak.snr;
  }
  return lhs.dm < rhs.dm;
}

void sort_peaks(gaffa::DmPeaks& peaks) {
  std::sort(peaks.begin(), peaks.end(), dm_peak_less);
}

std::optional<std::size_t> remaining_peak_limit(
    const gaffa::DmPeaks& peaks, const Config& config) {
  if (config.max_total_raw_peaks == 0) {
    return std::nullopt;
  }
  if (peaks.size() > config.max_total_raw_peaks) {
    throw std::runtime_error("maximum total raw peak limit exceeded");
  }
  return config.max_total_raw_peaks - peaks.size();
}

SearchCoordinate make_search_coordinate(std::size_t source_nsamples,
                                         std::size_t prepared_nsamples,
                                         double tsamp) {
  const std::size_t valid_nsamples =
      std::min(source_nsamples, prepared_nsamples);
  return SearchCoordinate{
      .start_time_seconds = 0.0,
      .valid_duration_seconds =
          static_cast<double>(valid_nsamples) * tsamp,
      .tsamp_seconds = tsamp,
      .source_nsamples = source_nsamples,
      .prepared_nsamples = prepared_nsamples,
      .padded = prepared_nsamples > source_nsamples,
  };
}

double common_valid_duration(std::span<const SearchRunInfo> runs) {
  if (runs.empty()) {
    throw std::invalid_argument("candidate search produced no search runs");
  }
  const double start_time = runs.front().coordinate.start_time_seconds;
  double duration = std::numeric_limits<double>::infinity();
  for (const SearchRunInfo& run : runs) {
    const SearchCoordinate& coordinate = run.coordinate;
    if (!std::isfinite(coordinate.start_time_seconds) ||
        !std::isfinite(coordinate.valid_duration_seconds) ||
        !std::isfinite(coordinate.tsamp_seconds) ||
        !(coordinate.valid_duration_seconds > 0.0) ||
        !(coordinate.tsamp_seconds > 0.0)) {
      throw std::invalid_argument(
          "search run has an invalid candidate time coordinate");
    }
    const double tolerance =
        16.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, std::abs(start_time),
                  std::abs(coordinate.start_time_seconds)});
    if (std::abs(coordinate.start_time_seconds - start_time) > tolerance) {
      throw std::invalid_argument(
          "search runs have different time origins; coordinate normalization "
          "is not available");
    }
    duration = std::min(duration, coordinate.valid_duration_seconds);
  }
  if (!(duration > 0.0) || !std::isfinite(duration)) {
    throw std::invalid_argument(
        "search runs have no common valid observation duration");
  }
  return duration;
}

template <typename T>
void run_dm_range(const gaffa::FilterbankData& filterbank,
                  const Config& config, const DmRangeRuntime& dm_range,
                  FileTiming& timing, RawSearchResult& result,
                  ProgressTracker* progress) {
  const auto dedispersion_begin = std::chrono::steady_clock::now();
  auto dedispersed = dedisperse<T>(filterbank, config, dm_range.config);
  const auto dedispersion_end = std::chrono::steady_clock::now();
  timing.dedispersion_seconds +=
      std::chrono::duration<double>(dedispersion_end - dedispersion_begin)
          .count();
  if (progress != nullptr) {
    progress->complete_dm_range();
    progress->begin_search_phase();
  }
  const auto search_begin = dedispersion_end;

  // Run backend phases in a stable order. A worker owns one active Program,
  // so each search range is prepared, consumed, and released before the next
  // range. Native CPU is first, then Native CUDA, then Loki CUDA.
  const auto run_backend = [&](Backend backend) {
    for (const auto& search : config.search_ranges) {
      if (search.backend != backend) {
        continue;
      }
      const std::size_t prepared =
          search.backend == Backend::NativeCpu
              ? dedispersed.shape.nsamples
              : prepared_length(search, dedispersed.shape.nsamples);
      SearchRunResult run{
          .info = SearchRunInfo{
              .dm_range_id = dm_range.id,
              .search_range_id = search.id,
              .backend = backend,
              .coordinate = make_search_coordinate(
                  dedispersed.shape.nsamples, prepared,
                  filterbank.header.tsamp),
          },
      };
      const auto peak_limit = remaining_peak_limit(result.peaks, config);
      if (progress != nullptr) {
        const std::size_t total_units =
            backend == Backend::NativeCpu
                ? 1
                : dedispersed.shape.ndm / config.dm_tile_size +
                      (dedispersed.shape.ndm % config.dm_tile_size == 0 ? 0
                                                                        : 1);
        progress->begin_search_run(dm_range.id, search.id, backend,
                                   total_units);
      }
      switch (backend) {
        case Backend::NativeCpu:
          run.peaks = run_native_cpu(
              dedispersed, dm_range, search, config,
              filterbank.header.tsamp, peak_limit);
          if (progress != nullptr) {
            progress->complete_search_units();
          }
          break;
        case Backend::NativeCuda:
          {
            CudaSearchResult search_result = run_cuda_phase(
              dedispersed, dm_range, search, config,
              filterbank.header.tsamp, config.native_cuda_devices, peak_limit,
              progress);
            run.peaks = std::move(search_result.peaks);
            run.complete = search_result.complete;
            run.warnings = std::move(search_result.warnings);
          }
          break;
        case Backend::LokiCuda:
          {
            CudaSearchResult search_result = run_cuda_phase(
              dedispersed, dm_range, search, config,
              filterbank.header.tsamp, config.loki_cuda_devices, peak_limit,
              progress);
            run.peaks = std::move(search_result.peaks);
            run.complete = search_result.complete;
            run.warnings = std::move(search_result.warnings);
          }
          break;
      }
      run.info.raw_peak_count = run.peaks.size();
      run.info.complete = run.complete;
      run.info.warnings = run.warnings;
      if (progress != nullptr) {
        progress->finish_search_run(run.info.raw_peak_count);
      }
      append_checked(result.peaks, std::move(run.peaks), config);
      result.complete &= run.complete;
      result.warnings.insert(result.warnings.end(), run.warnings.begin(),
                             run.warnings.end());
      result.runs.push_back(std::move(run.info));
    }
  };
  run_backend(Backend::NativeCpu);
  run_backend(Backend::NativeCuda);
  run_backend(Backend::LokiCuda);
  const auto search_end = std::chrono::steady_clock::now();
  timing.search_seconds +=
      std::chrono::duration<double>(search_end - search_begin).count();
}

template <typename T>
RawSearchResult run_typed(const gaffa::FilterbankData& filterbank,
                          const Config& config, FileTiming& timing,
                          ProgressTracker* progress) {
  const auto dm_ranges = make_dm_ranges(config);
  RawSearchResult result;
  result.runs.reserve(dm_ranges.size() * config.search_ranges.size());
  if (progress != nullptr) {
    progress->begin_work(
        dm_ranges.size(),
        checked_multiply(dm_ranges.size(), config.search_ranges.size(),
                         "search progress count overflow"));
  }
  for (const auto& dm_range : dm_ranges) {
    if (progress != nullptr) {
      progress->begin_dm_range(dm_range.id);
    }
    run_dm_range<T>(filterbank, config, dm_range, timing, result, progress);
  }
  sort_peaks(result.peaks);
  return result;
}

RawSearchResult run_filterbank(const gaffa::FilterbankData& filterbank,
                               const Config& config, FileTiming& timing,
                               ProgressTracker* progress) {
  return std::visit(
      [&](const auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        return run_typed<T>(filterbank, config, timing, progress);
      },
      filterbank.samples);
}

}  // namespace

FileResult search_file(const Config& config,
                       const std::filesystem::path& input,
                       ProgressTracker* progress) {
  const auto total_begin = std::chrono::steady_clock::now();
  const auto read_begin = total_begin;
  if (progress != nullptr) {
    progress->begin_read();
  }
  const gaffa::FilterbankData filterbank = gaffa::read_filterbank(input);
  const auto read_end = std::chrono::steady_clock::now();
  if (config.dedispersion_backend == DedispersionBackend::CudaSubband) {
    check_device_ids({config.dedispersion_device});
  }
  FileResult result;
  result.input = input;
  result.observation_nsamples =
      static_cast<std::size_t>(filterbank.header.nsamples);
  result.tsamp_seconds = filterbank.header.tsamp;
  result.observation_seconds = filterbank.header.tsamp *
                               static_cast<double>(result.observation_nsamples);
  result.timing.read_seconds =
      std::chrono::duration<double>(read_end - read_begin).count();
  if (progress != nullptr) {
    progress->finish_read(result.observation_nsamples,
                          result.tsamp_seconds,
                          result.observation_seconds);
  }
  RawSearchResult raw =
      run_filterbank(filterbank, config, result.timing, progress);
  const auto search_end = std::chrono::steady_clock::now();

  const auto candidate_begin = search_end;
  const double candidate_duration =
      common_valid_duration(raw.runs);
  if (progress != nullptr) {
    progress->begin_candidate(raw.peaks.size());
  }
  const auto candidates = gaffa::make_candidates_cpu(
      raw.peaks,
      make_harmonic_context(filterbank.header, candidate_duration),
      make_candidate_options(config));
  result.raw_peak_count = raw.peaks.size();
  result.candidates = std::move(candidates);
  result.search_runs = std::move(raw.runs);
  result.complete = raw.complete;
  result.warnings = std::move(raw.warnings);
  const auto candidate_end = std::chrono::steady_clock::now();
  result.timing.candidate_seconds =
      std::chrono::duration<double>(candidate_end - candidate_begin).count();
  result.timing.total_seconds =
      std::chrono::duration<double>(candidate_end - total_begin).count();
  if (progress != nullptr) {
    progress->finish_candidate(ProgressSummary{
        .elapsed_seconds = result.timing.candidate_seconds,
        .raw_peaks = result.raw_peak_count,
        .candidate_groups = result.candidates.candidate_set.candidates.size(),
        .harmonic_relations =
            result.candidates.harmonic_relations.size(),
        .final_candidates = result.candidates.selected.size(),
    });
    progress->finish_file();
  }
  return result;
}

std::vector<std::filesystem::path> discover_inputs(
    const std::filesystem::path& input) {
  std::vector<std::filesystem::path> inputs;
  if (std::filesystem::is_regular_file(input)) {
    inputs.push_back(input);
  } else if (std::filesystem::is_directory(input)) {
    for (const auto& entry : std::filesystem::directory_iterator(input)) {
      if (entry.is_regular_file() && entry.path().extension() == ".fil") {
        inputs.push_back(entry.path());
      }
    }
    std::sort(inputs.begin(), inputs.end());
    if (inputs.empty()) {
      throw std::invalid_argument(
          "input directory contains no regular .fil files");
    }
  } else {
    throw std::invalid_argument(
        "input must be a regular filterbank file or a directory");
  }
  return inputs;
}

void search_each_file(const Config& config,
                      std::span<const std::filesystem::path> inputs,
                      ProgressTracker& progress,
                      FileResultConsumer consumer) {
  if (inputs.empty()) {
    throw std::invalid_argument("search requires at least one input file");
  }
  if (!consumer) {
    throw std::invalid_argument("search file consumer must be callable");
  }

  try {
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      const auto& input = inputs[index];
      Config file_config = config;
      file_config.input = input;
      progress.begin_file(input, index + 1, inputs.size());
      consumer(search_file(file_config, input, &progress));
    }
  } catch (...) {
    progress.mark_failed();
    throw;
  }
}

}  // namespace gaffa_search
