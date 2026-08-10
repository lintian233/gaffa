#include "cuda_worker.h"

#include "debug.h"

#include "gaffa/cuda_memory.h"
#include "gaffa/ffa_cuda.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/ffa_search.h"
#include "gaffa/preprocessing.h"
#include "gaffa/preprocessing_cuda.h"
#include "gaffa/time_series_cuda.h"

#ifdef GAFFA_SEARCH_ENABLE_LOKI
#include "gaffa/loki_dm_search.h"
#include "gaffa/loki_pffa.h"
#endif

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace gaffa_search {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

std::size_t next_power_of_two(std::size_t value) {
  if (value == 0) {
    throw std::invalid_argument("search input length must be non-zero");
  }
  std::size_t result = 1;
  while (result < value) {
    if (result > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("search input length power-of-two overflow");
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

gaffa::PreprocessPlan make_preprocess_plan(const std::string& name,
                                           double tsamp,
                                           double median_seconds) {
  if (name == "none") {
    return {};
  }
  if (name == "normalise") {
    return gaffa::PreprocessPlan{
        .steps = {gaffa::PreprocessStep{
            .kind = gaffa::PreprocessStepKind::Normalise,
        }},
    };
  }
  return gaffa::make_riptide_preprocess_plan(
      tsamp,
      gaffa::RiptidePreprocessOptions{
          .running_median_width_seconds = median_seconds,
          .normalise = true,
      });
}

struct StreamOwner {
  cudaStream_t stream = nullptr;

  explicit StreamOwner(int device_id) {
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice worker");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }

  StreamOwner(const StreamOwner&) = delete;
  StreamOwner& operator=(const StreamOwner&) = delete;

  ~StreamOwner() {
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }
};

}  // namespace

struct CudaWorker::Impl {
  enum class ActiveBackend {
    None,
    Native,
    Loki,
  };

  int device_id = 0;
  std::size_t tile_capacity = 0;
  std::unique_ptr<StreamOwner> stream;
  ActiveBackend active_backend = ActiveBackend::None;

  std::size_t source_nsamples = 0;
  std::size_t prepared_nsamples = 0;
  WindowMode window_mode = WindowMode::Truncate;
  float snr_threshold = 0.0F;
  std::size_t max_peaks = 0;
  gaffa::PeakReductionOptions reduction{};

  std::unique_ptr<gaffa::CudaPreprocessProgram> preprocess;
  std::unique_ptr<gaffa::CudaFfaProgram> native;
#ifdef GAFFA_SEARCH_ENABLE_LOKI
  std::unique_ptr<gaffa::LokiPffaProgram> loki;
#endif

  gaffa::CudaDeviceBuffer<std::uint32_t> raw_input;
  gaffa::CudaDeviceBuffer<float> source_float;
  gaffa::CudaDeviceBuffer<float> prepared_input;

  Impl(int device, std::size_t capacity)
      : device_id(device), tile_capacity(capacity),
        stream(std::make_unique<StreamOwner>(device)) {
    if (capacity == 0) {
      throw std::invalid_argument("CUDA worker tile capacity must be > 0");
    }
  }

  void release_programs() {
    if (preprocess != nullptr) {
      preprocess->synchronize();
    }
    if (native != nullptr) {
      native->clear();
    }
    preprocess.reset();
    native.reset();
#ifdef GAFFA_SEARCH_ENABLE_LOKI
    loki.reset();
#endif
    raw_input = {};
    source_float = {};
    prepared_input = {};
    active_backend = ActiveBackend::None;
    source_nsamples = 0;
    prepared_nsamples = 0;
  }

  void release_programs_noexcept() noexcept {
    try {
      release_programs();
    } catch (...) {
      // Destructors must not throw. CUDA allocation destructors still release
      // their owning-device memory independently.
    }
  }

  void allocate_buffers() {
    const std::size_t source_elements =
        checked_multiply(tile_capacity, source_nsamples,
                         "CUDA worker source buffer size overflow");
    const std::size_t prepared_elements =
        checked_multiply(tile_capacity, prepared_nsamples,
                         "CUDA worker prepared buffer size overflow");
    source_float = gaffa::CudaDeviceBuffer<float>(source_elements);
    if (prepared_nsamples != source_nsamples) {
      prepared_input = gaffa::CudaDeviceBuffer<float>(prepared_elements);
    }
  }

  void prepare_common(const SearchRangeConfig& search,
                      std::size_t source_length, double tsamp,
                      float threshold, std::size_t peak_limit,
                      const std::string& preprocess_name,
                      double median_seconds) {
    release_programs();
    source_nsamples = source_length;
    window_mode = search.window_mode;
    if (search.backend == Backend::LokiCuda) {
      prepared_nsamples = search.window_mode == WindowMode::ZeroPad
                              ? next_power_of_two(source_length)
                              : floor_power_of_two(source_length);
    } else {
      prepared_nsamples = search.window_mode == WindowMode::ZeroPad
                              ? next_power_of_two(source_length)
                              : source_length;
    }
    snr_threshold = threshold;
    max_peaks = peak_limit;
    reduction = {};

    const gaffa::PreprocessPlan preprocess_plan = make_preprocess_plan(
        preprocess_name, tsamp, median_seconds);
    if (!preprocess_plan.steps.empty()) {
      preprocess = std::make_unique<gaffa::CudaPreprocessProgram>(
          preprocess_plan,
          gaffa::CudaPreprocessProgramOptions{.device_id = device_id},
          gaffa::CudaPreprocessExecutionOptions{
              .series_tile_size = tile_capacity,
              .max_nsamples = source_nsamples,
              .stream = stream->stream,
          });
    }
    allocate_buffers();
  }

  void upload_and_prepare(std::span<const std::uint32_t> tile,
                          std::size_t nseries) {
    const auto begin = detail::DebugClock::now();
    if (nseries == 0 || nseries > tile_capacity ||
        tile.size() != checked_multiply(nseries, source_nsamples,
                                        "CUDA worker input size overflow")) {
      throw std::invalid_argument("invalid uint32 CUDA worker tile");
    }
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice worker upload");
    const std::size_t raw_capacity = checked_multiply(
        tile_capacity, source_nsamples, "CUDA worker raw buffer size overflow");
    if (raw_input.size() != raw_capacity) {
      raw_input = gaffa::CudaDeviceBuffer<std::uint32_t>(raw_capacity);
    }
    const std::size_t bytes = tile.size() * sizeof(std::uint32_t);
    check_cuda(cudaMemcpyAsync(raw_input.data(), tile.data(), bytes,
                               cudaMemcpyHostToDevice, stream->stream),
               "CUDA worker uint32 H2D");
    const std::size_t active_elements = checked_multiply(
        nseries, source_nsamples, "CUDA worker active input size overflow");
    auto raw_view = static_cast<const gaffa::CudaDeviceBuffer<std::uint32_t>&>(
                         raw_input)
                        .as_span(device_id);
    raw_view.count = active_elements;
    auto float_view = source_float.as_span(device_id);
    float_view.count = active_elements;
    gaffa::convert_time_series_batch_to_float_cuda(
        raw_view, nseries, source_nsamples, float_view,
        gaffa::CudaLaunchOptions{
            .device_id = device_id,
            .threads_per_block = 256,
            .stream = stream->stream,
            .synchronize_after_call = false,
        });
    upload_and_prepare_float(source_float.data(), nseries);
    DEBUGPRINT("cuda_worker upload_prepare device=" << device_id
                                                      << " input=uint32"
                                                      << " nseries=" << nseries
                                                      << " host_seconds="
                                                      << detail::debug_seconds(begin));
  }

  void upload_and_prepare(std::span<const float> tile, std::size_t nseries) {
    const auto begin = detail::DebugClock::now();
    if (nseries == 0 || nseries > tile_capacity ||
        tile.size() != checked_multiply(nseries, source_nsamples,
                                        "CUDA worker input size overflow")) {
      throw std::invalid_argument("invalid float CUDA worker tile");
    }
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice worker upload");
    check_cuda(cudaMemcpyAsync(source_float.data(), tile.data(),
                               tile.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream->stream),
               "CUDA worker float H2D");
    upload_and_prepare_float(source_float.data(), nseries);
    DEBUGPRINT("cuda_worker upload_prepare device=" << device_id
                                                      << " input=float"
                                                      << " nseries=" << nseries
                                                      << " host_seconds="
                                                      << detail::debug_seconds(begin));
  }

  void upload_and_prepare_float(float* source, std::size_t nseries) {
    if (preprocess != nullptr) {
      const auto preprocess_begin = detail::DebugClock::now();
      preprocess_time_series_batch_inplace_cuda(
          *preprocess,
          gaffa::MutableCudaTimeSeriesBatchView{
              .data = source,
              .nseries = nseries,
              .nsamples = source_nsamples,
              .device_id = device_id,
          });
      preprocess->synchronize();
      DEBUGPRINT("cuda_worker preprocess_wait device=" << device_id
                                                         << " nseries="
                                                         << nseries
                                                         << " host_seconds="
                                                         << detail::debug_seconds(preprocess_begin));
    }

    float* prepared = source;
    if (prepared_nsamples != source_nsamples) {
      const auto window_begin = detail::DebugClock::now();
      check_cuda(cudaSetDevice(device_id), "cudaSetDevice worker window");
      const std::size_t copied =
          window_mode == WindowMode::ZeroPad
              ? source_nsamples
              : std::min(source_nsamples, prepared_nsamples);
      if (window_mode == WindowMode::ZeroPad) {
        check_cuda(cudaMemsetAsync(
                       prepared_input.data(), 0, prepared_input.bytes(),
                       stream->stream),
                   "CUDA worker zero padding");
      }
      check_cuda(cudaMemcpy2DAsync(
                     prepared_input.data(),
                     prepared_nsamples * sizeof(float), source,
                     source_nsamples * sizeof(float), copied * sizeof(float),
                     nseries, cudaMemcpyDeviceToDevice, stream->stream),
                 "CUDA worker window copy");
      prepared = prepared_input.data();
      DEBUGPRINT("cuda_worker window_enqueue device=" << device_id
                                                        << " source_nsamples="
                                                        << source_nsamples
                                                        << " prepared_nsamples="
                                                        << prepared_nsamples
                                                        << " nseries=" << nseries
                                                        << " host_seconds="
                                                        << detail::debug_seconds(window_begin));
    }

    active_prepared = gaffa::CudaTimeSeriesBatchView{
        .data = prepared,
        .nseries = nseries,
        .nsamples = prepared_nsamples,
        .device_id = device_id,
    };
  }

  gaffa::CudaTimeSeriesBatchView active_prepared{};

  gaffa::DmPeaks attach(gaffa::SeriesPeaks peaks,
                        std::span<const double> dms,
                        std::size_t global_dm_index_begin) const {
    return gaffa::attach_dm_trials(
        peaks,
        gaffa::DmTrialView{
            .values = dms,
            .index_offset = global_dm_index_begin,
        });
  }

  template <typename T>
  CudaSearchResult run_native_tile(
      std::span<const T> tile, std::size_t nseries,
      std::size_t input_nsamples, std::span<const double> dms,
      std::size_t global_dm_index_begin) {
    if (active_backend != ActiveBackend::Native || native == nullptr ||
        source_nsamples != input_nsamples) {
      throw std::logic_error("CUDA worker is not prepared for Native search");
    }
    const auto tile_begin = detail::DebugClock::now();
    upload_and_prepare(tile, nseries);
    const auto ffa_begin = detail::DebugClock::now();
    const gaffa::FfaBatchSearchResult raw = gaffa::search_ffa_raw_batch_cuda(
        *native, active_prepared,
        gaffa::FfaSearchOptions{
            .snr_threshold = snr_threshold,
            .max_peaks = max_peaks,
        });
    const double ffa_host_seconds = detail::debug_seconds(ffa_begin);
    const std::size_t raw_peak_count = raw.peaks.size();
    const auto projection_begin = detail::DebugClock::now();
    gaffa::SeriesPeaks peaks;
    peaks.reserve(raw.peaks.size());
    const auto& observation = native->execution_plan().observation();
    for (const gaffa::FfaBatchPeak& peak : raw.peaks) {
      peaks.push_back(gaffa::SeriesPeak{
          .series_index = peak.series_index,
          .peak = gaffa::periodic_peak_from_ffa(peak.peak, observation),
      });
    }
    const double projection_seconds = detail::debug_seconds(projection_begin);
    const auto attach_begin = detail::DebugClock::now();
    CudaSearchResult result{
        .peaks = attach(peaks, dms, global_dm_index_begin),
        .complete = raw.complete,
        .warnings = std::move(raw.warnings),
    };
    DEBUGPRINT("cuda_worker native_tile device=" << device_id
                                                  << " nseries=" << nseries
                                                  << " raw_peaks=" << raw_peak_count
                                                  << " ffa_host_seconds="
                                                  << ffa_host_seconds
                                                  << " projection_seconds="
                                                  << projection_seconds
                                                  << " attach_seconds="
                                                  << detail::debug_seconds(attach_begin)
                                                  << " total_host_seconds="
                                                  << detail::debug_seconds(tile_begin));
    return result;
  }

#ifdef GAFFA_SEARCH_ENABLE_LOKI
  template <typename T>
  CudaSearchResult run_loki_tile(
      std::span<const T> tile, std::size_t nseries,
      std::size_t input_nsamples, std::span<const double> dms,
      std::size_t global_dm_index_begin) {
    if (active_backend != ActiveBackend::Loki || loki == nullptr ||
        source_nsamples != input_nsamples) {
      throw std::logic_error("CUDA worker is not prepared for Loki search");
    }
    const auto tile_begin = detail::DebugClock::now();
    upload_and_prepare(tile, nseries);
    const auto loki_begin = detail::DebugClock::now();
    const gaffa::SeriesPeaks peaks = loki->search_batch(
        active_prepared,
        gaffa::LokiPffaExecutionOptions{
            .stream = stream->stream,
            .max_peaks_per_series =
                max_peaks == 0 ? std::numeric_limits<std::size_t>::max()
                               : max_peaks,
            .reduction = reduction,
        });
    const double loki_host_seconds = detail::debug_seconds(loki_begin);
    const auto& diagnostics = loki->last_search_diagnostics();
    const auto attach_begin = detail::DebugClock::now();
    CudaSearchResult result{
        .peaks = attach(peaks, dms, global_dm_index_begin),
        .complete = diagnostics.complete,
        .warnings = diagnostics.warnings,
    };
    DEBUGPRINT("cuda_worker loki_tile device=" << device_id
                                                << " nseries=" << nseries
                                                << " peaks=" << result.peaks.size()
                                                << " loki_host_seconds="
                                                << loki_host_seconds
                                                << " attach_seconds="
                                                << detail::debug_seconds(attach_begin)
                                                << " total_host_seconds="
                                                << detail::debug_seconds(tile_begin));
    return result;
  }
#endif
};

CudaWorker::CudaWorker(int device_id, std::size_t tile_capacity)
    : impl_(std::make_unique<Impl>(device_id, tile_capacity)) {}

CudaWorker::~CudaWorker() {
  if (impl_ != nullptr) {
    impl_->release_programs_noexcept();
  }
}

CudaWorker::CudaWorker(CudaWorker&&) noexcept = default;
CudaWorker& CudaWorker::operator=(CudaWorker&&) noexcept = default;

int CudaWorker::device_id() const noexcept { return impl_->device_id; }

void CudaWorker::prepare_native(const SearchRangeConfig& search,
                                std::size_t source_nsamples, double tsamp,
                                float threshold, std::size_t peak_limit,
                                const std::string& preprocess_name,
                                double median_seconds,
                                std::size_t max_peak_buffer_bytes,
                                const gaffa::PeakReductionOptions& reduction) {
  const auto begin = detail::DebugClock::now();
  impl_->prepare_common(search, source_nsamples, tsamp, threshold,
                        peak_limit, preprocess_name, median_seconds);
  impl_->reduction = reduction;
  const double min_period =
      tsamp * static_cast<double>(search.bins_min);
  if (search.period_min < min_period) {
    throw std::invalid_argument(
        "search period_min is smaller than tsamp * bins_min");
  }
  const auto plan = gaffa::make_riptide_ffa_plan(
      impl_->prepared_nsamples, tsamp,
      gaffa::RiptideFfaPlanOptions{
          .period_min = search.period_min,
          .period_max = search.period_max,
          .bins_min = search.bins_min,
          .bins_max = search.bins_max,
          .duty_cycle_max = search.duty_cycle_max,
          .width_trial_spacing = search.width_trial_spacing,
      });
  impl_->native = std::make_unique<gaffa::CudaFfaProgram>(
      plan, gaffa::CudaFfaProgramOptions{.device_id = impl_->device_id},
      gaffa::CudaFfaExecutionOptions{
          .series_tile_size = impl_->tile_capacity,
          .initial_peak_buffer_bytes = std::min<std::size_t>(
              64ULL * 1024ULL * 1024ULL, max_peak_buffer_bytes),
          .max_peak_buffer_bytes = max_peak_buffer_bytes,
          .reduction = reduction,
          .stream = impl_->stream->stream,
      });
  impl_->active_backend = Impl::ActiveBackend::Native;
  DEBUGPRINT("cuda_worker prepare device=" << impl_->device_id
                                             << " backend=native-cuda"
                                             << " source_nsamples="
                                             << source_nsamples
                                             << " prepared_nsamples="
                                             << impl_->prepared_nsamples
                                             << " host_seconds="
                                             << detail::debug_seconds(begin));
}

CudaSearchResult CudaWorker::run_native(
    std::span<const std::uint32_t> tile, std::size_t nseries,
    std::size_t source_nsamples, std::span<const double> dms,
    std::size_t global_dm_index_begin) {
  return impl_->run_native_tile(tile, nseries, source_nsamples, dms,
                                global_dm_index_begin);
}

CudaSearchResult CudaWorker::run_native(
    std::span<const float> tile, std::size_t nseries,
    std::size_t source_nsamples, std::span<const double> dms,
    std::size_t global_dm_index_begin) {
  return impl_->run_native_tile(tile, nseries, source_nsamples, dms,
                                global_dm_index_begin);
}

#ifdef GAFFA_SEARCH_ENABLE_LOKI
void CudaWorker::prepare_loki(const SearchRangeConfig& search,
                              std::size_t source_nsamples, double tsamp,
                              float threshold, std::size_t peak_limit,
                              const std::string& preprocess_name,
                              double median_seconds,
                              const gaffa::PeakReductionOptions& reduction) {
  const auto begin = detail::DebugClock::now();
  impl_->prepare_common(search, source_nsamples, tsamp, threshold,
                        peak_limit, preprocess_name, median_seconds);
  impl_->reduction = reduction;
  const auto plan = gaffa::make_loki_pffa_plan(
      impl_->prepared_nsamples, tsamp,
      gaffa::LokiTaylorSearchSpace{
          .frequency_hz = {
              .minimum = 1.0 / search.period_max,
              .maximum = 1.0 / search.period_min,
          },
          .acceleration_m_per_s2 = search.motion.accel,
          .jerk_m_per_s3 = search.motion.jerk,
      },
      gaffa::LokiPffaPlanOptions{
          .phase_bins_min = search.bins_min,
          .phase_bins_max = search.bins_max,
          .duty_cycle_max = search.duty_cycle_max,
          .width_spacing = search.width_trial_spacing,
          .snr_threshold = threshold,
      });
  impl_->loki = std::make_unique<gaffa::LokiPffaProgram>(
      plan, gaffa::LokiPffaProgramOptions{.device_id = impl_->device_id});
  impl_->active_backend = Impl::ActiveBackend::Loki;
  DEBUGPRINT("cuda_worker prepare device=" << impl_->device_id
                                             << " backend=loki-cuda"
                                             << " source_nsamples="
                                             << source_nsamples
                                             << " prepared_nsamples="
                                             << impl_->prepared_nsamples
                                             << " host_seconds="
                                             << detail::debug_seconds(begin));
}

CudaSearchResult CudaWorker::run_loki(
    std::span<const std::uint32_t> tile, std::size_t nseries,
    std::size_t source_nsamples, std::span<const double> dms,
    std::size_t global_dm_index_begin) {
  return impl_->run_loki_tile(tile, nseries, source_nsamples, dms,
                              global_dm_index_begin);
}

CudaSearchResult CudaWorker::run_loki(
    std::span<const float> tile, std::size_t nseries,
    std::size_t source_nsamples, std::span<const double> dms,
    std::size_t global_dm_index_begin) {
  return impl_->run_loki_tile(tile, nseries, source_nsamples, dms,
                              global_dm_index_begin);
}
#endif

void CudaWorker::reset() { impl_->release_programs(); }

}  // namespace gaffa_search
