#include "gaffa/loki_pffa.h"

#include <loki/algorithms/ffa.hpp>
#include <loki/algorithms/regions.hpp>
#include <loki/common/plans.hpp>
#include <loki/detection/score.hpp>
#include <loki/search/configs.hpp>
#include <loki/utils/workspace.hpp>

#include <cub/cub.cuh>

#include <cuda/std/span>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace gaffa {
namespace {

constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kPlannerSafetyBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaxPlannerAttempts = 8;

__global__ void fill_unit_variance_kernel(float* output, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = 1.0F;
  }
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void check_cuda_driver(CUresult status, const char* operation) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* message = nullptr;
  (void)cuGetErrorString(status, &message);
  throw std::runtime_error(std::string(operation) + ": " +
                           (message != nullptr
                                ? message
                                : "unknown CUDA driver error"));
}

int cuda_stream_device(cudaStream_t stream) {
  CUcontext stream_context = nullptr;
  check_cuda_driver(cuStreamGetCtx(reinterpret_cast<CUstream>(stream),
                                   &stream_context),
                    "cuStreamGetCtx");
  check_cuda_driver(cuCtxPushCurrent(stream_context), "cuCtxPushCurrent");

  CUdevice stream_device = -1;
  const CUresult device_status = cuCtxGetDevice(&stream_device);
  CUcontext popped_context = nullptr;
  const CUresult pop_status = cuCtxPopCurrent(&popped_context);
  check_cuda_driver(pop_status, "cuCtxPopCurrent");
  if (popped_context != stream_context) {
    throw std::runtime_error("CUDA context stack changed while querying Loki stream");
  }
  check_cuda_driver(device_status, "cuCtxGetDevice");
  return static_cast<int>(stream_device);
}

class DeviceGuard final {
 public:
  explicit DeviceGuard(int device_id) {
    check_cuda(cudaGetDevice(&previous_device_), "cudaGetDevice");
    if (previous_device_ != device_id) {
      check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_device_ = 0;
  bool restore_ = false;
};

class ActiveSearchGuard final {
 public:
  explicit ActiveSearchGuard(std::atomic_flag& active) : active_(active) {
    if (active_.test_and_set(std::memory_order_acquire)) {
      throw std::logic_error(
          "LokiPffaProgram does not permit concurrent search() calls");
    }
  }

  ~ActiveSearchGuard() {
    active_.clear(std::memory_order_release);
  }

  ActiveSearchGuard(const ActiveSearchGuard&) = delete;
  ActiveSearchGuard& operator=(const ActiveSearchGuard&) = delete;

 private:
  std::atomic_flag& active_;
};

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs,
                        const char* message) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::overflow_error(message);
  }
  return lhs + rhs;
}

unsigned int reduction_grid_blocks(std::size_t count, const char* operation) {
  const std::size_t blocks = count / 256U + (count % 256U != 0 ? 1U : 0U);
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error(std::string(operation) + " grid overflow");
  }
  return static_cast<unsigned int>(blocks);
}

std::vector<loki::ParamLimit> make_loki_limits(
    const LokiTaylorSearchSpace& search_space) {
  std::vector<loki::ParamLimit> limits;
  if (search_space.snap_m_per_s4.has_value()) {
    limits.push_back(
        {.min = search_space.snap_m_per_s4->minimum,
         .max = search_space.snap_m_per_s4->maximum});
  }
  if (search_space.jerk_m_per_s3.has_value()) {
    limits.push_back(
        {.min = search_space.jerk_m_per_s3->minimum,
         .max = search_space.jerk_m_per_s3->maximum});
  }
  if (search_space.acceleration_m_per_s2.has_value()) {
    limits.push_back({.min = search_space.acceleration_m_per_s2->minimum,
                      .max = search_space.acceleration_m_per_s2->maximum});
  }
  limits.push_back({.min = search_space.frequency_hz.minimum,
                    .max = search_space.frequency_hz.maximum});
  return limits;
}

double parameter_value(const loki::ParamLimit& limit, std::size_t count,
                       std::size_t index) {
  if (count <= 1) {
    return (limit.min + limit.max) / 2.0;
  }
  return limit.min + ((limit.max - limit.min) /
                      static_cast<double>(count)) *
                         (static_cast<double>(index) + 0.5);
}

struct LokiPeakRef {
  float snr = 0.0F;
  std::uint32_t index = 0;
};

static_assert(sizeof(LokiPeakRef) == 8);

struct LokiReductionSummary {
  std::size_t count = 0;
  bool overflow = false;
};

__global__ void build_loki_reduction_refs_kernel(
    const float* scores, const std::uint32_t* indices, std::size_t count,
    LokiPeakRef* refs) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    refs[index] = LokiPeakRef{
        .snr = scores[index],
        .index = indices[index],
    };
  }
}

__global__ void build_loki_reduction_group_keys_kernel(
    const LokiPeakRef* refs, std::size_t count, std::size_t width_count,
    std::uint32_t* keys) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    keys[index] = static_cast<std::uint32_t>(refs[index].index / width_count);
  }
}

__global__ void select_loki_reduction_peaks_kernel(
    const std::uint32_t* sorted_keys, const LokiPeakRef* sorted_refs,
    std::size_t count, std::size_t top_k, std::size_t max_groups,
    std::size_t output_capacity, std::uint32_t* group_count,
    unsigned long long* output_count, unsigned int* overflow,
    LokiPeakRef* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count ||
      (index != 0 && sorted_keys[index] == sorted_keys[index - 1])) {
    return;
  }
  const std::size_t group = atomicAdd(group_count, 1U);
  if (group >= max_groups) {
    atomicExch(overflow, 1U);
    return;
  }
  const std::uint32_t key = sorted_keys[index];
  for (std::size_t rank = 0; rank < top_k && index + rank < count; ++rank) {
    if (sorted_keys[index + rank] != key) {
      break;
    }
    const unsigned long long output_index = atomicAdd(output_count, 1ULL);
    if (output_index >= output_capacity) {
      atomicExch(overflow, 1U);
      continue;
    }
    output[output_index] = sorted_refs[index + rank];
  }
}

std::size_t query_loki_reduction_sort_bytes(std::size_t capacity) {
  std::size_t score_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(
      nullptr, score_bytes, static_cast<const float*>(nullptr),
      static_cast<float*>(nullptr), static_cast<const LokiPeakRef*>(nullptr),
      static_cast<LokiPeakRef*>(nullptr), capacity);
  std::size_t group_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, group_bytes, static_cast<const std::uint32_t*>(nullptr),
      static_cast<std::uint32_t*>(nullptr),
      static_cast<const LokiPeakRef*>(nullptr),
      static_cast<LokiPeakRef*>(nullptr), capacity);
  return std::max(score_bytes, group_bytes);
}

class LokiReductionWorkspace {
 public:
  LokiReductionWorkspace() = default;

  LokiReductionWorkspace(const PeakReductionOptions& options,
                         std::size_t input_capacity)
      : top_k_(options.top_k_per_group),
        max_groups_(options.max_groups_per_series),
        input_capacity_(input_capacity) {
    if (top_k_ == 0) {
      return;
    }
    if (max_groups_ == 0) {
      throw std::invalid_argument(
          "Loki peak reduction max_groups_per_series must be > 0");
    }
    if (max_groups_ > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error(
          "Loki peak reduction max_groups_per_series exceeds uint32_t");
    }
    allocate();
  }

  LokiReductionWorkspace(const LokiReductionWorkspace&) = delete;
  LokiReductionWorkspace& operator=(const LokiReductionWorkspace&) = delete;
  LokiReductionWorkspace(LokiReductionWorkspace&&) noexcept = default;
  LokiReductionWorkspace& operator=(LokiReductionWorkspace&&) noexcept = default;

  [[nodiscard]] bool enabled() const noexcept { return top_k_ != 0; }

  [[nodiscard]] bool matches(const PeakReductionOptions& options,
                             std::size_t capacity) const noexcept {
    return top_k_ == options.top_k_per_group &&
           max_groups_ == options.max_groups_per_series &&
           input_capacity_ >= capacity;
  }

  [[nodiscard]] std::size_t bytes() const noexcept {
    return refs_a_.bytes() + refs_b_.bytes() + sorted_scores_.bytes() +
           keys_a_.bytes() + keys_b_.bytes() + group_count_.bytes() +
           output_.bytes() + output_count_.bytes() + overflow_.bytes() +
           temp_.bytes();
  }

  static std::size_t estimate_bytes(const PeakReductionOptions& options,
                                    std::size_t input_capacity) {
    if (!options.enabled()) {
      return 0;
    }
    if (options.max_groups_per_series == 0 || input_capacity == 0) {
      throw std::invalid_argument(
          "Loki peak reduction requires a non-empty input capacity and "
          "max_groups_per_series > 0");
    }
    if (options.max_groups_per_series >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error(
          "Loki peak reduction max_groups_per_series exceeds uint32_t");
    }

    std::size_t bytes = 0;
    const auto add_array = [&bytes](std::size_t count,
                                    std::size_t element_size,
                                    const char* message) {
      bytes = checked_add(
          bytes, checked_multiply(count, element_size, message),
          "Loki reduction workspace byte size overflow");
    };
    add_array(input_capacity, sizeof(LokiPeakRef),
              "Loki reduction reference byte size overflow");
    add_array(input_capacity, sizeof(LokiPeakRef),
              "Loki reduction reference byte size overflow");
    add_array(input_capacity, sizeof(float),
              "Loki reduction score byte size overflow");
    add_array(input_capacity, sizeof(std::uint32_t),
              "Loki reduction group key byte size overflow");
    add_array(input_capacity, sizeof(std::uint32_t),
              "Loki reduction group key byte size overflow");
    add_array(1, sizeof(std::uint32_t),
              "Loki reduction group counter byte size overflow");
    const std::size_t output_count = checked_multiply(
        options.max_groups_per_series, options.top_k_per_group,
        "Loki reduction output size overflow");
    add_array(output_count, sizeof(LokiPeakRef),
              "Loki reduction output byte size overflow");
    add_array(1, sizeof(unsigned long long),
              "Loki reduction output counter byte size overflow");
    add_array(1, sizeof(unsigned int),
              "Loki reduction overflow byte size overflow");
    bytes = checked_add(
        bytes, query_loki_reduction_sort_bytes(input_capacity),
        "Loki reduction workspace byte size overflow");
    return bytes;
  }

  [[nodiscard]] const CudaDeviceBuffer<LokiPeakRef>& output() const noexcept {
    return output_;
  }

  LokiReductionSummary reduce(const CudaDeviceBuffer<float>& scores,
                              const CudaDeviceBuffer<std::uint32_t>& indices,
                              std::size_t count, std::size_t width_count,
                              cudaStream_t stream) {
    if (!enabled() || count > input_capacity_ || width_count == 0) {
      throw std::invalid_argument("invalid Loki reduction input");
    }
    if (count == 0) {
      return {};
    }
    check_cuda(cudaMemsetAsync(group_count_.data(), 0,
                               group_count_.bytes(), stream),
               "Loki reduction group counter reset");
    check_cuda(cudaMemsetAsync(output_count_.data(), 0,
                               output_count_.bytes(), stream),
               "Loki reduction output counter reset");
    check_cuda(cudaMemsetAsync(overflow_.data(), 0, overflow_.bytes(), stream),
               "Loki reduction overflow reset");

    const unsigned int blocks =
        reduction_grid_blocks(count, "Loki reduction");
    build_loki_reduction_refs_kernel<<<blocks, 256, 0, stream>>>(
        scores.data(), indices.data(), count, refs_a_.data());
    check_cuda(cudaGetLastError(), "Loki reduction ref launch");

    std::size_t temp_bytes = temp_.bytes();
    cub::DeviceRadixSort::SortPairsDescending(
        temp_.data(), temp_bytes, scores.data(), sorted_scores_.data(),
        refs_a_.data(), refs_b_.data(), count, 0, sizeof(float) * 8, stream);
    build_loki_reduction_group_keys_kernel<<<blocks, 256, 0, stream>>>(
        refs_b_.data(), count, width_count, keys_a_.data());
    check_cuda(cudaGetLastError(), "Loki reduction key launch");
    cub::DeviceRadixSort::SortPairs(
        temp_.data(), temp_bytes, keys_a_.data(), keys_b_.data(),
        refs_b_.data(), refs_a_.data(), count, 0, sizeof(std::uint32_t) * 8,
        stream);
    select_loki_reduction_peaks_kernel<<<blocks, 256, 0, stream>>>(
        keys_b_.data(), refs_a_.data(), count, top_k_, max_groups_,
        output_.size(), group_count_.data(), output_count_.data(),
        overflow_.data(), output_.data());
    check_cuda(cudaGetLastError(), "Loki reduction select launch");
    check_cuda(cudaStreamSynchronize(stream), "Loki reduction synchronize");

    unsigned long long output_count = 0;
    unsigned int overflow = 0;
    check_cuda(cudaMemcpy(&output_count, output_count_.data(),
                          sizeof(output_count), cudaMemcpyDeviceToHost),
               "Loki reduction count D2H");
    check_cuda(cudaMemcpy(&overflow, overflow_.data(), sizeof(overflow),
                          cudaMemcpyDeviceToHost),
               "Loki reduction overflow D2H");
    if (output_count > output_.size()) {
      throw std::logic_error("Loki reduction output count is invalid");
    }
    return LokiReductionSummary{
        .count = static_cast<std::size_t>(output_count),
        .overflow = overflow != 0,
    };
  }

 private:
  void allocate() {
    refs_a_ = CudaDeviceBuffer<LokiPeakRef>(input_capacity_);
    refs_b_ = CudaDeviceBuffer<LokiPeakRef>(input_capacity_);
    sorted_scores_ = CudaDeviceBuffer<float>(input_capacity_);
    keys_a_ = CudaDeviceBuffer<std::uint32_t>(input_capacity_);
    keys_b_ = CudaDeviceBuffer<std::uint32_t>(input_capacity_);
    group_count_ = CudaDeviceBuffer<std::uint32_t>(1);
    output_ = CudaDeviceBuffer<LokiPeakRef>(checked_multiply(
        max_groups_, top_k_, "Loki reduction output size overflow"));
    output_count_ = CudaDeviceBuffer<unsigned long long>(1);
    overflow_ = CudaDeviceBuffer<unsigned int>(1);
    temp_ = CudaDeviceMemory(query_loki_reduction_sort_bytes(input_capacity_));
  }

  std::size_t top_k_ = 0;
  std::size_t max_groups_ = 0;
  std::size_t input_capacity_ = 0;
  CudaDeviceBuffer<LokiPeakRef> refs_a_;
  CudaDeviceBuffer<LokiPeakRef> refs_b_;
  CudaDeviceBuffer<float> sorted_scores_;
  CudaDeviceBuffer<std::uint32_t> keys_a_;
  CudaDeviceBuffer<std::uint32_t> keys_b_;
  CudaDeviceBuffer<std::uint32_t> group_count_;
  CudaDeviceBuffer<LokiPeakRef> output_;
  CudaDeviceBuffer<unsigned long long> output_count_;
  CudaDeviceBuffer<unsigned int> overflow_;
  CudaDeviceMemory temp_;
};

}  // namespace

struct LokiPffaProgram::Impl {
  struct Region {
    loki::search::PulsarSearchConfig config;
    std::unique_ptr<loki::plans::FFAPlan<float>> plan;
    std::vector<std::uint32_t> widths;
    std::size_t score_count = 0;
    std::size_t transient_bytes = 0;

    Region(loki::search::PulsarSearchConfig config_in,
           std::unique_ptr<loki::plans::FFAPlan<float>> plan_in,
           std::vector<std::uint32_t> widths_in, std::size_t score_count_in,
           std::size_t transient_bytes_in)
        : config(std::move(config_in)),
          plan(std::move(plan_in)),
          widths(std::move(widths_in)),
          score_count(score_count_in),
          transient_bytes(transient_bytes_in) {}
  };

  struct Layout {
    std::vector<Region> regions;
    std::size_t workspace_buffer_elements = 0;
    std::size_t workspace_coordinate_elements = 0;
    std::size_t workspace_levels = 0;
    std::size_t parameter_count = 0;
    std::size_t fold_elements = 0;
    std::size_t score_elements = 0;
    std::size_t width_elements = 0;
    std::size_t transient_peak_bytes = 0;
    std::size_t persistent_bytes = 0;
  };

  explicit Impl(LokiPffaPlan plan_in, LokiPffaProgramOptions options_in)
      : plan(std::move(plan_in)), options(options_in) {}

  [[nodiscard]] std::size_t resolve_budget_bytes() const;
  [[nodiscard]] Layout make_layout(std::size_t budget_bytes) const;
  [[nodiscard]] Layout make_layout_once(std::size_t planner_budget_bytes) const;
  [[nodiscard]] static loki::search::PulsarSearchConfig make_base_config(
      const LokiPffaPlan& plan, std::size_t budget_bytes);
  void initialize(cudaStream_t stream);
  [[nodiscard]] PeriodicPeak make_peak(const Region& region, float snr,
                                       std::uint32_t flat_index) const;

  ~Impl() {
    if (options.device_id < 0) {
      return;
    }
    try {
      DeviceGuard guard(options.device_id);
      counter.reset();
      reduction.reset();
      workspace.reset();
    } catch (...) {
      // Destructors cannot report CUDA/Loki cleanup failures.
    }
  }

  LokiPffaPlan plan;
  LokiPffaProgramOptions options;
  std::optional<Layout> layout;
  std::unique_ptr<loki::memory::FFAWorkspaceCUDA<float>> workspace;
  std::unique_ptr<loki::memory::DeviceCounter> counter;
  CudaDeviceBuffer<float> unit_variance;
  CudaDeviceBuffer<float> fold;
  CudaDeviceBuffer<float> scores;
  CudaDeviceBuffer<std::uint32_t> indices;
  CudaDeviceBuffer<std::uint32_t> widths;
  std::unique_ptr<LokiReductionWorkspace> reduction;
  LokiPffaSearchDiagnostics diagnostics;
  std::size_t resolved_budget_bytes = 0;
  std::atomic_flag search_active = ATOMIC_FLAG_INIT;
};

loki::search::PulsarSearchConfig LokiPffaProgram::Impl::make_base_config(
    const LokiPffaPlan& plan, std::size_t budget_bytes) {
  const auto& options = plan.options();
  const std::vector<loki::ParamLimit> limits = make_loki_limits(plan.search_space());
  return loki::search::PulsarSearchConfig(
      plan.input_nsamples(), plan.tsamp_seconds(), options.phase_bins_min,
      options.eta, limits, options.duty_cycle_max, options.width_spacing,
      false, 1, static_cast<double>(budget_bytes) / static_cast<double>(kGiB),
      2.0, options.phase_bins_max, options.phase_bins_min, std::nullopt,
      plan.input_nsamples(), options.snr_threshold, 1);
}

std::size_t LokiPffaProgram::Impl::resolve_budget_bytes() const {
  if (options.memory_budget_bytes != 0) {
    return options.memory_budget_bytes;
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  const std::size_t signal_bytes = checked_multiply(
      plan.input_nsamples(), sizeof(float), "Loki input size overflow");
  const std::size_t available = checked_add(
      free_bytes, signal_bytes, "Loki automatic memory budget overflow");
  if (available <= options.memory_reserve_bytes) {
    throw std::runtime_error("insufficient free device memory after Loki reserve");
  }
  return available - options.memory_reserve_bytes;
}

LokiPffaProgram::Impl::Layout LokiPffaProgram::Impl::make_layout_once(
    std::size_t planner_budget_bytes) const {
  const auto base = make_base_config(plan, planner_budget_bytes);
  loki::regions::FFARegionPlanner<float> planner(base, true);
  LokiPffaProgram::Impl::Layout layout;
  const auto& stats = planner.get_stats();
  layout.workspace_buffer_elements = stats.get_max_buffer_size();
  layout.workspace_coordinate_elements = stats.get_max_coord_size();
  layout.workspace_levels = stats.get_max_ffa_levels();
  layout.parameter_count = base.get_nparams();

  for (const auto& config : planner.get_cfgs()) {
    auto plan = std::make_unique<loki::plans::FFAPlan<float>>(config);
    if (plan->get_nsegments().empty() || plan->get_nsegments().back() != 1U) {
      throw std::runtime_error(
          "Loki P-FFA CUDA scoring requires one final FFA segment");
    }
    std::vector<std::uint32_t> region_widths;
    for (const auto width : config.get_scoring_widths()) {
      if (width > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Loki boxcar width exceeds uint32_t");
      }
      region_widths.push_back(static_cast<std::uint32_t>(width));
    }
    if (region_widths.empty()) {
      throw std::runtime_error("Loki P-FFA region has no boxcar width trials");
    }
    const std::size_t ncoords = plan->get_ncoords().back();
    const std::size_t score_count = checked_multiply(
        ncoords, region_widths.size(), "Loki score count overflow");
    const auto frequency_grid = plan->compute_param_grid(0).back();
    const std::size_t phase_elements = checked_multiply(
        frequency_grid.size(), plan->get_segment_lens().front(),
        "Loki phase map size overflow");
    const std::size_t phase_map_bytes = checked_multiply(
        phase_elements, sizeof(std::uint32_t), "Loki phase map byte overflow");
    const std::size_t frequency_bytes = checked_multiply(
        frequency_grid.size(), sizeof(double), "Loki frequency grid byte overflow");
    const std::size_t transient_bytes = checked_add(
        phase_map_bytes, frequency_bytes, "Loki transient size overflow");

    layout.fold_elements = std::max(layout.fold_elements,
                                    static_cast<std::size_t>(plan->get_buffer_size()));
    layout.score_elements = std::max(layout.score_elements, score_count);
    layout.width_elements = std::max(layout.width_elements, region_widths.size());
    layout.transient_peak_bytes = std::max(layout.transient_peak_bytes, transient_bytes);
    layout.regions.emplace_back(config, std::move(plan), std::move(region_widths),
                                score_count, transient_bytes);
  }
  if (layout.regions.empty()) {
    throw std::runtime_error("Loki P-FFA planner produced no execution regions");
  }
  const std::size_t direct_buffer_bytes = checked_add(
      checked_add(checked_multiply(plan.input_nsamples(), sizeof(float),
                                   "Loki variance size overflow"),
                  checked_multiply(layout.fold_elements, sizeof(float),
                                   "Loki fold size overflow"),
                  "Loki persistent size overflow"),
      checked_add(checked_multiply(layout.score_elements, sizeof(float),
                                   "Loki score size overflow"),
                  checked_add(checked_multiply(layout.score_elements, sizeof(std::uint32_t),
                                               "Loki index size overflow"),
                              checked_multiply(layout.width_elements, sizeof(std::uint32_t),
                                               "Loki width size overflow"),
                              "Loki persistent size overflow"),
                  "Loki persistent size overflow"),
      "Loki persistent size overflow");
  // Loki's workspace contains an internal fold buffer and coordinate storage
  // that are not represented by Gaffa's explicit output buffers above. Its
  // public planner estimate also includes the caller-owned signal and the
  // unit-variance series, both of which belong to this execution budget.
  const double planner_bytes_double =
      static_cast<double>(stats.get_device_memory_usage()) *
      static_cast<double>(kGiB);
  if (!std::isfinite(planner_bytes_double) || planner_bytes_double < 0.0 ||
      planner_bytes_double >
          static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("invalid Loki planner memory estimate");
  }
  const std::size_t planner_bytes =
      static_cast<std::size_t>(std::ceil(planner_bytes_double));
  const std::size_t metadata_bytes = checked_add(
      checked_multiply(layout.width_elements, sizeof(std::uint32_t),
                       "Loki width metadata size overflow"),
      sizeof(std::uint32_t), "Loki counter metadata size overflow");
  layout.persistent_bytes = std::max(
      direct_buffer_bytes,
      checked_add(planner_bytes, metadata_bytes,
                  "Loki planner persistent size overflow"));
  return layout;
}

LokiPffaProgram::Impl::Layout LokiPffaProgram::Impl::make_layout(
    std::size_t budget_bytes) const {
  std::size_t planner_budget_bytes = budget_bytes;
  for (std::size_t attempt = 0; attempt < kMaxPlannerAttempts; ++attempt) {
    Layout layout = make_layout_once(planner_budget_bytes);
    const std::size_t peak_bytes = checked_add(
        layout.persistent_bytes, layout.transient_peak_bytes,
        "Loki peak memory size overflow");
    if (peak_bytes <= budget_bytes) {
      return layout;
    }

    const std::size_t excess_bytes = peak_bytes - budget_bytes;
    const std::size_t reduction_bytes = checked_add(
        excess_bytes, kPlannerSafetyBytes, "Loki planner budget reduction overflow");
    if (reduction_bytes >= planner_budget_bytes) {
      break;
    }
    planner_budget_bytes -= reduction_bytes;
  }
  throw std::runtime_error(
      "Loki P-FFA execution layout exceeds the configured device memory budget "
      "after region splitting");
}

void LokiPffaProgram::Impl::initialize(cudaStream_t stream) {
  if (layout.has_value()) {
    return;
  }
  const std::size_t budget = resolve_budget_bytes();
  auto new_layout = make_layout(budget);

  auto workspace = std::make_unique<loki::memory::FFAWorkspaceCUDA<float>>(
      new_layout.workspace_buffer_elements,
      new_layout.workspace_coordinate_elements, new_layout.workspace_levels,
      new_layout.parameter_count);
  auto counter = std::make_unique<loki::memory::DeviceCounter>();
  CudaDeviceBuffer<float> unit_variance(plan.input_nsamples());
  CudaDeviceBuffer<float> fold(new_layout.fold_elements);
  CudaDeviceBuffer<float> scores(new_layout.score_elements);
  CudaDeviceBuffer<std::uint32_t> indices(new_layout.score_elements);
  CudaDeviceBuffer<std::uint32_t> widths(new_layout.width_elements);

  constexpr unsigned int threads_per_block = 256;
  const std::size_t blocks =
      (unit_variance.size() + threads_per_block - 1) / threads_per_block;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("Loki unit variance launch grid overflow");
  }
  fill_unit_variance_kernel<<<static_cast<unsigned int>(blocks),
                              threads_per_block, 0, stream>>>(
      unit_variance.data(), unit_variance.size());
  check_cuda(cudaGetLastError(), "fill_unit_variance_kernel launch");
  check_cuda(cudaStreamSynchronize(stream),
             "fill_unit_variance_kernel synchronize");

  this->workspace = std::move(workspace);
  this->counter = std::move(counter);
  this->unit_variance = std::move(unit_variance);
  this->fold = std::move(fold);
  this->scores = std::move(scores);
  this->indices = std::move(indices);
  this->widths = std::move(widths);
  resolved_budget_bytes = budget;
  layout.emplace(std::move(new_layout));
}

PeriodicPeak LokiPffaProgram::Impl::make_peak(const Region& region,
                                              float snr,
                                              std::uint32_t flat_index) const {
  const std::size_t width_count = region.widths.size();
  const std::size_t coordinate_index = flat_index / width_count;
  const std::size_t width_index = flat_index % width_count;
  const auto& counts = region.plan->get_param_counts().back();
  const auto& strides = region.plan->get_param_cart_strides().back();
  const auto limits = region.config.get_param_limits();
  if (counts.size() != limits.size() || strides.size() != limits.size() ||
      width_index >= region.widths.size()) {
    throw std::runtime_error("invalid Loki compact candidate index metadata");
  }
  std::vector<double> values;
  values.reserve(limits.size());
  std::size_t remaining = coordinate_index;
  for (std::size_t index = 0; index < limits.size(); ++index) {
    const std::size_t parameter_index = remaining / strides[index];
    remaining -= parameter_index * strides[index];
    if (parameter_index >= counts[index]) {
      throw std::runtime_error("Loki compact candidate coordinate is out of range");
    }
    values.push_back(parameter_value(limits[index], counts[index], parameter_index));
  }

  PeriodicPeak peak{
      .motion = {
          .order = MotionOrder::Frequency,
          // Loki's direct coordinates are defined on the full plan-length
          // search window, which may include benchmark-local zero padding.
          .reference_time_seconds =
              0.5 * static_cast<double>(plan.input_nsamples()) *
              plan.tsamp_seconds(),
          .frequency_hz = values.back(),
      },
      .phase_bin = std::nullopt,
      .phase_bins = static_cast<std::size_t>(region.config.get_nbins()),
      .boxcar_width_bins = region.widths[width_index],
      .snr = snr,
  };
  peak.duty_cycle = static_cast<double>(peak.boxcar_width_bins) /
                    static_cast<double>(peak.phase_bins);
  if (values.size() >= 2) {
    peak.motion.order = MotionOrder::Acceleration;
    peak.motion.acceleration_m_per_s2 = values[values.size() - 2];
  }
  if (values.size() >= 3) {
    peak.motion.order = MotionOrder::Jerk;
    peak.motion.jerk_m_per_s3 = values[values.size() - 3];
  }
  if (values.size() >= 4) {
    peak.motion.order = MotionOrder::Snap;
    peak.motion.snap_m_per_s4 = values[values.size() - 4];
  }
  return peak;
}

LokiPffaProgram::LokiPffaProgram(LokiPffaPlan plan,
                                 LokiPffaProgramOptions options)
    : impl_(std::make_unique<Impl>(std::move(plan), options)) {
  if (options.device_id < 0) {
    throw std::invalid_argument("invalid Loki P-FFA program options");
  }
}

LokiPffaProgram::~LokiPffaProgram() = default;
LokiPffaProgram::LokiPffaProgram(LokiPffaProgram&&) noexcept = default;
LokiPffaProgram& LokiPffaProgram::operator=(LokiPffaProgram&&) noexcept = default;

int LokiPffaProgram::device_id() const noexcept {
  return impl_->options.device_id;
}

const LokiPffaPlan& LokiPffaProgram::plan() const noexcept {
  return impl_->plan;
}

const LokiPffaSearchDiagnostics& LokiPffaProgram::last_search_diagnostics()
    const noexcept {
  return impl_->diagnostics;
}

std::vector<PeriodicPeak> LokiPffaProgram::search(
    CudaSpan<const float> normalised_time_series,
    LokiPffaExecutionOptions execution_options) {
  if (normalised_time_series.data == nullptr ||
      normalised_time_series.count != impl_->plan.input_nsamples()) {
    throw std::invalid_argument(
        "Loki P-FFA input must be a non-null full plan-length device series");
  }
  if (normalised_time_series.device_id != impl_->options.device_id) {
    throw std::invalid_argument("Loki P-FFA input belongs to another CUDA device");
  }
  if (execution_options.max_peaks_per_series == 0) {
    throw std::invalid_argument("Loki P-FFA max_peaks_per_series must be > 0");
  }
  if (execution_options.reduction.enabled() &&
      execution_options.reduction.max_groups_per_series == 0) {
    throw std::invalid_argument(
        "Loki peak reduction max_groups_per_series must be > 0");
  }
  if (!std::isfinite(execution_options.reduction.frequency_tolerance_hz) ||
      execution_options.reduction.frequency_tolerance_hz < 0.0) {
    throw std::invalid_argument(
        "Loki peak reduction frequency tolerance must be finite and "
        "non-negative");
  }

  ActiveSearchGuard active_search(impl_->search_active);
  impl_->diagnostics = LokiPffaSearchDiagnostics{};
  DeviceGuard guard(impl_->options.device_id);
  if (execution_options.stream != nullptr) {
    if (cuda_stream_device(execution_options.stream) !=
        impl_->options.device_id) {
      throw std::invalid_argument("Loki P-FFA stream belongs to another CUDA device");
    }
  }
  impl_->initialize(execution_options.stream);
  if (execution_options.reduction.enabled()) {
    if (impl_->reduction == nullptr ||
        !impl_->reduction->matches(execution_options.reduction,
                                   impl_->layout->score_elements)) {
      const std::size_t base_bytes = checked_add(
          impl_->layout->persistent_bytes, impl_->layout->transient_peak_bytes,
          "Loki peak memory size overflow");
      const std::size_t reduction_bytes =
          LokiReductionWorkspace::estimate_bytes(
              execution_options.reduction, impl_->layout->score_elements);
      if (base_bytes > impl_->resolved_budget_bytes ||
          reduction_bytes > impl_->resolved_budget_bytes - base_bytes) {
        throw std::runtime_error(
            "Loki peak reduction workspace exceeds the configured device "
            "memory budget");
      }
      auto reduction = std::make_unique<LokiReductionWorkspace>(
          execution_options.reduction, impl_->layout->score_elements);
      if (reduction->bytes() >
          impl_->resolved_budget_bytes - base_bytes) {
        throw std::logic_error(
            "Loki reduction workspace estimate is smaller than allocation");
      }
      impl_->reduction = std::move(reduction);
    }
  } else {
    impl_->reduction.reset();
  }

  std::vector<PeriodicPeak> peaks;
  for (const auto& region : impl_->layout->regions) {
    check_cuda(cudaMemcpyAsync(impl_->widths.data(), region.widths.data(),
                               region.widths.size() * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, execution_options.stream),
               "cudaMemcpyAsync Loki boxcar widths");
    auto ffa = std::make_unique<loki::algorithms::FFACUDA<float>>(
        *impl_->workspace, region.config, impl_->options.device_id);
    ffa->execute(
        cuda::std::span<const float>(normalised_time_series.data,
                                     normalised_time_series.count),
        cuda::std::span<const float>(impl_->unit_variance.data(),
                                     impl_->unit_variance.size()),
        cuda::std::span<float>(impl_->fold.data(), region.plan->get_buffer_size()),
        execution_options.stream);
    const std::size_t passing = loki::detection::score_and_filter_cuda_d(
        cuda::std::span<const float>(impl_->fold.data(), region.plan->get_fold_size()),
        cuda::std::span<const std::uint32_t>(impl_->widths.data(),
                                             region.widths.size()),
        cuda::std::span<float>(impl_->scores.data(), region.score_count),
        cuda::std::span<std::uint32_t>(impl_->indices.data(), region.score_count),
        impl_->plan.options().snr_threshold,
        region.plan->get_ncoords().back(), region.config.get_nbins(),
        execution_options.stream, *impl_->counter);
    if (peaks.size() > execution_options.max_peaks_per_series ||
        passing > execution_options.max_peaks_per_series - peaks.size()) {
      throw std::runtime_error(
          "Loki P-FFA compact peak limit exceeded; raise max_peaks_per_series "
          "or increase the SNR threshold");
    }
    if (passing == 0) {
      ffa.reset();
      continue;
    }
    std::size_t selected_count = passing;
    LokiReductionSummary reduction_summary;
    if (impl_->reduction != nullptr) {
      reduction_summary = impl_->reduction->reduce(
          impl_->scores, impl_->indices, passing, region.widths.size(),
          execution_options.stream);
      selected_count = reduction_summary.count;
      if (reduction_summary.overflow) {
        impl_->diagnostics.complete = false;
        impl_->diagnostics.warnings.push_back(
            "Loki CUDA reduction exceeded max_groups_per_series; "
            "some coordinate groups were discarded");
      }
    }
    std::vector<LokiPeakRef> host_refs;
    std::vector<float> host_scores;
    std::vector<std::uint32_t> host_indices;
    if (impl_->reduction != nullptr) {
      host_refs.resize(selected_count);
      check_cuda(cudaMemcpyAsync(
                     host_refs.data(), impl_->reduction->output().data(),
                     selected_count * sizeof(LokiPeakRef),
                     cudaMemcpyDeviceToHost, execution_options.stream),
                 "cudaMemcpyAsync Loki reduced peaks");
    } else {
      host_scores.resize(selected_count);
      host_indices.resize(selected_count);
      check_cuda(cudaMemcpyAsync(host_scores.data(), impl_->scores.data(),
                                 selected_count * sizeof(float),
                                 cudaMemcpyDeviceToHost,
                                 execution_options.stream),
                 "cudaMemcpyAsync Loki compact scores");
      check_cuda(cudaMemcpyAsync(host_indices.data(), impl_->indices.data(),
                                 selected_count * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost,
                                 execution_options.stream),
                 "cudaMemcpyAsync Loki compact indices");
    }
    check_cuda(cudaStreamSynchronize(execution_options.stream),
               "cudaStreamSynchronize Loki compact peaks");
    peaks.reserve(
        checked_add(peaks.size(), selected_count, "Loki peak count overflow"));
    for (std::size_t index = 0; index < selected_count; ++index) {
      if (impl_->reduction != nullptr) {
        peaks.push_back(impl_->make_peak(
            region, host_refs[index].snr, host_refs[index].index));
      } else {
        peaks.push_back(impl_->make_peak(
            region, host_scores[index], host_indices[index]));
      }
    }
    // FFACUDA owns the region-specific phase map. Reset under the owning
    // device guard before moving on to the next region.
    ffa.reset();
  }
  return peaks;
}

SeriesPeaks LokiPffaProgram::search_batch(
    CudaTimeSeriesBatchView normalised_batch,
    LokiPffaExecutionOptions options) {
  if (normalised_batch.data == nullptr || normalised_batch.nseries == 0 ||
      normalised_batch.nsamples != plan().input_nsamples()) {
    throw std::invalid_argument(
        "Loki P-FFA batch must be non-empty and match the plan length");
  }
  if (normalised_batch.device_id != device_id()) {
    throw std::invalid_argument(
        "Loki P-FFA batch belongs to another CUDA device");
  }

  LokiPffaSearchDiagnostics batch_diagnostics;
  SeriesPeaks result;
  for (std::size_t series_index = 0;
       series_index < normalised_batch.nseries; ++series_index) {
    const CudaSpan<const float> series{
        .data = normalised_batch.data +
                series_index * normalised_batch.nsamples,
        .count = normalised_batch.nsamples,
        .device_id = normalised_batch.device_id,
    };
    std::vector<PeriodicPeak> peaks = search(series, options);
    batch_diagnostics.complete &= impl_->diagnostics.complete;
    batch_diagnostics.warnings.insert(
        batch_diagnostics.warnings.end(), impl_->diagnostics.warnings.begin(),
        impl_->diagnostics.warnings.end());
    result.reserve(result.size() + peaks.size());
    for (PeriodicPeak& peak : peaks) {
      result.push_back(SeriesPeak{
          .series_index = series_index,
          .peak = std::move(peak),
      });
    }
  }
  impl_->diagnostics = std::move(batch_diagnostics);
  return result;
}

}  // namespace gaffa
