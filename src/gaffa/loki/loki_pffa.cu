#include "gaffa/loki_pffa.h"

#include <loki/algorithms/ffa.hpp>
#include <loki/algorithms/regions.hpp>
#include <loki/common/plans.hpp>
#include <loki/detection/score.hpp>
#include <loki/search/configs.hpp>
#include <loki/utils/workspace.hpp>

#include <cuda/std/span>
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

std::vector<loki::ParamLimit> make_loki_limits(
    const LokiTaylorSearchSpace& search_space) {
  std::vector<loki::ParamLimit> limits;
  if (search_space.snap.has_value()) {
    limits.push_back(
        {.min = search_space.snap->minimum, .max = search_space.snap->maximum});
  }
  if (search_space.jerk.has_value()) {
    limits.push_back(
        {.min = search_space.jerk->minimum, .max = search_space.jerk->maximum});
  }
  if (search_space.acceleration.has_value()) {
    limits.push_back({.min = search_space.acceleration->minimum,
                      .max = search_space.acceleration->maximum});
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
  [[nodiscard]] static LokiPffaPeak make_peak(const Region& region, float snr,
                                               std::uint32_t flat_index);

  ~Impl() {
    if (options.device_id < 0) {
      return;
    }
    try {
      DeviceGuard guard(options.device_id);
      counter.reset();
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

LokiPffaPeak LokiPffaProgram::Impl::make_peak(const Region& region,
                                               float snr,
                                               std::uint32_t flat_index) {
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

  LokiPffaPeak peak{
      .snr = snr,
      .frequency_hz = values.back(),
      .phase_bins = static_cast<std::size_t>(region.config.get_nbins()),
      .boxcar_width = region.widths[width_index],
  };
  peak.duty_cycle = static_cast<float>(peak.boxcar_width) /
                    static_cast<float>(peak.phase_bins);
  if (values.size() >= 2) {
    peak.loki_acceleration = values[values.size() - 2];
  }
  if (values.size() >= 3) {
    peak.loki_jerk = values[values.size() - 3];
  }
  if (values.size() >= 4) {
    peak.loki_snap = values[values.size() - 4];
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

std::vector<LokiPffaPeak> LokiPffaProgram::search(
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
  if (execution_options.max_compact_peaks_total == 0) {
    throw std::invalid_argument("Loki P-FFA max_compact_peaks_total must be > 0");
  }

  ActiveSearchGuard active_search(impl_->search_active);
  DeviceGuard guard(impl_->options.device_id);
  if (execution_options.stream != nullptr) {
    int stream_device = -1;
    check_cuda(cudaStreamGetDevice(execution_options.stream, &stream_device),
               "cudaStreamGetDevice");
    if (stream_device != impl_->options.device_id) {
      throw std::invalid_argument("Loki P-FFA stream belongs to another CUDA device");
    }
  }
  impl_->initialize(execution_options.stream);

  std::vector<LokiPffaPeak> peaks;
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
    if (passing > execution_options.max_compact_peaks_total - peaks.size()) {
      throw std::runtime_error(
          "Loki P-FFA compact peak limit exceeded; raise max_compact_peaks_total "
          "or increase the SNR threshold");
    }
    if (passing == 0) {
      ffa.reset();
      continue;
    }
    std::vector<float> host_scores(passing);
    std::vector<std::uint32_t> host_indices(passing);
    check_cuda(cudaMemcpyAsync(host_scores.data(), impl_->scores.data(),
                               passing * sizeof(float), cudaMemcpyDeviceToHost,
                               execution_options.stream),
               "cudaMemcpyAsync Loki compact scores");
    check_cuda(cudaMemcpyAsync(host_indices.data(), impl_->indices.data(),
                               passing * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               execution_options.stream),
               "cudaMemcpyAsync Loki compact indices");
    check_cuda(cudaStreamSynchronize(execution_options.stream),
               "cudaStreamSynchronize Loki compact peaks");
    peaks.reserve(checked_add(peaks.size(), passing, "Loki peak count overflow"));
    for (std::size_t index = 0; index < passing; ++index) {
      peaks.push_back(Impl::make_peak(region, host_scores[index], host_indices[index]));
    }
    // FFACUDA owns the region-specific phase map. Reset under the owning
    // device guard before moving on to the next region.
    ffa.reset();
  }
  return peaks;
}

}  // namespace gaffa
