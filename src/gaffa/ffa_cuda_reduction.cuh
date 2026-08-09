// Device-side bounded reduction for Native FFA detections. This file is
// included by ffa_cuda.cu and intentionally has no public CUDA ABI.

struct FfaCudaReductionTask {
  double effective_tsamp = 0.0;
  std::uint64_t bins = 0;
  std::uint64_t rows = 0;
};

struct FfaCudaReductionRef {
  std::uint64_t group_key = 0;
  std::uint32_t raw_index = 0;
  float snr = 0.0F;
};

static_assert(sizeof(FfaCudaReductionRef) == 16);

struct FfaCudaReductionSummary {
  std::size_t count = 0;
  bool overflow = false;
  bool raw_peak_limit_exceeded = false;
  std::vector<std::size_t> raw_counts;
};

__global__ void build_ffa_reduction_refs_kernel(
    const FfaCudaPeak* raw_peaks,
    const FfaCudaReductionTask* tasks,
    std::size_t task_offset,
    std::size_t count,
    double frequency_tolerance_hz,
    std::uint64_t frequency_bucket_count,
    FfaCudaReductionRef* refs,
    float* scores,
    unsigned long long* raw_counts) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const FfaCudaPeak peak = raw_peaks[index];
  atomicAdd(raw_counts + peak.series_index, 1ULL);
  const FfaCudaReductionTask task = tasks[task_offset + peak.task_index];
  double period = task.effective_tsamp * static_cast<double>(task.bins);
  if (task.rows > 1) {
    const double bins = static_cast<double>(task.bins);
    period = task.effective_tsamp * bins * bins /
             (bins - static_cast<double>(peak.shift) /
                         static_cast<double>(task.rows - 1));
  }
  const double frequency = 1.0 / period;
  const std::uint64_t frequency_bucket = static_cast<std::uint64_t>(
      floor(frequency / frequency_tolerance_hz));
  const std::uint64_t group_key =
      static_cast<std::uint64_t>(peak.series_index) *
          frequency_bucket_count +
      frequency_bucket;
  refs[index] = FfaCudaReductionRef{
      .group_key = group_key,
      .raw_index = static_cast<std::uint32_t>(index),
      .snr = peak.snr,
  };
  scores[index] = peak.snr;
}

__global__ void extract_reduction_group_keys_kernel(
    const FfaCudaReductionRef* refs, std::size_t count,
    std::uint64_t* group_keys) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    group_keys[index] = refs[index].group_key;
  }
}

__global__ void select_ffa_reduction_peaks_kernel(
    const std::uint64_t* sorted_group_keys,
    const FfaCudaReductionRef* sorted_refs,
    const FfaCudaPeak* raw_peaks,
    std::size_t count,
    std::uint64_t frequency_bucket_count,
    std::size_t top_k,
    std::size_t max_groups_per_series,
    std::size_t output_capacity,
    std::uint32_t* group_counts,
    unsigned long long* output_count,
    unsigned int* overflow,
    FfaCudaPeak* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count ||
      (index != 0 && sorted_group_keys[index] == sorted_group_keys[index - 1])) {
    return;
  }

  const std::uint64_t group_key = sorted_group_keys[index];
  const std::size_t series =
      static_cast<std::size_t>(group_key / frequency_bucket_count);
  const std::size_t group_index = atomicAdd(group_counts + series, 1U);
  if (group_index >= max_groups_per_series) {
    atomicExch(overflow, 1U);
    return;
  }

  for (std::size_t rank = 0; rank < top_k && index + rank < count; ++rank) {
    if (sorted_group_keys[index + rank] != group_key) {
      break;
    }
    const unsigned long long write_index = atomicAdd(output_count, 1ULL);
    if (write_index >= output_capacity) {
      atomicExch(overflow, 1U);
      continue;
    }
    output[write_index] = raw_peaks[sorted_refs[index + rank].raw_index];
  }
}

std::size_t query_reduction_sort_bytes(std::size_t capacity) {
  std::size_t bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(
      nullptr, bytes, static_cast<const float*>(nullptr),
      static_cast<float*>(nullptr),
      static_cast<const FfaCudaReductionRef*>(nullptr),
      static_cast<FfaCudaReductionRef*>(nullptr), capacity);
  std::size_t group_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, group_bytes, static_cast<const std::uint64_t*>(nullptr),
      static_cast<std::uint64_t*>(nullptr),
      static_cast<const FfaCudaReductionRef*>(nullptr),
      static_cast<FfaCudaReductionRef*>(nullptr), capacity);
  return std::max(bytes, group_bytes);
}

std::uint64_t native_frequency_bucket_count(
    const CudaFfaExecutionPlan& plan,
    const PeakReductionOptions& options,
    double& tolerance_hz) {
  tolerance_hz = options.frequency_tolerance_hz;
  if (tolerance_hz == 0.0) {
    tolerance_hz = 1.0 / plan.observation().duration_seconds();
  }
  if (!(tolerance_hz > 0.0) || !std::isfinite(tolerance_hz)) {
    throw std::invalid_argument(
        "CUDA FFA frequency reduction tolerance must be finite and > 0");
  }

  double maximum_frequency = 0.0;
  for (const auto& group : plan.groups()) {
    for (const auto& task_layout : group.tasks) {
      const auto& task = task_layout.task;
      const double bins = static_cast<double>(task.bins);
      const double shift = static_cast<double>(task.rows_eval - 1);
      const double denominator =
          task.rows <= 1 ? bins : bins - shift / static_cast<double>(task.rows - 1);
      const double period = task.effective_tsamp * bins * bins / denominator;
      maximum_frequency = std::max(maximum_frequency, 1.0 / period);
    }
  }
  const long double count =
      std::ceil(static_cast<long double>(maximum_frequency) /
                static_cast<long double>(tolerance_hz)) + 1.0L;
  if (!(count > 0.0L) ||
      count > static_cast<long double>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::overflow_error(
        "CUDA FFA frequency reduction bucket count exceeds uint32_t");
  }
  return static_cast<std::uint64_t>(count);
}

class FfaCudaReductionWorkspace {
 public:
  FfaCudaReductionWorkspace() = default;

  FfaCudaReductionWorkspace(const CudaFfaExecutionPlan& plan,
                            const CudaFfaExecutionOptions& options,
                            std::size_t input_capacity,
                            int device_id)
      : enabled_(options.reduction.enabled()),
        top_k_(options.reduction.top_k_per_group),
        max_groups_(options.reduction.max_groups_per_series),
        input_capacity_(input_capacity),
        series_capacity_(options.series_tile_size),
        device_id_(device_id) {
    if (!enabled_) {
      return;
    }
    if (top_k_ == 0 || max_groups_ == 0) {
      throw std::invalid_argument(
          "CUDA peak reduction requires top_k_per_group and "
          "max_groups_per_series to be > 0");
    }
    frequency_bucket_count_ = native_frequency_bucket_count(
        plan, options.reduction, frequency_tolerance_hz_);
    for (const auto& group : plan.groups()) {
      task_offsets_.push_back(task_metadata_.size());
      for (const auto& task : group.tasks) {
        task_metadata_.push_back(FfaCudaReductionTask{
            .effective_tsamp = task.task.effective_tsamp,
            .bins = task.task.bins,
            .rows = task.task.rows,
        });
      }
    }
    allocate(input_capacity);
  }

  FfaCudaReductionWorkspace(const FfaCudaReductionWorkspace&) = delete;
  FfaCudaReductionWorkspace& operator=(const FfaCudaReductionWorkspace&) = delete;
  FfaCudaReductionWorkspace(FfaCudaReductionWorkspace&&) noexcept = default;
  FfaCudaReductionWorkspace& operator=(FfaCudaReductionWorkspace&&) noexcept = default;

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  [[nodiscard]] std::size_t output_capacity() const noexcept {
    return output_.size();
  }

  [[nodiscard]] double frequency_tolerance_hz() const noexcept {
    return frequency_tolerance_hz_;
  }

  [[nodiscard]] std::uint64_t frequency_bucket_count() const noexcept {
    return frequency_bucket_count_;
  }

  [[nodiscard]] std::size_t bytes() const noexcept {
    return task_metadata_device_.bytes() + refs_a_.bytes() + refs_b_.bytes() +
           scores_a_.bytes() + scores_b_.bytes() + group_keys_a_.bytes() +
           group_keys_b_.bytes() + group_counts_.bytes() + output_.bytes() +
           raw_counts_.bytes() + output_count_.bytes() + overflow_.bytes() +
           temp_storage_.bytes();
  }

  void resize_input(std::size_t capacity) {
    if (!enabled_ || capacity <= input_capacity_) {
      return;
    }
    input_capacity_ = capacity;
    allocate(capacity);
  }

  FfaCudaReductionSummary reduce(std::size_t group_index,
                                 const CudaDeviceBuffer<FfaCudaPeak>& raw,
                                 std::size_t count,
                                 std::size_t max_peaks,
                                 cudaStream_t stream) {
    if (!enabled_) {
      throw std::logic_error("CUDA peak reduction is disabled");
    }
    if (group_index >= task_offsets_.size() || count > input_capacity_) {
      throw std::invalid_argument("CUDA peak reduction input is out of range");
    }
    if (count == 0) {
      return FfaCudaReductionSummary{
          .count = 0,
          .overflow = false,
          .raw_peak_limit_exceeded = false,
          .raw_counts = std::vector<std::size_t>(series_capacity_, 0),
      };
    }
    check_cuda(cudaMemsetAsync(group_counts_.data(), 0,
                               group_counts_.bytes(), stream),
               "CUDA FFA reduction group counter reset");
    check_cuda(cudaMemsetAsync(output_count_.data(), 0,
                               output_count_.bytes(), stream),
               "CUDA FFA reduction output counter reset");
    check_cuda(cudaMemsetAsync(overflow_.data(), 0, overflow_.bytes(), stream),
               "CUDA FFA reduction overflow reset");
    check_cuda(cudaMemsetAsync(raw_counts_.data(), 0, raw_counts_.bytes(),
                               stream),
               "CUDA FFA reduction raw count reset");

    const unsigned int blocks = checked_grid_dim(
        (count + 255U) / 256U, "CUDA FFA reduction encode grid overflow");
    build_ffa_reduction_refs_kernel<<<blocks, 256, 0, stream>>>(
        raw.data(), task_metadata_device_.data(), task_offsets_[group_index],
        count, frequency_tolerance_hz_, frequency_bucket_count_, refs_a_.data(),
        scores_a_.data(), raw_counts_.data());
    check_cuda(cudaGetLastError(), "CUDA FFA reduction encode launch");

    std::size_t temp_bytes = temp_storage_.bytes();
    cub::DeviceRadixSort::SortPairsDescending(
        temp_storage_.data(), temp_bytes, scores_a_.data(),
        scores_b_.data(), refs_a_.data(), refs_b_.data(), count, 0,
        sizeof(float) * 8, stream);
    extract_reduction_group_keys_kernel<<<blocks, 256, 0, stream>>>(
        refs_b_.data(), count, group_keys_a_.data());
    check_cuda(cudaGetLastError(), "CUDA FFA reduction key extraction launch");
    cub::DeviceRadixSort::SortPairs(
        temp_storage_.data(), temp_bytes, group_keys_a_.data(),
        group_keys_b_.data(), refs_b_.data(), refs_a_.data(), count, 0,
        sizeof(std::uint64_t) * 8, stream);

    select_ffa_reduction_peaks_kernel<<<blocks, 256, 0, stream>>>(
        group_keys_b_.data(), refs_a_.data(), raw.data(), count,
        frequency_bucket_count_, top_k_, max_groups_, output_.size(),
        group_counts_.data(), output_count_.data(), overflow_.data(),
        output_.data());
    check_cuda(cudaGetLastError(), "CUDA FFA reduction select launch");
    check_cuda(cudaStreamSynchronize(stream), "CUDA FFA reduction synchronize");

    unsigned long long count_host = 0;
    unsigned int overflow_host = 0;
    check_cuda(cudaMemcpy(&count_host, output_count_.data(), sizeof(count_host),
                          cudaMemcpyDeviceToHost),
               "CUDA FFA reduction count D2H");
    check_cuda(cudaMemcpy(&overflow_host, overflow_.data(), sizeof(overflow_host),
                          cudaMemcpyDeviceToHost),
               "CUDA FFA reduction overflow D2H");
    if (count_host > output_.size()) {
      throw std::logic_error("CUDA FFA reduction output count is invalid");
    }
    std::vector<unsigned long long> raw_counts_host(series_capacity_);
    check_cuda(cudaMemcpy(raw_counts_host.data(), raw_counts_.data(),
                          raw_counts_.bytes(), cudaMemcpyDeviceToHost),
               "CUDA FFA reduction raw counts D2H");
    std::vector<std::size_t> raw_counts_result(series_capacity_);
    bool raw_peak_limit_exceeded = false;
    for (std::size_t series = 0; series < series_capacity_; ++series) {
      if (raw_counts_host[series] >
          std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(
            "CUDA FFA reduction raw count exceeds host size_t");
      }
      raw_counts_result[series] =
          static_cast<std::size_t>(raw_counts_host[series]);
      raw_peak_limit_exceeded |=
          max_peaks != 0 && raw_counts_result[series] > max_peaks;
    }
    return FfaCudaReductionSummary{
        .count = static_cast<std::size_t>(count_host),
        .overflow = overflow_host != 0,
        .raw_peak_limit_exceeded = raw_peak_limit_exceeded,
        .raw_counts = std::move(raw_counts_result),
    };
  }

  [[nodiscard]] const CudaDeviceBuffer<FfaCudaPeak>& output() const noexcept {
    return output_;
  }

 private:
  void allocate(std::size_t capacity) {
    CudaDeviceBuffer<FfaCudaReductionTask> metadata(task_metadata_.size());
    if (!task_metadata_.empty()) {
      check_cuda(cudaMemcpy(metadata.data(), task_metadata_.data(),
                            metadata.bytes(), cudaMemcpyHostToDevice),
                 "CUDA FFA reduction task metadata H2D");
    }
    CudaDeviceBuffer<FfaCudaReductionRef> refs_a(capacity);
    CudaDeviceBuffer<FfaCudaReductionRef> refs_b(capacity);
    CudaDeviceBuffer<float> scores_a(capacity);
    CudaDeviceBuffer<float> scores_b(capacity);
    CudaDeviceBuffer<std::uint64_t> group_keys_a(capacity);
    CudaDeviceBuffer<std::uint64_t> group_keys_b(capacity);
    CudaDeviceBuffer<std::uint32_t> group_counts(series_capacity_);
    const std::size_t output_count = checked_multiply(
        checked_multiply(series_capacity_, max_groups_,
                         "CUDA FFA reduction output size overflow"),
        top_k_, "CUDA FFA reduction output size overflow");
    CudaDeviceBuffer<FfaCudaPeak> output(output_count);
    CudaDeviceBuffer<unsigned long long> output_counter(1);
    CudaDeviceBuffer<unsigned int> overflow(1);
    CudaDeviceMemory temp(query_reduction_sort_bytes(capacity));

    task_metadata_device_ = std::move(metadata);
    refs_a_ = std::move(refs_a);
    refs_b_ = std::move(refs_b);
    scores_a_ = std::move(scores_a);
    scores_b_ = std::move(scores_b);
    group_keys_a_ = std::move(group_keys_a);
    group_keys_b_ = std::move(group_keys_b);
    group_counts_ = std::move(group_counts);
    raw_counts_ = CudaDeviceBuffer<unsigned long long>(series_capacity_);
    output_ = std::move(output);
    output_count_ = std::move(output_counter);
    overflow_ = std::move(overflow);
    temp_storage_ = std::move(temp);
  }

  bool enabled_ = false;
  std::size_t top_k_ = 0;
  std::size_t max_groups_ = 0;
  std::size_t input_capacity_ = 0;
  std::size_t series_capacity_ = 0;
  int device_id_ = 0;
  double frequency_tolerance_hz_ = 0.0;
  std::uint64_t frequency_bucket_count_ = 0;
  std::vector<std::size_t> task_offsets_;
  std::vector<FfaCudaReductionTask> task_metadata_;
  CudaDeviceBuffer<FfaCudaReductionTask> task_metadata_device_;
  CudaDeviceBuffer<FfaCudaReductionRef> refs_a_;
  CudaDeviceBuffer<FfaCudaReductionRef> refs_b_;
  CudaDeviceBuffer<float> scores_a_;
  CudaDeviceBuffer<float> scores_b_;
  CudaDeviceBuffer<std::uint64_t> group_keys_a_;
  CudaDeviceBuffer<std::uint64_t> group_keys_b_;
  CudaDeviceBuffer<std::uint32_t> group_counts_;
  CudaDeviceBuffer<unsigned long long> raw_counts_;
  CudaDeviceBuffer<FfaCudaPeak> output_;
  CudaDeviceBuffer<unsigned long long> output_count_;
  CudaDeviceBuffer<unsigned int> overflow_;
  CudaDeviceMemory temp_storage_;
};

std::size_t estimate_ffa_cuda_reduction_bytes(
    const CudaFfaExecutionPlan& plan,
    const CudaFfaExecutionOptions& options,
    std::size_t input_capacity) {
  if (!options.reduction.enabled()) {
    return 0;
  }
  if (options.reduction.max_groups_per_series == 0 || input_capacity == 0) {
    throw std::invalid_argument(
        "CUDA peak reduction requires a non-empty input capacity and "
        "max_groups_per_series > 0");
  }

  double tolerance_hz = 0.0;
  (void)native_frequency_bucket_count(plan, options.reduction, tolerance_hz);

  std::size_t task_count = 0;
  for (const auto& group : plan.groups()) {
    task_count = checked_add(
        task_count, group.tasks.size(),
        "CUDA FFA reduction task metadata size overflow");
  }

  std::size_t bytes = 0;
  const auto add_array = [&bytes](std::size_t count, std::size_t element_size,
                                  const char* message) {
    bytes = checked_add(bytes, checked_multiply(count, element_size, message),
                        "CUDA FFA reduction workspace byte size overflow");
  };
  add_array(task_count, sizeof(FfaCudaReductionTask),
            "CUDA FFA reduction task metadata byte size overflow");
  add_array(input_capacity, sizeof(FfaCudaReductionRef),
            "CUDA FFA reduction reference byte size overflow");
  add_array(input_capacity, sizeof(FfaCudaReductionRef),
            "CUDA FFA reduction reference byte size overflow");
  add_array(input_capacity, sizeof(float),
            "CUDA FFA reduction score byte size overflow");
  add_array(input_capacity, sizeof(float),
            "CUDA FFA reduction score byte size overflow");
  add_array(input_capacity, sizeof(std::uint64_t),
            "CUDA FFA reduction group key byte size overflow");
  add_array(input_capacity, sizeof(std::uint64_t),
            "CUDA FFA reduction group key byte size overflow");
  add_array(options.series_tile_size, sizeof(std::uint32_t),
            "CUDA FFA reduction group counter byte size overflow");
  add_array(options.series_tile_size, sizeof(unsigned long long),
            "CUDA FFA reduction raw counter byte size overflow");
  const std::size_t output_count = checked_multiply(
      checked_multiply(options.series_tile_size,
                       options.reduction.max_groups_per_series,
                       "CUDA FFA reduction output size overflow"),
      options.reduction.top_k_per_group,
      "CUDA FFA reduction output size overflow");
  add_array(output_count, sizeof(FfaCudaPeak),
            "CUDA FFA reduction output byte size overflow");
  add_array(1, sizeof(unsigned long long),
            "CUDA FFA reduction output counter byte size overflow");
  add_array(1, sizeof(unsigned int),
            "CUDA FFA reduction overflow byte size overflow");
  bytes = checked_add(bytes, query_reduction_sort_bytes(input_capacity),
                      "CUDA FFA reduction workspace byte size overflow");
  return bytes;
}
