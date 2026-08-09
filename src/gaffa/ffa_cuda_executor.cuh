// Public CUDA FFA objects and the stateful tile executor.
struct CudaFfaProgramState {
  CudaFfaExecutionPlan execution_plan;
  FfaCudaProgramLayout layout;
  FfaCudaProgramOps ops;
};

struct CudaFfaProgramImpl {
  explicit CudaFfaProgramImpl(int device_id,
                              CudaFfaExecutionOptions execution_options)
      : device_id(device_id), execution_options(std::move(execution_options)) {}

  int device_id = 0;
  CudaFfaExecutionOptions execution_options;
  std::optional<CudaFfaExecutionPlan> execution_plan;
  std::optional<FfaSearchPlan> source_plan;
  FfaCudaProgramLayout layout;
  FfaCudaProgramOps ops;
  std::unique_ptr<CudaFfaWorkspace> workspace;
};

bool same_ffa_search_plan(const FfaSearchPlan& lhs,
                          const FfaSearchPlan& rhs) {
  if (lhs.observation.nsamples != rhs.observation.nsamples ||
      lhs.observation.tsamp_seconds != rhs.observation.tsamp_seconds ||
      lhs.width_trials != rhs.width_trials ||
      lhs.tasks.size() != rhs.tasks.size()) {
    return false;
  }

  for (std::size_t index = 0; index < lhs.tasks.size(); ++index) {
    const FfaSearchTask& left = lhs.tasks[index];
    const FfaSearchTask& right = rhs.tasks[index];
    if (left.downsample_factor != right.downsample_factor ||
        left.effective_tsamp != right.effective_tsamp ||
        left.prepared_nsamples != right.prepared_nsamples ||
        left.bins != right.bins || left.rows != right.rows ||
        left.rows_eval != right.rows_eval ||
        left.period_begin != right.period_begin ||
        left.period_end != right.period_end) {
      return false;
    }
  }
  return true;
}

CudaFfaProgramState build_cuda_ffa_program_state(
    CudaFfaExecutionPlan execution_plan,
    int device_id) {
  CudaFfaProgramState state{.execution_plan = std::move(execution_plan)};
  FfaCudaProgramBuildState build_state;
  build_state.layout.groups.reserve(state.execution_plan.groups().size());

  for (const auto& group : state.execution_plan.groups()) {
    FfaCudaProgramGroup program_group;
    program_group.tasks.reserve(group.tasks.size());
    for (const auto& task : group.tasks) {
      append_program_task_layout(task, build_state, program_group, device_id);
    }
    build_state.layout.groups.push_back(std::move(program_group));
  }
  configure_subtree_dynamic_shared_memory(build_state.max_subtree_shared_bytes);
  state.ops = materialize_program_ops(build_state);
  state.layout = std::move(build_state.layout);
  return state;
}

CudaFfaExecutionPlan::CudaFfaExecutionPlan(
    FfaObservation observation,
    std::vector<CudaFfaPrepareGroup> groups,
    std::size_t max_prepared_nsamples,
    std::size_t max_transform_elements,
    std::size_t max_detection_slots_per_series)
    : observation_(observation),
      groups_(std::move(groups)),
      max_prepared_nsamples_(max_prepared_nsamples),
      max_transform_elements_(max_transform_elements),
      max_detection_slots_per_series_(max_detection_slots_per_series) {}

const FfaObservation& CudaFfaExecutionPlan::observation() const noexcept {
  return observation_;
}

std::span<const CudaFfaPrepareGroup> CudaFfaExecutionPlan::groups() const
    noexcept {
  return groups_;
}

std::size_t CudaFfaExecutionPlan::max_prepared_nsamples() const noexcept {
  return max_prepared_nsamples_;
}

std::size_t CudaFfaExecutionPlan::max_transform_elements() const noexcept {
  return max_transform_elements_;
}

std::size_t CudaFfaExecutionPlan::max_detection_slots_per_series() const
    noexcept {
  return max_detection_slots_per_series_;
}

CudaFfaProgram::CudaFfaProgram(
    const CudaFfaProgramOptions& program_options,
    const CudaFfaExecutionOptions& execution_options)
    : impl_(nullptr) {
  validate_program_options(program_options);
  validate_execution_options(execution_options);
  check_cuda(cudaSetDevice(program_options.device_id), "cudaSetDevice");
  impl_ = std::make_unique<CudaFfaProgramImpl>(program_options.device_id,
                                               execution_options);
}

CudaFfaProgram::CudaFfaProgram(
    CudaFfaExecutionPlan execution_plan,
    const CudaFfaProgramOptions& program_options,
    const CudaFfaExecutionOptions& execution_options)
    : CudaFfaProgram(program_options, execution_options) {
  CudaDeviceScope device_scope(device_id());
  CudaFfaProgramState state = build_cuda_ffa_program_state(
      std::move(execution_plan), device_id());
  auto workspace = std::make_unique<CudaFfaWorkspace>(
      state.execution_plan, impl_->execution_options, device_id());
  impl_->execution_plan.emplace(std::move(state.execution_plan));
  impl_->layout = std::move(state.layout);
  impl_->ops = std::move(state.ops);
  impl_->workspace = std::move(workspace);
}

CudaFfaProgram::CudaFfaProgram(
    const FfaSearchPlan& plan,
    const CudaFfaProgramOptions& program_options,
    const CudaFfaExecutionOptions& execution_options)
    : CudaFfaProgram(program_options, execution_options) {
  prepare(plan);
}

CudaFfaProgram::~CudaFfaProgram() = default;

CudaFfaProgram::CudaFfaProgram(CudaFfaProgram&&) noexcept = default;

CudaFfaProgram& CudaFfaProgram::operator=(CudaFfaProgram&&) noexcept = default;

bool CudaFfaProgram::empty() const noexcept {
  return impl_ == nullptr;
}

bool CudaFfaProgram::prepared() const noexcept {
  return impl_ != nullptr && impl_->execution_plan.has_value() &&
         impl_->workspace != nullptr;
}

void CudaFfaProgram::prepare(const FfaSearchPlan& plan) {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  validate_ffa_search_plan(plan);
  if (impl_->source_plan.has_value() &&
      same_ffa_search_plan(*impl_->source_plan, plan)) {
    return;
  }

  // Copy the logical plan before releasing the active device state. The copy
  // is host-only and gives the resource transition a clean commit point.
  std::optional<FfaSearchPlan> source_plan = plan;
  CudaFfaExecutionPlan execution_plan = make_ffa_cuda_execution_plan(plan);
  const CudaFfaWorkspaceShape required_workspace =
      estimate_ffa_cuda_workspace(execution_plan, impl_->execution_options);
  const bool reuse_workspace =
      !impl_->execution_options.reduction.enabled() &&
      impl_->workspace != nullptr &&
      impl_->workspace->can_hold(required_workspace);

  CudaDeviceScope device_scope(device_id());

  // Build metadata before changing the active state. This keeps validation and
  // metadata construction independent from the lifetime of the large buffers.
  CudaFfaProgramState state = build_cuda_ffa_program_state(
      std::move(execution_plan), device_id());

  if (!reuse_workspace) {
    // A capacity increase is rare. Release the old large buffers before
    // allocating the replacement so a plan transition does not require two
    // complete FFA workspaces simultaneously. If allocation fails, the
    // Program remains unprepared, just as after clear().
    clear();
    impl_->workspace = std::make_unique<CudaFfaWorkspace>(
        required_workspace, state.execution_plan, impl_->execution_options,
        device_id());
  } else {
    // Device metadata is also active state. Do not release it while an earlier
    // search can still be using it.
    check_cuda(cudaStreamSynchronize(impl_->execution_options.stream),
               "CUDA FFA prepare synchronize");
  }

  impl_->execution_plan.emplace(std::move(state.execution_plan));
  impl_->layout = std::move(state.layout);
  impl_->ops = std::move(state.ops);
  impl_->source_plan = std::move(source_plan);
}

void CudaFfaProgram::clear() {
  if (empty()) {
    return;
  }
  CudaDeviceScope device_scope(device_id());
  check_cuda(cudaStreamSynchronize(impl_->execution_options.stream),
             "CUDA FFA program clear synchronize");
  impl_->workspace.reset();
  impl_->ops = FfaCudaProgramOps{};
  impl_->layout = FfaCudaProgramLayout{};
  impl_->execution_plan.reset();
  impl_->source_plan.reset();
}

const CudaFfaExecutionPlan& CudaFfaProgram::execution_plan() const {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  if (!prepared()) {
    throw std::logic_error("CUDA FFA program must be prepared");
  }
  return *impl_->execution_plan;
}

int CudaFfaProgram::device_id() const {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  return impl_->device_id;
}

std::size_t CudaFfaProgram::tile_capacity() const {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  return impl_->execution_options.series_tile_size;
}

cudaStream_t CudaFfaProgram::stream() const {
  if (empty()) {
    throw std::logic_error("CUDA FFA program must not be empty");
  }
  return impl_->execution_options.stream;
}

const CudaFfaWorkspaceShape& CudaFfaProgram::workspace_shape() const {
  (void)execution_plan();
  // This is the allocated capacity. It may exceed the active plan's exact
  // requirement after a smaller plan is prepared.
  return impl_->workspace->shape;
}

std::size_t CudaFfaProgram::device_metadata_bytes() const {
  (void)execution_plan();
  return checked_add(
      checked_add(
          checked_add(impl_->ops.subtree_ops.bytes(),
                      impl_->ops.subtree_schedules.bytes(),
                      "CUDA FFA program metadata byte size overflow"),
          checked_add(
              checked_add(impl_->ops.subtree_levels.bytes(),
                          impl_->ops.subtree_copy_ops.bytes(),
                          "CUDA FFA program metadata byte size overflow"),
              impl_->ops.subtree_merge_ops.bytes(),
              "CUDA FFA program metadata byte size overflow"),
          "CUDA FFA program metadata byte size overflow"),
      checked_add(
          checked_add(impl_->ops.copy_ops.bytes(), impl_->ops.merge_ops.bytes(),
                      "CUDA FFA program metadata byte size overflow"),
          impl_->ops.detection_trials.bytes(),
          "CUDA FFA program metadata byte size overflow"),
      "CUDA FFA program metadata byte size overflow");
}

CudaFfaExecutionPlan make_ffa_cuda_execution_plan(const FfaSearchPlan& plan) {
  validate_plan_for_workspace(plan);

  const std::size_t input_nsamples = plan.observation.nsamples;
  std::vector<CudaFfaPrepareGroup> groups;
  std::size_t max_prepared_nsamples = 0;
  std::size_t max_transform_elements = 0;
  // A prepare group appends all of its task detections to one compact
  // buffer. Track the largest group total, rather than the largest task.
  std::size_t max_detection_slots_per_series = 0;
  for (const auto& task : plan.tasks) {
    const std::size_t transform_elements = checked_multiply(
        task.rows, task.bins, "CUDA FFA task layout size overflow");
    const FfaDetectionPlan detection_plan =
        make_ffa_detection_plan(plan.width_trials, task.bins);
    const std::size_t detection_slots_per_series = checked_multiply(
        task.rows_eval, detection_plan.boxcar_trials.size(),
        "CUDA FFA detection slot count overflow");
    const CudaFfaTaskLayout task_layout{
        .task = task,
        .shape = FfaTransformShape{
            .rows = task.rows,
            .bins = task.bins,
        },
        .detection_plan = detection_plan,
        .transform_elements = transform_elements,
        .detection_slots_per_series = detection_slots_per_series,
    };

    auto group = std::find_if(
        groups.begin(), groups.end(),
        [&](const CudaFfaPrepareGroup& candidate) {
          return candidate.prepare_key.input_nsamples == input_nsamples &&
                 candidate.prepare_key.downsample_factor ==
                     task.downsample_factor;
        });
    if (group == groups.end()) {
      group = groups.insert(
          groups.end(),
          CudaFfaPrepareGroup{
              .prepare_key = CudaFfaPrepareKey{
                  .input_nsamples = input_nsamples,
                  .downsample_factor = task.downsample_factor,
              },
              .prepared_nsamples = task.prepared_nsamples,
          });
    }

    if (group->prepared_nsamples != task.prepared_nsamples) {
      throw std::logic_error(
          "CUDA FFA prepare group has inconsistent prepared_nsamples");
    }
    group->tasks.push_back(task_layout);
    group->detection_slots_per_series = checked_add(
        group->detection_slots_per_series, detection_slots_per_series,
        "CUDA FFA prepare group detection slot count overflow");
    max_prepared_nsamples =
        std::max(max_prepared_nsamples, task.prepared_nsamples);
    max_transform_elements =
        std::max(max_transform_elements, transform_elements);
    max_detection_slots_per_series = std::max(
        max_detection_slots_per_series, group->detection_slots_per_series);
  }

  return CudaFfaExecutionPlan(plan.observation, std::move(groups), max_prepared_nsamples,
                              max_transform_elements,
                              max_detection_slots_per_series);
}

CudaFfaWorkspaceShape estimate_ffa_cuda_workspace(
    const CudaFfaExecutionPlan& plan,
    const CudaFfaExecutionOptions& options) {
  validate_execution_options(options);
  if (plan.groups().empty()) {
    throw std::invalid_argument(
        "CUDA FFA execution plan must contain at least one group");
  }
  if (plan.max_prepared_nsamples() == 0) {
    throw std::invalid_argument(
        "CUDA FFA execution plan max_prepared_nsamples must be > 0");
  }
  if (plan.max_transform_elements() == 0) {
    throw std::invalid_argument(
        "CUDA FFA execution plan max_task_elements must be > 0");
  }
  if (plan.max_detection_slots_per_series() == 0) {
    throw std::invalid_argument(
        "CUDA FFA execution plan max_detection_slots_per_series must be > 0");
  }

  const std::size_t prepared_count = checked_multiply(
      options.series_tile_size, plan.max_prepared_nsamples(),
      "CUDA FFA prepared workspace element count overflow");
  const std::size_t task_count = checked_multiply(
      options.series_tile_size, plan.max_transform_elements(),
      "CUDA FFA transform workspace element count overflow");
  const std::size_t detection_count = checked_multiply(
      options.series_tile_size, plan.max_detection_slots_per_series(),
      "CUDA FFA detection workspace element count overflow");
  const std::size_t initial_peak_records =
      options.initial_peak_buffer_bytes / sizeof(FfaCudaPeak);
  const std::size_t initial_peak_bytes = checked_multiply(
      std::min(detection_count, initial_peak_records), sizeof(FfaCudaPeak),
      "CUDA FFA detection peak workspace byte size overflow");

  CudaFfaWorkspaceShape shape{
      .series_tile_size = options.series_tile_size,
      .max_prepared_nsamples = plan.max_prepared_nsamples(),
      .max_task_elements = plan.max_transform_elements(),
      .max_detection_slots_per_series =
          plan.max_detection_slots_per_series(),
      .prepared_bytes = checked_float_bytes(
          prepared_count, "CUDA FFA prepared workspace byte size overflow"),
      .scratch_bytes = checked_float_bytes(
          task_count, "CUDA FFA scratch workspace byte size overflow"),
      .output_bytes = checked_float_bytes(
          task_count, "CUDA FFA output workspace byte size overflow"),
      .detection_compact_bytes = initial_peak_bytes,
  };
  shape.reduction_bytes = estimate_ffa_cuda_reduction_bytes(
      plan, options,
      shape.detection_compact_bytes / sizeof(FfaCudaPeak));
  shape.total_bytes = checked_add(
      checked_add(
          checked_add(shape.prepared_bytes, shape.scratch_bytes,
                      "CUDA FFA workspace byte size overflow"),
          shape.output_bytes, "CUDA FFA workspace byte size overflow"),
      checked_add(shape.detection_compact_bytes, shape.reduction_bytes,
                  "CUDA FFA workspace byte size overflow"),
      "CUDA FFA workspace byte size overflow");

  if (options.workspace_bytes_limit != 0 &&
      shape.total_bytes > options.workspace_bytes_limit) {
    throw std::runtime_error("CUDA FFA workspace exceeds workspace_bytes_limit");
  }

  return shape;
}

void prepare_ffa_input_cuda(CudaTimeSeriesBatchView input,
                            const FfaSearchTask& task,
                            CudaSpan<float> output,
                            const CudaLaunchOptions& options) {
  validate_prepare_arguments(input, task, output, options);
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");

  const auto threads = static_cast<unsigned int>(options.threads_per_block);
  const std::size_t x_blocks =
      (task.prepared_nsamples + threads - 1) / threads;
  const dim3 block(threads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA prepare grid x overflow"),
      checked_grid_dim(input.nseries, "CUDA FFA prepare grid y overflow"));

  if (task.downsample_factor == 1.0) {
    prepare_copy_float_kernel<<<grid, block, 0, options.stream>>>(
        input.data, input.nsamples, output.data, task.prepared_nsamples);
    check_cuda(cudaGetLastError(), "prepare_ffa_input_cuda kernel launch");
    if (options.synchronize_after_call) {
      check_cuda(cudaStreamSynchronize(options.stream),
                 "prepare_ffa_input_cuda synchronize");
    }
  } else {
    downsample_weighted_sum_cuda(input, task.downsample_factor, output,
                                 options);
    return;
  }
}

void ffa_transform_block_cuda(CudaFfaInput input,
                              CudaFfaBuffer scratch,
                              CudaFfaBuffer output,
                              const CudaLaunchOptions& options) {
  (void)validate_transform_arguments(input, scratch, output, options);
  if (!options.synchronize_after_call) {
    throw std::invalid_argument(
        "CUDA FFA materialized transform requires synchronize_after_call=true");
  }
  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");

  const std::vector<FfaCudaTransformLevel> levels =
      build_transform_levels(input.shape);
  const std::vector<FfaCudaDeviceTransformLevel> device_levels =
      materialize_device_levels(levels, input.shape);
  launch_transform_levels(levels, device_levels, input, scratch, output,
                          options);
}

void ffa_transform_block_cuda(const CudaFfaProgram& program,
                              std::size_t group_index,
                              std::size_t task_index,
                              CudaFfaInput input,
                              CudaFfaBuffer scratch,
                              CudaFfaBuffer output,
                              const CudaLaunchOptions& options) {
  if (program.empty()) {
    throw std::invalid_argument("CUDA FFA program must not be empty");
  }
  if (!program.prepared()) {
    throw std::logic_error("CUDA FFA program must be prepared");
  }
  validate_launch_options(options);
  if (program.device_id() != options.device_id) {
    throw std::invalid_argument(
        "CUDA FFA program device_id must match launch device_id");
  }
  if (group_index >= program.impl_->layout.groups.size()) {
    throw std::out_of_range("CUDA FFA program group_index is out of range");
  }
  const auto& group = program.impl_->layout.groups[group_index];
  const auto& plan_group = program.execution_plan().groups()[group_index];
  if (task_index >= group.tasks.size()) {
    throw std::out_of_range("CUDA FFA program task_index is out of range");
  }

  const auto& program_task = group.tasks[task_index];
  const auto& task_layout = plan_group.tasks[task_index];
  (void)validate_transform_arguments(input, scratch, output, options);
  validate_transform_matches_task(input, scratch, output, task_layout);
  if (program_task.level_count == 0) {
    throw std::logic_error("CUDA FFA program task has no transform levels");
  }

  check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice");
  launch_program_transform(program_task, program.impl_->layout,
                           program.impl_->ops, input, scratch, output,
                           options);
}

namespace detail {

class CudaFfaCapacityError final : public std::runtime_error {
 public:
  explicit CudaFfaCapacityError(const std::string& message)
      : std::runtime_error(message) {}
};

// Executes one dense [series][sample] tile using workspace owned by a
// long-lived CudaFfaProgram. It intentionally owns no CUDA allocations.
class CudaFfaTileRunner {
 private:
  static CudaFfaWorkspace& workspace_for(CudaFfaProgram& program) {
    if (program.empty()) {
      throw std::invalid_argument("CUDA FFA program must not be empty");
    }
    if (!program.prepared() || program.impl_->workspace == nullptr) {
      throw std::logic_error("CUDA FFA program must be prepared");
    }
    return *program.impl_->workspace;
  }

 public:
  CudaFfaTileRunner(CudaFfaProgram& program, FfaSearchOptions search_options)
      : program_(program), search_options_(search_options),
        workspace_(workspace_for(program)) {
    validate_search_options(search_options_);
    check_cuda(cudaSetDevice(program_.device_id()), "cudaSetDevice");
  }

  FfaBatchSearchResult run_tile(CudaTimeSeriesBatchView input) {
    validate_tile(input);
    if (program_.impl_->execution_options.reduction.enabled()) {
      return run_reduced_tile(input);
    }
    return run_unreduced_tile(input);
  }

 private:
  FfaBatchSearchResult run_unreduced_tile(CudaTimeSeriesBatchView input) {
    FfaBatchSearchResult result;
    std::vector<std::size_t> peak_counts(input.nseries, 0);
    const CudaLaunchOptions launch =
        program_.impl_->execution_options.async_launch_options(
            program_.device_id());
    const auto groups = program_.execution_plan().groups();
    for (std::size_t group_index = 0; group_index < groups.size();
         ++group_index) {
      for (;;) {
        prepare_group_input(group_index, input, launch);
        const CudaFfaGroupPeakSummary summary = execute_task_batch(
            group_index, input, launch, 0, groups[group_index].tasks.size());
        if (!summary.overflow) {
          collect_group_peaks(group_index, input.nseries, summary.count,
                              peak_counts, result);
          break;
        }
        grow_peak_buffer(group_index, summary.count);
      }
    }
    std::sort(result.peaks.begin(), result.peaks.end(),
              [](const FfaBatchPeak& lhs, const FfaBatchPeak& rhs) {
                if (lhs.series_index != rhs.series_index) {
                  return lhs.series_index < rhs.series_index;
                }
                return is_better_ffa_peak(lhs.peak, rhs.peak);
              });
    return result;
  }

  FfaBatchSearchResult run_reduced_tile(CudaTimeSeriesBatchView input) {
    FfaBatchSearchResult result;
    NativePeakAccumulator accumulator(
        input.nseries, program_.impl_->execution_options.reduction,
        workspace_.reduction.frequency_tolerance_hz(),
        workspace_.reduction.frequency_bucket_count(),
        search_options_.max_peaks);
    const CudaLaunchOptions launch =
        program_.impl_->execution_options.async_launch_options(
            program_.device_id());
    const auto groups = program_.execution_plan().groups();

    for (std::size_t group_index = 0; group_index < groups.size();
         ++group_index) {
      prepare_group_input(group_index, input, launch);
      for (std::size_t task_begin = 0;
           task_begin < groups[group_index].tasks.size();) {
        const std::size_t task_end = choose_task_batch_end(
            groups[group_index], task_begin, input.nseries);
        CudaFfaGroupPeakSummary summary{};
        bool discarded = false;
        for (;;) {
          summary = execute_task_batch(group_index, input, launch, task_begin,
                                       task_end);
          if (!summary.overflow) {
            break;
          }
          try {
            grow_peak_buffer(group_index, summary.count);
          } catch (const CudaFfaCapacityError& error) {
            result.complete = false;
            result.warnings.push_back(
                "Native CUDA reduction discarded task batch " +
                std::to_string(task_begin) + "-" +
                std::to_string(task_end) + ": " + error.what());
            discarded = true;
            break;
          }
        }
        if (!discarded) {
          const FfaCudaReductionSummary reduced = workspace_.reduction.reduce(
              group_index, workspace_.detection_compact, summary.count,
              search_options_.max_peaks,
              program_.impl_->execution_options.stream);
          if (reduced.raw_peak_limit_exceeded) {
            throw std::runtime_error(
                "CUDA FFA detection peak count exceeded max_peaks safety guard");
          }
          accumulator.add_raw_counts(reduced.raw_counts);
          if (reduced.overflow) {
            result.complete = false;
            result.warnings.push_back(
                "Native CUDA reduction discarded task batch " +
                std::to_string(task_begin) + "-" + std::to_string(task_end) +
                ": max_groups_per_series was exceeded");
          }
          // The selector writes retained groups before setting overflow. Keep
          // that valid partial result; only the omitted groups are lossy.
          collect_reduced_peaks(group_index, input.nseries, reduced.count,
                                accumulator);
        }
        task_begin = task_end;
      }
    }
    accumulator.finish(result);
    return result;
  }

  class NativePeakAccumulator {
   public:
    NativePeakAccumulator(std::size_t nseries,
                          const PeakReductionOptions& options,
                          double frequency_tolerance_hz,
                          std::uint64_t frequency_bucket_count,
                          std::size_t max_peaks)
        : nseries_(nseries), options_(options),
          frequency_tolerance_hz_(frequency_tolerance_hz),
          frequency_bucket_count_(frequency_bucket_count),
          max_peaks_(max_peaks), raw_counts_(nseries, 0),
          series_groups_(nseries) {}

    void add(FfaBatchPeak peak) {
      const std::size_t series_index = peak.series_index;
      const std::uint64_t bucket = static_cast<std::uint64_t>(std::floor(
          peak.peak.frequency / frequency_tolerance_hz_));
      const std::uint64_t key =
          static_cast<std::uint64_t>(series_index) *
              frequency_bucket_count_ + bucket;
      auto [group_it, inserted] = groups_.try_emplace(key);
      auto& values = group_it->second;
      if (inserted) {
        series_groups_[series_index].insert(key);
      }
      values.push_back(std::move(peak));
      std::sort(values.begin(), values.end(),
                [](const FfaBatchPeak& lhs, const FfaBatchPeak& rhs) {
                  return is_better_ffa_peak(lhs.peak, rhs.peak);
                });
      if (values.size() > options_.top_k_per_group) {
        values.resize(options_.top_k_per_group);
      }
      trim_series(series_index);
    }

    void add_raw_counts(std::span<const std::size_t> counts) {
      if (counts.size() != raw_counts_.size()) {
        throw std::logic_error(
            "CUDA FFA reduction raw count series size mismatch");
      }
      for (std::size_t index = 0; index < counts.size(); ++index) {
        raw_counts_[index] = checked_add(
            raw_counts_[index], counts[index],
            "CUDA FFA reduction raw peak count overflow");
        if (max_peaks_ != 0 && raw_counts_[index] > max_peaks_) {
          throw std::runtime_error(
              "CUDA FFA detection peak count exceeded max_peaks safety guard");
        }
      }
    }

    void finish(FfaBatchSearchResult& result) {
      if (groups_discarded_) {
        result.complete = false;
        result.warnings.push_back(
            "Native CUDA reduction discarded frequency groups because "
            "max_groups_per_series was exceeded");
      }
      std::vector<std::vector<std::vector<FfaBatchPeak>>> per_series(
          nseries_);
      for (auto& entry : groups_) {
        auto& values = entry.second;
        if (!values.empty()) {
          per_series[values.front().series_index].push_back(std::move(values));
        }
      }
      for (auto& groups : per_series) {
        std::sort(groups.begin(), groups.end(),
                  [](const auto& lhs, const auto& rhs) {
                    return is_better_ffa_peak(lhs.front().peak,
                                              rhs.front().peak);
                  });
        if (groups.size() > options_.max_groups_per_series) {
          result.complete = false;
          groups.resize(options_.max_groups_per_series);
        }
        for (auto& group : groups) {
          result.peaks.insert(result.peaks.end(), group.begin(), group.end());
        }
      }
      std::sort(result.peaks.begin(), result.peaks.end(),
                [](const FfaBatchPeak& lhs, const FfaBatchPeak& rhs) {
                  if (lhs.series_index != rhs.series_index) {
                    return lhs.series_index < rhs.series_index;
                  }
                  return is_better_ffa_peak(lhs.peak, rhs.peak);
                });
    }

   private:
    void trim_series(std::size_t series_index) {
      auto& keys = series_groups_[series_index];
      while (keys.size() > options_.max_groups_per_series) {
        auto worst = keys.begin();
        auto candidate = keys.begin();
        ++candidate;
        for (; candidate != keys.end(); ++candidate) {
          const auto& worst_values = groups_.at(*worst);
          const auto& candidate_values = groups_.at(*candidate);
          if (is_better_ffa_peak(worst_values.front().peak,
                                 candidate_values.front().peak) ||
              (!is_better_ffa_peak(candidate_values.front().peak,
                                   worst_values.front().peak) &&
               *candidate > *worst)) {
            worst = candidate;
          }
        }
        groups_.erase(*worst);
        keys.erase(worst);
        groups_discarded_ = true;
      }
    }

    std::size_t nseries_ = 0;
    PeakReductionOptions options_{};
    double frequency_tolerance_hz_ = 0.0;
    std::uint64_t frequency_bucket_count_ = 0;
    std::size_t max_peaks_ = 0;
    std::vector<std::size_t> raw_counts_;
    std::unordered_map<std::uint64_t, std::vector<FfaBatchPeak>> groups_;
    std::vector<std::unordered_set<std::uint64_t>> series_groups_;
    bool groups_discarded_ = false;
  };

  struct CudaFfaGroupPeakSummary {
    std::size_t count = 0;
    bool overflow = false;
  };

  void reset_group_peak_buffer() {
    check_cuda(cudaMemsetAsync(workspace_.detection_peak_count.data(), 0,
                               sizeof(unsigned long long),
                               program_.impl_->execution_options.stream),
               "CUDA FFA peak count reset");
    check_cuda(cudaMemsetAsync(workspace_.detection_peak_overflow.data(), 0,
                               sizeof(unsigned int),
                               program_.impl_->execution_options.stream),
               "CUDA FFA peak overflow reset");
  }

  void prepare_group_input(std::size_t group_index,
                           CudaTimeSeriesBatchView input,
                           const CudaLaunchOptions& launch) {
    const auto& group = program_.execution_plan().groups()[group_index];
    const auto prepared = workspace_.prepared_span(
        input.nseries, group.prepared_nsamples, program_.device_id());
    prepare_ffa_input_cuda(input, group.tasks.front().task, prepared, launch);
  }

  std::size_t choose_task_batch_end(const CudaFfaPrepareGroup& group,
                                    std::size_t begin,
                                    std::size_t nseries) const {
    const std::size_t per_series = workspace_.peak_capacity_records() / nseries;
    std::size_t slots = 0;
    std::size_t end = begin;
    while (end < group.tasks.size()) {
      const std::size_t next = checked_add(
          slots, group.tasks[end].detection_slots_per_series,
          "CUDA FFA task batch slot count overflow");
      if (end != begin && next > per_series) {
        break;
      }
      slots = next;
      ++end;
      if (next > per_series) {
        break;
      }
    }
    return std::max(end, begin + 1);
  }

  CudaFfaGroupPeakSummary execute_task_batch(
      std::size_t group_index,
      CudaTimeSeriesBatchView input,
      const CudaLaunchOptions& launch,
      std::size_t task_begin,
      std::size_t task_end) {
    const auto& group = program_.execution_plan().groups()[group_index];
    reset_group_peak_buffer();

    for (std::size_t task_index = task_begin; task_index < task_end;
         ++task_index) {
      const auto& task = group.tasks[task_index];
      const CudaFfaInput prepared_input = workspace_.prepared_input(
          input.nseries, group, task, program_.device_id());
      CudaFfaBuffer scratch = workspace_.transform_buffer(
          workspace_.scratch, input.nseries, task, program_.device_id());
      CudaFfaBuffer output = workspace_.transform_buffer(
          workspace_.output, input.nseries, task, program_.device_id());
      const auto& program_task =
          program_.impl_->layout.groups[group_index].tasks[task_index];

      // One stream orders prepare -> transform and every successive task.
      // A terminal merge can feed detection directly, avoiding one full
      // materialized transform write/read pair.
      const bool fuse_terminal =
          can_fuse_terminal_merge(program_task, program_.impl_->layout);
      if (fuse_terminal) {
        launch_program_transform_before_terminal(
            program_task, program_.impl_->layout, program_.impl_->ops,
            prepared_input, scratch, output, launch);
      } else {
        ffa_transform_block_cuda(program_, group_index, task_index,
                                 prepared_input, scratch, output, launch);
      }
      enqueue_task_detection(group_index, task_index, input.nseries,
                             prepared_input, scratch, output, fuse_terminal);
    }

    check_cuda(cudaStreamSynchronize(program_.impl_->execution_options.stream),
               "CUDA FFA group synchronize");
    unsigned long long count = 0;
    unsigned int overflow = 0;
    check_cuda(cudaMemcpy(&count, workspace_.detection_peak_count.data(),
                          sizeof(count), cudaMemcpyDeviceToHost),
               "CUDA FFA peak count D2H");
    check_cuda(cudaMemcpy(&overflow, workspace_.detection_peak_overflow.data(),
                          sizeof(overflow), cudaMemcpyDeviceToHost),
               "CUDA FFA peak overflow D2H");
    if (count > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("CUDA FFA peak count exceeds host size_t");
    }
    return CudaFfaGroupPeakSummary{
        .count = static_cast<std::size_t>(count),
        .overflow = overflow != 0 || count > workspace_.peak_capacity_records(),
    };
  }

  void collect_reduced_peaks(std::size_t group_index,
                             std::size_t nseries,
                             std::size_t selected_count,
                             NativePeakAccumulator& accumulator) {
    std::vector<FfaCudaPeak> compact_peaks(selected_count);
    if (!compact_peaks.empty()) {
      check_cuda(cudaMemcpy(compact_peaks.data(),
                            workspace_.reduction.output().data(),
                            compact_peaks.size() * sizeof(FfaCudaPeak),
                            cudaMemcpyDeviceToHost),
                 "CUDA FFA reduced peaks D2H");
    }
    const auto& group = program_.execution_plan().groups()[group_index];
    for (const FfaCudaPeak& compact_peak : compact_peaks) {
      if (compact_peak.task_index >= group.tasks.size()) {
        throw std::logic_error("CUDA FFA reduced peak task index is invalid");
      }
      const auto& task = group.tasks[compact_peak.task_index];
      if (compact_peak.series_index >= nseries ||
          compact_peak.shift >= task.task.rows_eval ||
          compact_peak.width_index >= task.detection_plan.boxcar_trials.size() ||
          !std::isfinite(compact_peak.snr)) {
        throw std::logic_error("CUDA FFA reduced peak metadata is invalid");
      }
      const FfaBoxcarTrial& trial = task.detection_plan.boxcar_trials[
          compact_peak.width_index];
      accumulator.add(FfaBatchPeak{
          .series_index = compact_peak.series_index,
          .peak = make_ffa_peak(task.task, trial, compact_peak.shift,
                                compact_peak.phase, compact_peak.snr),
      });
    }
  }

  void grow_peak_buffer(std::size_t group_index, std::size_t required_records) {
    const auto& options = program_.impl_->execution_options;
    // Detection appends results from every task in a prepare group to the
    // same compact buffer. The plan-wide maximum is a safe capacity for any
    // group and must not be reduced to one task's slot count.
    const std::size_t plan_records = checked_multiply(
        workspace_.shape.series_tile_size,
        workspace_.shape.max_detection_slots_per_series,
        "CUDA FFA peak record count overflow");
    const std::size_t max_records = std::min(
        plan_records, options.max_peak_buffer_bytes / sizeof(FfaCudaPeak));
    const std::size_t current_records = workspace_.peak_capacity_records();
    if (required_records > max_records) {
      throw CudaFfaCapacityError(
          "CUDA FFA compact peak buffer exceeded max_peak_buffer_bytes: "
          "group_index=" + std::to_string(group_index) +
          " observed_peak_records=" + std::to_string(required_records) +
          " required_bytes=" +
          std::to_string(checked_multiply(required_records,
                                          sizeof(FfaCudaPeak),
                                          "CUDA FFA peak byte size overflow")) +
          " max_peak_buffer_bytes=" +
          std::to_string(options.max_peak_buffer_bytes));
    }
    const std::size_t doubled_records = checked_multiply(
        current_records, 2, "CUDA FFA peak buffer capacity overflow");
    const std::size_t next_records =
        std::min(max_records, std::max(doubled_records, required_records));
    if (next_records <= current_records) {
      throw CudaFfaCapacityError(
          "CUDA FFA peak buffer growth made no progress");
    }
    const std::size_t next_bytes = checked_multiply(
        next_records, sizeof(FfaCudaPeak),
        "CUDA FFA peak buffer byte size overflow");
    if (options.workspace_bytes_limit != 0 &&
        checked_add(workspace_.shape.total_bytes -
                        workspace_.shape.detection_compact_bytes,
                    next_bytes, "CUDA FFA workspace byte size overflow") >
            options.workspace_bytes_limit) {
      throw CudaFfaCapacityError(
          "CUDA FFA compact peak buffer growth exceeds workspace_bytes_limit");
    }
    workspace_.resize_peak_buffer(
        next_records, program_.execution_plan(), program_.impl_->execution_options,
        program_.device_id());
  }

  void enqueue_task_detection(std::size_t group_index,
                              std::size_t task_index,
                              std::size_t nseries,
                              CudaFfaInput input,
                              CudaFfaBuffer scratch,
                              CudaFfaBuffer transform,
                              bool fuse_terminal) {
    const auto& program_group = program_.impl_->layout.groups[group_index];
    const auto& program_task = program_group.tasks[task_index];
    const auto& task =
        program_.execution_plan().groups()[group_index].tasks[task_index];
    if (task_index > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA FFA task index overflow");
    }
    if (program_task.detection_trial_count !=
        task.detection_plan.boxcar_trials.size()) {
      throw std::logic_error("CUDA FFA detection metadata mismatch");
    }
    const std::size_t profile_and_prefix_elements =
        fuse_terminal
            ? checked_add(
                  task.shape.bins,
                  checked_add(task.shape.bins,
                              task.detection_plan.max_width + 1,
                              "CUDA FFA terminal detection shared size overflow"),
                  "CUDA FFA terminal detection shared size overflow")
            : task.shape.bins + task.detection_plan.max_width + 1;
    const std::size_t shared_bytes = checked_float_bytes(
        profile_and_prefix_elements,
        "CUDA FFA detection shared-memory size overflow");
    const dim3 grid(
        checked_grid_dim(task.task.rows_eval,
                         "CUDA FFA detection grid x overflow"),
        checked_grid_dim(nseries, "CUDA FFA detection grid y overflow"));
    if (fuse_terminal) {
      const FfaCudaProgramLevel& terminal = program_.impl_->layout.levels[
          program_task.level_begin + program_task.level_count - 1];
      detect_ffa_terminal_merge_kernel<<<
          grid, kFfaDetectionThreads, shared_bytes,
          program_.impl_->execution_options.stream>>>(
          input.data, scratch.data, transform.data, input.stride,
          scratch.stride, transform.stride, task.task.rows_eval,
          task.shape.bins,
          program_.impl_->ops.merge_ops.data() + terminal.merge_offset,
          program_.impl_->ops.detection_trials.data() +
              program_task.detection_trial_offset,
          program_task.detection_trial_count, task.detection_plan.max_width,
          ffa_task_stdnoise(program_.execution_plan().observation(),
                            task.task),
          search_options_.snr_threshold,
          static_cast<std::uint32_t>(task_index),
          workspace_.detection_compact.data(),
          workspace_.detection_compact.size(),
          workspace_.detection_peak_count.data(),
          workspace_.detection_peak_overflow.data());
    } else {
      detect_ffa_rows_kernel<<<grid, kFfaDetectionThreads, shared_bytes,
                               program_.impl_->execution_options.stream>>>(
          transform.data, transform.stride, task.task.rows_eval, task.shape.bins,
          program_.impl_->ops.detection_trials.data() +
              program_task.detection_trial_offset,
          program_task.detection_trial_count, task.detection_plan.max_width,
          ffa_task_stdnoise(program_.execution_plan().observation(),
                            task.task),
          search_options_.snr_threshold,
          static_cast<std::uint32_t>(task_index),
          workspace_.detection_compact.data(),
          workspace_.detection_compact.size(),
          workspace_.detection_peak_count.data(),
          workspace_.detection_peak_overflow.data());
    }
    check_cuda(cudaGetLastError(), "CUDA FFA detection kernel launch");
  }

  void collect_group_peaks(std::size_t group_index,
                           std::size_t nseries,
                           std::size_t selected_count,
                           std::vector<std::size_t>& peak_counts,
                           FfaBatchSearchResult& result) {
    std::vector<FfaCudaPeak> compact_peaks(
        selected_count);
    if (!compact_peaks.empty()) {
      check_cuda(cudaMemcpy(compact_peaks.data(),
                            workspace_.detection_compact.data(),
                            compact_peaks.size() * sizeof(FfaCudaPeak),
                            cudaMemcpyDeviceToHost),
                 "CUDA FFA detection compact peaks D2H");
    }
    for (const FfaCudaPeak& compact_peak : compact_peaks) {
      const auto& group = program_.execution_plan().groups()[group_index];
      if (compact_peak.task_index >= group.tasks.size()) {
        throw std::logic_error("CUDA FFA peak task index is invalid");
      }
      const auto& task = group.tasks[compact_peak.task_index];
      if (compact_peak.series_index >= nseries ||
          compact_peak.shift >= task.task.rows_eval ||
          compact_peak.width_index >= task.detection_plan.boxcar_trials.size() ||
          !std::isfinite(compact_peak.snr)) {
        throw std::logic_error("CUDA FFA detection peak metadata is invalid");
      }
      std::size_t& count = peak_counts[compact_peak.series_index];
      if (search_options_.max_peaks != 0 && count >= search_options_.max_peaks) {
        throw std::runtime_error(
            "CUDA FFA detection peak count exceeded max_peaks safety guard");
      }
      const FfaBoxcarTrial& trial = task.detection_plan.boxcar_trials[
          compact_peak.width_index];
      result.peaks.push_back(FfaBatchPeak{
          .series_index = compact_peak.series_index,
          .peak = make_ffa_peak(task.task, trial, compact_peak.shift,
                                compact_peak.phase, compact_peak.snr),
      });
      ++count;
    }
  }

  void validate_tile(CudaTimeSeriesBatchView input) const {
    if (input.data == nullptr) {
      throw std::invalid_argument("CUDA FFA executor input data must not be null");
    }
    if (input.nseries == 0 || input.nseries > program_.tile_capacity()) {
      throw std::invalid_argument(
          "CUDA FFA tile nseries must be within program tile_capacity");
    }
    if (input.device_id != program_.device_id()) {
      throw std::invalid_argument(
          "CUDA FFA executor input device_id must match program device_id");
    }
    for (const auto& group : program_.execution_plan().groups()) {
      if (group.prepare_key.input_nsamples != input.nsamples) {
        throw std::invalid_argument(
            "CUDA FFA executor tile nsamples must match execution plan");
      }
    }
  }

  CudaFfaProgram& program_;
  FfaSearchOptions search_options_;
  CudaFfaWorkspace& workspace_;
};

}  // namespace detail

FfaBatchSearchResult search_ffa_raw_batch_cuda(
    CudaFfaProgram& program,
    CudaTimeSeriesBatchView batch,
    const FfaSearchOptions& options) {
  detail::CudaFfaTileRunner runner(program, options);
  return runner.run_tile(batch);
}

FfaSearchResult search_ffa_raw_cuda(
    CudaFfaProgram& program,
    CudaSpan<const float> time_series,
    const FfaSearchOptions& options) {
  if (time_series.data == nullptr || time_series.count == 0) {
    throw std::invalid_argument("CUDA FFA time series must not be empty");
  }
  if (time_series.device_id != program.device_id()) {
    throw std::invalid_argument(
        "CUDA FFA time series device_id must match program device_id");
  }
  const FfaBatchSearchResult batch = search_ffa_raw_batch_cuda(
      program,
      CudaTimeSeriesBatchView{
          .data = time_series.data,
          .nseries = 1,
          .nsamples = time_series.count,
          .device_id = time_series.device_id,
      },
      options);
  FfaSearchResult result;
  result.peaks.reserve(batch.peaks.size());
  result.complete = batch.complete;
  result.warnings = std::move(batch.warnings);
  for (const FfaBatchPeak& peak : batch.peaks) {
    if (peak.series_index != 0) {
      throw std::logic_error("CUDA FFA single-series result has invalid index");
    }
    result.peaks.push_back(peak.peak);
  }
  return result;
}

FfaSearchResult search_ffa_raw_cuda(
    CudaSpan<const float> time_series,
    const FfaSearchPlan& plan,
    const FfaSearchOptions& options,
    const CudaFfaProgramOptions& program_options,
    const CudaFfaExecutionOptions& execution_options) {
  if (time_series.data == nullptr || time_series.count == 0) {
    throw std::invalid_argument("CUDA FFA time series must not be empty");
  }
  if (time_series.device_id != program_options.device_id) {
    throw std::invalid_argument(
        "CUDA FFA time series device_id must match program device_id");
  }
  CudaFfaProgram program(plan, program_options, execution_options);
  return search_ffa_raw_cuda(program, time_series, options);
}

SeriesPeaks search_ffa_batch_cuda(
    CudaFfaProgram& program,
    CudaTimeSeriesBatchView batch,
    const FfaSearchOptions& options) {
  const FfaBatchSearchResult raw =
      search_ffa_raw_batch_cuda(program, batch, options);

  std::vector<FfaPeak> ffa_peaks;
  ffa_peaks.reserve(raw.peaks.size());
  for (const FfaBatchPeak& peak : raw.peaks) {
    ffa_peaks.push_back(peak.peak);
  }
  const std::vector<PeriodicPeak> periodic = periodic_peaks_from_ffa(
      ffa_peaks, program.execution_plan().observation());

  SeriesPeaks result;
  result.reserve(raw.peaks.size());
  for (std::size_t index = 0; index < raw.peaks.size(); ++index) {
    result.push_back(SeriesPeak{
        .series_index = raw.peaks[index].series_index,
        .peak = periodic[index],
    });
  }
  return result;
}

std::vector<PeriodicPeak> search_ffa_cuda(
    CudaFfaProgram& program,
    CudaSpan<const float> preprocessed_time_series,
    const FfaSearchOptions& options) {
  const FfaSearchResult raw =
      search_ffa_raw_cuda(program, preprocessed_time_series, options);
  return periodic_peaks_from_ffa(
      raw.peaks, program.execution_plan().observation());
}

std::vector<PeriodicPeak> search_ffa_cuda(
    std::span<const float> preprocessed_time_series,
    const FfaSearchPlan& plan,
    const FfaSearchOptions& options,
    const CudaFfaProgramOptions& program_options,
    const CudaFfaExecutionOptions& execution_options) {
  if (preprocessed_time_series.empty()) {
    throw std::invalid_argument("CUDA FFA time series must not be empty");
  }
  if (preprocessed_time_series.size() != plan.observation.nsamples) {
    throw std::invalid_argument(
        "CUDA FFA time series must match plan observation nsamples");
  }

  CudaDeviceScope device_scope(program_options.device_id);
  CudaFfaProgram program(plan, program_options, execution_options);
  CudaDeviceBuffer<float> device_samples(preprocessed_time_series.size());
  check_cuda(cudaMemcpy(device_samples.data(), preprocessed_time_series.data(),
                        device_samples.bytes(), cudaMemcpyHostToDevice),
             "CUDA FFA host input H2D");

  return search_ffa_cuda(
      program,
      static_cast<const CudaDeviceBuffer<float>&>(device_samples)
          .as_span(program_options.device_id),
      options);
}
