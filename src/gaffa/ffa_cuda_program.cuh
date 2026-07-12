// Cached Program metadata and reusable workspace representation.
struct FfaCudaProgramLevel {
  std::size_t subtree_offset = 0;
  std::size_t subtree_count = 0;
  std::size_t max_subtree_rows = 0;
  std::size_t copy_offset = 0;
  std::size_t copy_count = 0;
  std::size_t max_copy_elements = 0;
  std::size_t merge_offset = 0;
  std::size_t merge_count = 0;
  std::size_t max_merge_rows = 0;
};

struct FfaCudaProgramTask {
  std::size_t level_begin = 0;
  std::size_t level_count = 0;
  std::size_t detection_trial_offset = 0;
  std::size_t detection_trial_count = 0;
};

struct FfaCudaProgramGroup {
  std::vector<FfaCudaProgramTask> tasks;
};

struct FfaCudaProgramLayout {
  std::vector<FfaCudaProgramGroup> groups;
  std::vector<FfaCudaProgramLevel> levels;
};

// Build-only host state. The operation vectors are copied once to device memory
// and intentionally do not remain resident in CudaFfaProgramImpl.
struct FfaCudaProgramBuildState {
  FfaCudaProgramLayout layout;
  std::vector<FfaCudaSubtreeOp> subtree_ops;
  std::vector<FfaCudaSubtreeSchedule> subtree_schedules;
  std::vector<FfaCudaSubtreeLevel> subtree_levels;
  std::vector<FfaCudaCopyOp> subtree_copy_ops;
  std::vector<FfaCudaMergeOp> subtree_merge_ops;
  std::vector<FfaCudaCopyOp> copy_ops;
  std::vector<FfaCudaMergeOp> merge_ops;
  std::vector<FfaCudaBoxcarTrial> detection_trials;
  std::size_t max_subtree_shared_bytes = 0;
};

struct FfaCudaProgramOps {
  CudaDeviceBuffer<FfaCudaSubtreeOp> subtree_ops;
  CudaDeviceBuffer<FfaCudaSubtreeSchedule> subtree_schedules;
  CudaDeviceBuffer<FfaCudaSubtreeLevel> subtree_levels;
  CudaDeviceBuffer<FfaCudaCopyOp> subtree_copy_ops;
  CudaDeviceBuffer<FfaCudaMergeOp> subtree_merge_ops;
  CudaDeviceBuffer<FfaCudaCopyOp> copy_ops;
  CudaDeviceBuffer<FfaCudaMergeOp> merge_ops;
  CudaDeviceBuffer<FfaCudaBoxcarTrial> detection_trials;
};

std::size_t find_or_append_subtree_schedule(
    std::size_t rows,
    FfaCudaProgramBuildState& build_state) {
  for (std::size_t index = 0; index < build_state.subtree_schedules.size();
       ++index) {
    if (build_state.subtree_schedules[index].rows == rows) {
      return index;
    }
  }

  const auto schedule_levels = build_transform_levels(
      FfaCudaBufferRole::Input, FfaCudaBufferRole::Scratch,
      FfaCudaBufferRole::Output, 0, rows, 0);
  const FfaCudaSubtreeSchedule schedule{
      .rows = rows,
      .level_offset = build_state.subtree_levels.size(),
      .level_count = schedule_levels.size(),
  };
  for (const auto& level : schedule_levels) {
    build_state.subtree_levels.push_back(FfaCudaSubtreeLevel{
        .copy_offset = build_state.subtree_copy_ops.size(),
        .copy_count = level.copy_ops.size(),
        .merge_offset = build_state.subtree_merge_ops.size(),
        .merge_count = level.merge_ops.size(),
    });
    build_state.subtree_copy_ops.insert(build_state.subtree_copy_ops.end(),
                                        level.copy_ops.begin(),
                                        level.copy_ops.end());
    build_state.subtree_merge_ops.insert(build_state.subtree_merge_ops.end(),
                                         level.merge_ops.begin(),
                                         level.merge_ops.end());
  }
  build_state.subtree_schedules.push_back(schedule);
  return build_state.subtree_schedules.size() - 1;
}

void append_program_task_layout(const CudaFfaTaskLayout& task_layout,
                                FfaCudaProgramBuildState& build_state,
                                FfaCudaProgramGroup& group,
                                int device_id) {
  FfaCudaProgramTask task{
      .level_begin = build_state.layout.levels.size(),
      .detection_trial_offset = build_state.detection_trials.size(),
  };
  const FfaTransformShape shape = task_layout.shape;
  const std::vector<FfaCudaTransformLevel> levels =
      build_transform_levels(shape,
                             select_shared_subtree_rows(shape.bins, device_id));
  for (const auto& level : levels) {
    const FfaCudaProgramLevel program_level{
        .subtree_offset = build_state.subtree_ops.size(),
        .subtree_count = level.subtree_ops.size(),
        .max_subtree_rows = max_subtree_rows(level),
        .copy_offset = build_state.copy_ops.size(),
        .copy_count = level.copy_ops.size(),
        .max_copy_elements = max_copy_elements(level, shape.bins),
        .merge_offset = build_state.merge_ops.size(),
        .merge_count = level.merge_ops.size(),
        .max_merge_rows = max_merge_rows(level),
    };
    for (auto subtree : level.subtree_ops) {
      subtree.schedule_index =
          find_or_append_subtree_schedule(subtree.rows, build_state);
      build_state.max_subtree_shared_bytes = std::max(
          build_state.max_subtree_shared_bytes,
          subtree_shared_bytes(subtree.rows, shape.bins));
      build_state.subtree_ops.push_back(subtree);
    }
    build_state.copy_ops.insert(build_state.copy_ops.end(),
                                level.copy_ops.begin(), level.copy_ops.end());
    build_state.merge_ops.insert(build_state.merge_ops.end(),
                                 level.merge_ops.begin(), level.merge_ops.end());
    build_state.layout.levels.push_back(program_level);
  }
  task.level_count = build_state.layout.levels.size() - task.level_begin;
  for (const FfaBoxcarTrial& trial : task_layout.detection_plan.boxcar_trials) {
    if (trial.width > std::numeric_limits<std::uint32_t>::max() ||
        trial.width_index > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA FFA detection trial index overflow");
    }
    build_state.detection_trials.push_back(FfaCudaBoxcarTrial{
        .width = static_cast<std::uint32_t>(trial.width),
        .width_index = static_cast<std::uint32_t>(trial.width_index),
        .height = trial.height,
        .baseline = trial.baseline,
    });
  }
  task.detection_trial_count = build_state.detection_trials.size() -
                               task.detection_trial_offset;
  group.tasks.push_back(task);
}

FfaCudaProgramOps materialize_program_ops(
    const FfaCudaProgramBuildState& build_state) {
  FfaCudaProgramOps ops{
      .subtree_ops = CudaDeviceBuffer<FfaCudaSubtreeOp>(
          build_state.subtree_ops.size()),
      .subtree_schedules = CudaDeviceBuffer<FfaCudaSubtreeSchedule>(
          build_state.subtree_schedules.size()),
      .subtree_levels = CudaDeviceBuffer<FfaCudaSubtreeLevel>(
          build_state.subtree_levels.size()),
      .subtree_copy_ops = CudaDeviceBuffer<FfaCudaCopyOp>(
          build_state.subtree_copy_ops.size()),
      .subtree_merge_ops = CudaDeviceBuffer<FfaCudaMergeOp>(
          build_state.subtree_merge_ops.size()),
      .copy_ops = CudaDeviceBuffer<FfaCudaCopyOp>(build_state.copy_ops.size()),
      .merge_ops =
          CudaDeviceBuffer<FfaCudaMergeOp>(build_state.merge_ops.size()),
      .detection_trials = CudaDeviceBuffer<FfaCudaBoxcarTrial>(
          build_state.detection_trials.size()),
  };
  if (!build_state.subtree_ops.empty()) {
    check_cuda(cudaMemcpy(ops.subtree_ops.data(),
                          build_state.subtree_ops.data(),
                          build_state.subtree_ops.size() *
                              sizeof(FfaCudaSubtreeOp),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program subtree ops H2D");
  }
  if (!build_state.subtree_schedules.empty()) {
    check_cuda(cudaMemcpy(ops.subtree_schedules.data(),
                          build_state.subtree_schedules.data(),
                          build_state.subtree_schedules.size() *
                              sizeof(FfaCudaSubtreeSchedule),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program subtree schedules H2D");
  }
  if (!build_state.subtree_levels.empty()) {
    check_cuda(cudaMemcpy(ops.subtree_levels.data(),
                          build_state.subtree_levels.data(),
                          build_state.subtree_levels.size() *
                              sizeof(FfaCudaSubtreeLevel),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program subtree levels H2D");
  }
  if (!build_state.subtree_copy_ops.empty()) {
    check_cuda(cudaMemcpy(ops.subtree_copy_ops.data(),
                          build_state.subtree_copy_ops.data(),
                          build_state.subtree_copy_ops.size() *
                              sizeof(FfaCudaCopyOp),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program subtree copy ops H2D");
  }
  if (!build_state.subtree_merge_ops.empty()) {
    check_cuda(cudaMemcpy(ops.subtree_merge_ops.data(),
                          build_state.subtree_merge_ops.data(),
                          build_state.subtree_merge_ops.size() *
                              sizeof(FfaCudaMergeOp),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program subtree merge ops H2D");
  }
  if (!build_state.copy_ops.empty()) {
    check_cuda(cudaMemcpy(ops.copy_ops.data(), build_state.copy_ops.data(),
                          build_state.copy_ops.size() * sizeof(FfaCudaCopyOp),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program copy ops H2D");
  }
  if (!build_state.merge_ops.empty()) {
    check_cuda(cudaMemcpy(ops.merge_ops.data(), build_state.merge_ops.data(),
                          build_state.merge_ops.size() * sizeof(FfaCudaMergeOp),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program merge ops H2D");
  }
  if (!build_state.detection_trials.empty()) {
    check_cuda(cudaMemcpy(ops.detection_trials.data(),
                          build_state.detection_trials.data(),
                          build_state.detection_trials.size() *
                              sizeof(FfaCudaBoxcarTrial),
                          cudaMemcpyHostToDevice),
               "CUDA FFA program detection trials H2D");
  }
  return ops;
}

void launch_program_subtree_level(const FfaCudaProgramLevel& level,
                                  const FfaCudaProgramOps& ops,
                                  CudaFfaInput input,
                                  CudaFfaBuffer scratch,
                                  CudaFfaBuffer output,
                                  const CudaLaunchOptions& options) {
  if (level.subtree_count == 0) {
    return;
  }
  const std::size_t shared_elements = checked_multiply(
      checked_multiply(level.max_subtree_rows, input.shape.bins,
                       "CUDA FFA subtree shared element count overflow"),
      3, "CUDA FFA subtree shared element count overflow");
  const std::size_t shared_bytes = checked_float_bytes(
      shared_elements, "CUDA FFA subtree shared byte size overflow");
  const dim3 block(static_cast<unsigned int>(options.threads_per_block));
  const dim3 grid(
      checked_grid_dim(level.subtree_count,
                       "CUDA FFA program subtree grid x overflow"),
      checked_grid_dim(input.nseries,
                       "CUDA FFA program subtree grid y overflow"));
  ffa_subtree_transform_kernel<<<grid, block, shared_bytes, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins,
      ops.subtree_ops.data() + level.subtree_offset, level.subtree_count,
      ops.subtree_schedules.data(), ops.subtree_levels.data(),
      ops.subtree_copy_ops.data(), ops.subtree_merge_ops.data(),
      level.max_subtree_rows);
  check_cuda(cudaGetLastError(), "CUDA FFA program subtree kernel launch");
}

void launch_program_copy_level(const FfaCudaProgramLevel& level,
                               const FfaCudaProgramOps& ops,
                               CudaFfaInput input,
                               CudaFfaBuffer scratch,
                               CudaFfaBuffer output,
                               const CudaLaunchOptions& options) {
  if (level.copy_count == 0) {
    return;
  }
  const auto threads = static_cast<unsigned int>(options.threads_per_block);
  const std::size_t x_blocks =
      (level.max_copy_elements + threads - 1) / threads;
  const dim3 block(threads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA program copy grid x overflow"),
      checked_grid_dim(level.copy_count,
                       "CUDA FFA program copy grid y overflow"),
      checked_grid_dim(input.nseries,
                       "CUDA FFA program copy grid z overflow"));
  ffa_copy_level_kernel<<<grid, block, 0, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins,
      ops.copy_ops.data() + level.copy_offset, level.copy_count,
      level.max_copy_elements);
  check_cuda(cudaGetLastError(), "CUDA FFA program copy kernel launch");
}

void launch_program_merge_level(const FfaCudaProgramLevel& level,
                                const FfaCudaProgramOps& ops,
                                CudaFfaInput input,
                                CudaFfaBuffer scratch,
                                CudaFfaBuffer output,
                                const CudaLaunchOptions& options) {
  if (level.merge_count == 0) {
    return;
  }
  const std::size_t x_blocks =
      (level.max_merge_rows + kFfaMergeWarpsPerBlock - 1) /
      kFfaMergeWarpsPerBlock;
  const dim3 block(kFfaMergeThreads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA program merge grid x overflow"),
      checked_grid_dim(level.merge_count,
                       "CUDA FFA program merge grid y overflow"),
      checked_grid_dim(input.nseries,
                       "CUDA FFA program merge grid z overflow"));
  ffa_merge_profiles_kernel<<<grid, block, 0, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins,
      ops.merge_ops.data() + level.merge_offset, level.merge_count,
      level.max_merge_rows);
  check_cuda(cudaGetLastError(), "CUDA FFA program merge kernel launch");
}

void launch_program_transform(const FfaCudaProgramTask& task,
                              const FfaCudaProgramLayout& layout,
                              const FfaCudaProgramOps& ops,
                              CudaFfaInput input,
                              CudaFfaBuffer scratch,
                              CudaFfaBuffer output,
                              const CudaLaunchOptions& options) {
  if (task.level_begin > layout.levels.size() ||
      task.level_count > layout.levels.size() - task.level_begin) {
    throw std::logic_error("CUDA FFA program level layout is invalid");
  }
  for (std::size_t index = 0; index < task.level_count; ++index) {
    const auto& level = layout.levels[task.level_begin + index];
    launch_program_subtree_level(level, ops, input, scratch, output, options);
    launch_program_copy_level(level, ops, input, scratch, output, options);
    launch_program_merge_level(level, ops, input, scratch, output, options);
  }
  if (options.synchronize_after_call) {
    check_cuda(cudaStreamSynchronize(options.stream),
               "CUDA FFA program transform synchronize");
  }
}

bool can_fuse_terminal_merge(const FfaCudaProgramTask& task,
                             const FfaCudaProgramLayout& layout) {
  if (task.level_count == 0 || task.level_begin >= layout.levels.size() ||
      task.level_count > layout.levels.size() - task.level_begin) {
    return false;
  }
  const FfaCudaProgramLevel& terminal =
      layout.levels[task.level_begin + task.level_count - 1];
  return terminal.subtree_count == 0 && terminal.copy_count == 0 &&
         terminal.merge_count == 1;
}

void launch_program_transform_before_terminal(
    const FfaCudaProgramTask& task,
    const FfaCudaProgramLayout& layout,
    const FfaCudaProgramOps& ops,
    CudaFfaInput input,
    CudaFfaBuffer scratch,
    CudaFfaBuffer output,
    const CudaLaunchOptions& options) {
  if (!can_fuse_terminal_merge(task, layout)) {
    throw std::logic_error("CUDA FFA task has no fusable terminal merge");
  }
  for (std::size_t index = 0; index + 1 < task.level_count; ++index) {
    const auto& level = layout.levels[task.level_begin + index];
    launch_program_subtree_level(level, ops, input, scratch, output, options);
    launch_program_copy_level(level, ops, input, scratch, output, options);
    launch_program_merge_level(level, ops, input, scratch, output, options);
  }
}

struct CudaFfaWorkspace {
  CudaFfaWorkspace(const CudaFfaExecutionPlan& plan,
                   const CudaFfaExecutionOptions& options,
                   int device_id)
      : shape(estimate_ffa_cuda_workspace(plan, options)) {
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
    prepared = CudaDeviceBuffer<float>(shape.prepared_bytes / sizeof(float));
    scratch = CudaDeviceBuffer<float>(shape.scratch_bytes / sizeof(float));
    output = CudaDeviceBuffer<float>(shape.output_bytes / sizeof(float));
    detection_compact = CudaDeviceBuffer<FfaCudaPeak>(
        shape.detection_compact_bytes / sizeof(FfaCudaPeak));
    detection_peak_count = CudaDeviceBuffer<unsigned int>(1);
    detection_peak_overflow = CudaDeviceBuffer<unsigned int>(1);
  }

  CudaSpan<float> prepared_span(std::size_t nseries,
                                std::size_t prepared_nsamples,
                                int device_id) {
    const std::size_t count = checked_multiply(
        nseries, prepared_nsamples, "CUDA FFA prepared tile size overflow");
    if (nseries == 0 || nseries > shape.series_tile_size ||
        prepared_nsamples > shape.max_prepared_nsamples ||
        count > prepared.size()) {
      throw std::invalid_argument("CUDA FFA prepared tile exceeds workspace");
    }
    return CudaSpan<float>{
        .data = prepared.data(),
        .count = count,
        .device_id = device_id,
    };
  }

  CudaFfaInput prepared_input(std::size_t nseries,
                              const CudaFfaPrepareGroup& group,
                              const CudaFfaTaskLayout& task,
                              int device_id) {
    (void)prepared_span(nseries, group.prepared_nsamples, device_id);
    return CudaFfaInput{
        .data = prepared.data(),
        .nseries = nseries,
        .nsamples = group.prepared_nsamples,
        .stride = group.prepared_nsamples,
        .shape = task.shape,
        .device_id = device_id,
    };
  }

  CudaFfaBuffer transform_buffer(CudaDeviceBuffer<float>& buffer,
                                 std::size_t nseries,
                                 const CudaFfaTaskLayout& task,
                                 int device_id) {
    const std::size_t count = checked_multiply(
        nseries, task.transform_elements,
        "CUDA FFA transform tile size overflow");
    if (nseries == 0 || nseries > shape.series_tile_size ||
        task.transform_elements > shape.max_task_elements ||
        count > buffer.size()) {
      throw std::invalid_argument("CUDA FFA transform tile exceeds workspace");
    }
    return CudaFfaBuffer{
        .data = buffer.data(),
        .nseries = nseries,
        .stride = task.transform_elements,
        .shape = task.shape,
        .device_id = device_id,
    };
  }

  CudaFfaWorkspaceShape shape;
  CudaDeviceBuffer<float> prepared;
  CudaDeviceBuffer<float> scratch;
  CudaDeviceBuffer<float> output;
  CudaDeviceBuffer<FfaCudaPeak> detection_compact;
  CudaDeviceBuffer<unsigned int> detection_peak_count;
  CudaDeviceBuffer<unsigned int> detection_peak_overflow;
};

}  // namespace
