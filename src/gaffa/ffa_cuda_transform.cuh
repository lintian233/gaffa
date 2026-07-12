// Host-side transform scheduling and launch helpers for the kernels above.
std::size_t checked_multiply(std::size_t lhs,
                             std::size_t rhs,
                             const char* message);

std::size_t checked_multiply(std::size_t lhs,
                             std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs,
                        std::size_t rhs,
                        const char* message) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::overflow_error(message);
  }
  return lhs + rhs;
}

std::size_t checked_float_bytes(std::size_t count, const char* message) {
  return checked_multiply(count, sizeof(float), message);
}

constexpr std::size_t kPreferredSharedSubtreeRows = 16;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::size_t subtree_shared_bytes(std::size_t rows, std::size_t bins) {
  return checked_float_bytes(
      checked_multiply(
          checked_multiply(rows, bins,
                           "CUDA FFA subtree shared element count overflow"),
          3, "CUDA FFA subtree shared element count overflow"),
      "CUDA FFA subtree shared byte size overflow");
}

std::size_t select_shared_subtree_rows(std::size_t bins, int device_id) {
  int optin_shared_bytes = 0;
  check_cuda(cudaDeviceGetAttribute(&optin_shared_bytes,
                                    cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    device_id),
             "cudaDeviceGetAttribute max opt-in shared memory");
  if (optin_shared_bytes <= 0) {
    return 0;
  }

  const std::size_t rows_by_capacity =
      static_cast<std::size_t>(optin_shared_bytes) /
      (3 * bins * sizeof(float));
  if (rows_by_capacity < 2) {
    return 0;
  }
  return std::min(kPreferredSharedSubtreeRows, rows_by_capacity);
}

void configure_subtree_dynamic_shared_memory(std::size_t shared_bytes) {
  if (shared_bytes == 0) {
    return;
  }
  if (shared_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("CUDA FFA subtree shared memory size overflow");
  }
  check_cuda(cudaFuncSetAttribute(
                 ffa_subtree_transform_kernel,
                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                 static_cast<int>(shared_bytes)),
             "cudaFuncSetAttribute CUDA FFA subtree shared memory");
}

void validate_program_options(const CudaFfaProgramOptions& options) {
  if (options.device_id < 0) {
    throw std::invalid_argument("CUDA FFA program device_id must be >= 0");
  }
}

void validate_execution_options(const CudaFfaExecutionOptions& options) {
  if (options.series_tile_size == 0) {
    throw std::invalid_argument("CUDA FFA series_tile_size must be > 0");
  }
  if (options.threads_per_block == 0) {
    throw std::invalid_argument("CUDA FFA threads_per_block must be > 0");
  }
  if (options.peak_buffer_bytes == 0) {
    throw std::invalid_argument("CUDA FFA peak_buffer_bytes must be > 0");
  }
}

void validate_launch_options(const CudaLaunchOptions& options) {
  if (options.device_id < 0) {
    throw std::invalid_argument("CUDA FFA launch device_id must be >= 0");
  }
  if (options.threads_per_block == 0) {
    throw std::invalid_argument(
        "CUDA FFA launch threads_per_block must be > 0");
  }
}

unsigned int checked_grid_dim(std::size_t value, const char* message) {
  if (value == 0 ||
      value > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::overflow_error(message);
  }
  return static_cast<unsigned int>(value);
}

bool is_no_downsample(double factor) {
  return factor == 1.0;
}

void append_level(FfaCudaTransformLevel& output,
                  const FfaCudaTransformLevel& input) {
  output.subtree_ops.insert(output.subtree_ops.end(), input.subtree_ops.begin(),
                            input.subtree_ops.end());
  output.copy_ops.insert(output.copy_ops.end(), input.copy_ops.begin(),
                         input.copy_ops.end());
  output.merge_ops.insert(output.merge_ops.end(), input.merge_ops.begin(),
                          input.merge_ops.end());
}

std::vector<FfaCudaTransformLevel> build_transform_levels(
    FfaCudaBufferRole input_role,
    FfaCudaBufferRole scratch_role,
    FfaCudaBufferRole output_role,
    std::size_t row_begin,
    std::size_t rows,
    std::size_t subtree_rows) {
  if (subtree_rows != 0 && rows <= subtree_rows) {
    FfaCudaTransformLevel level;
    level.subtree_ops.push_back(FfaCudaSubtreeOp{
        .input_role = input_role,
        .output_role = output_role,
        .input_begin_row = row_begin,
        .output_begin_row = row_begin,
        .rows = rows,
    });
    return {level};
  }
  if (rows == 1) {
    FfaCudaTransformLevel level;
    level.copy_ops.push_back(FfaCudaCopyOp{
        .input_role = input_role,
        .output_role = output_role,
        .input_begin_row = row_begin,
        .output_begin_row = row_begin,
        .rows = 1,
    });
    return {level};
  }

  if (rows == 2) {
    FfaCudaTransformLevel level;
    level.merge_ops.push_back(FfaCudaMergeOp{
        .head_role = input_role,
        .tail_role = input_role,
        .output_role = output_role,
        .head_begin_row = row_begin,
        .tail_begin_row = row_begin + 1,
        .output_begin_row = row_begin,
        .head_rows = 1,
        .tail_rows = 1,
        .output_rows = 2,
    });
    return {level};
  }

  const std::size_t head_rows = rows / 2;
  const std::size_t tail_rows = rows - head_rows;
  auto head_levels = build_transform_levels(
      input_role, output_role, scratch_role, row_begin, head_rows,
      subtree_rows);
  auto tail_levels = build_transform_levels(
      input_role, output_role, scratch_role, row_begin + head_rows, tail_rows,
      subtree_rows);

  std::vector<FfaCudaTransformLevel> levels(
      std::max(head_levels.size(), tail_levels.size()));
  for (std::size_t index = 0; index < head_levels.size(); ++index) {
    append_level(levels[index], head_levels[index]);
  }
  for (std::size_t index = 0; index < tail_levels.size(); ++index) {
    append_level(levels[index], tail_levels[index]);
  }

  FfaCudaTransformLevel merge_level;
  merge_level.merge_ops.push_back(FfaCudaMergeOp{
      .head_role = scratch_role,
      .tail_role = scratch_role,
      .output_role = output_role,
      .head_begin_row = row_begin,
      .tail_begin_row = row_begin + head_rows,
      .output_begin_row = row_begin,
      .head_rows = head_rows,
      .tail_rows = tail_rows,
      .output_rows = rows,
  });
  levels.push_back(std::move(merge_level));
  return levels;
}

std::vector<FfaCudaTransformLevel> build_transform_levels(
    FfaTransformShape shape,
    std::size_t subtree_rows = 0) {
  return build_transform_levels(FfaCudaBufferRole::Input,
                                FfaCudaBufferRole::Scratch,
                                FfaCudaBufferRole::Output, 0, shape.rows,
                                subtree_rows);
}

std::size_t max_copy_elements(const FfaCudaTransformLevel& level,
                              std::size_t bins) {
  std::size_t max_elements = 0;
  for (const auto& op : level.copy_ops) {
    max_elements = std::max(max_elements, checked_multiply(
                                              op.rows, bins,
                                              "CUDA FFA copy op size overflow"));
  }
  return max_elements;
}

std::size_t max_subtree_rows(const FfaCudaTransformLevel& level) {
  std::size_t max_rows = 0;
  for (const auto& op : level.subtree_ops) {
    max_rows = std::max(max_rows, op.rows);
  }
  return max_rows;
}

std::size_t max_merge_rows(const FfaCudaTransformLevel& level) {
  std::size_t max_rows = 0;
  for (const auto& op : level.merge_ops) {
    max_rows = std::max(max_rows, op.output_rows);
  }
  return max_rows;
}

void validate_task_prepared_nsamples(const FfaSearchTask& task,
                                     const char* prefix) {
  if (task.input_nsamples == 0) {
    throw std::invalid_argument(std::string(prefix) +
                                " task input_nsamples must be > 0");
  }
  if (task.prepared_nsamples == 0) {
    throw std::invalid_argument(std::string(prefix) +
                                " task prepared_nsamples must be > 0");
  }

  if (is_no_downsample(task.downsample_factor)) {
    if (task.prepared_nsamples != task.input_nsamples) {
      throw std::invalid_argument(
          std::string(prefix) +
          " no-downsample task prepared_nsamples must match input nsamples");
    }
    return;
  }

  const std::size_t expected_prepared =
      downsampled_size(task.input_nsamples, task.downsample_factor);
  if (task.prepared_nsamples != expected_prepared) {
    throw std::invalid_argument(
        std::string(prefix) +
        " task prepared_nsamples must match downsampled_size");
  }
}

void validate_plan_for_workspace(const FfaSearchPlan& plan) {
  if (plan.tasks.empty()) {
    throw std::invalid_argument("CUDA FFA plan must contain at least one task");
  }
  if (plan.width_trials.empty()) {
    throw std::invalid_argument("CUDA FFA plan width_trials must not be empty");
  }
  for (const auto& task : plan.tasks) {
    if (task.bins <= 1) {
      throw std::invalid_argument("CUDA FFA task bins must be > 1");
    }
    if (task.rows == 0) {
      throw std::invalid_argument("CUDA FFA task rows must be > 0");
    }
    if (task.rows_eval == 0 || task.rows_eval > task.rows) {
      throw std::invalid_argument(
          "CUDA FFA task rows_eval must satisfy 0 < rows_eval <= rows");
    }
    if (task.prepared_nsamples == 0) {
      throw std::invalid_argument(
          "CUDA FFA task prepared_nsamples must be > 0");
    }
    const std::size_t task_elements = checked_multiply(
        task.rows, task.bins, "CUDA FFA task shape size overflow");
    if (task_elements > task.prepared_nsamples) {
      throw std::invalid_argument(
          "CUDA FFA task rows * bins must be <= prepared_nsamples");
    }
    if (!(task.effective_tsamp > 0.0) || !std::isfinite(task.effective_tsamp)) {
      throw std::invalid_argument(
          "CUDA FFA task effective_tsamp must be finite and > 0");
    }
    if (!std::isfinite(task.downsample_factor) ||
        task.downsample_factor < 1.0) {
      throw std::invalid_argument(
          "CUDA FFA task downsample_factor must be finite and >= 1");
    }
    validate_task_prepared_nsamples(task, "CUDA FFA");
  }
}

void validate_search_options(const FfaSearchOptions& search_options) {
  if (!std::isfinite(search_options.snr_threshold)) {
    throw std::invalid_argument("CUDA FFA S/N threshold must be finite");
  }
}

void validate_prepare_arguments(CudaTimeSeriesBatchView input,
                                const FfaSearchTask& task,
                                CudaSpan<float> output,
                                const CudaLaunchOptions& options) {
  if (input.data == nullptr) {
    throw std::invalid_argument("CUDA FFA prepare input data must not be null");
  }
  if (input.nseries == 0) {
    throw std::invalid_argument("CUDA FFA prepare input nseries must be > 0");
  }
  if (input.nsamples == 0) {
    throw std::invalid_argument("CUDA FFA prepare input nsamples must be > 0");
  }
  validate_launch_options(options);
  if (input.device_id != options.device_id ||
      output.device_id != options.device_id) {
    throw std::invalid_argument(
        "CUDA FFA prepare device ids must match launch device_id");
  }
  if (output.data == nullptr) {
    throw std::invalid_argument("CUDA FFA prepare output data must not be null");
  }
  if (task.input_nsamples != input.nsamples) {
    throw std::invalid_argument(
        "CUDA FFA prepare task input_nsamples must match input nsamples");
  }
  if (!std::isfinite(task.downsample_factor) ||
      task.downsample_factor < 1.0) {
    throw std::invalid_argument(
        "CUDA FFA prepare downsample_factor must be finite and >= 1");
  }
  validate_task_prepared_nsamples(task, "CUDA FFA prepare");
  const std::size_t expected_output = checked_multiply(
      input.nseries, task.prepared_nsamples,
      "CUDA FFA prepare output element count overflow");
  if (output.count != expected_output) {
    throw std::invalid_argument(
        "CUDA FFA prepare output size must match nseries * prepared_nsamples");
  }
}

std::size_t validate_transform_arguments(
    CudaFfaInput input,
    CudaFfaBuffer scratch,
    CudaFfaBuffer output,
    const CudaLaunchOptions& options) {
  if (input.data == nullptr) {
    throw std::invalid_argument("CUDA FFA transform input data must not be null");
  }
  if (input.nseries == 0) {
    throw std::invalid_argument("CUDA FFA transform nseries must be > 0");
  }
  if (input.nsamples == 0) {
    throw std::invalid_argument(
        "CUDA FFA transform input nsamples must be > 0");
  }
  if (input.stride < input.nsamples) {
    throw std::invalid_argument(
        "CUDA FFA transform input stride must be >= nsamples");
  }
  if (input.shape.rows == 0) {
    throw std::invalid_argument("CUDA FFA transform rows must be > 0");
  }
  if (input.shape.bins <= 1) {
    throw std::invalid_argument("CUDA FFA transform bins must be > 1");
  }
  validate_launch_options(options);
  if (input.device_id != options.device_id ||
      scratch.device_id != options.device_id ||
      output.device_id != options.device_id) {
    throw std::invalid_argument(
        "CUDA FFA transform device ids must match launch device_id");
  }
  if (scratch.nseries != input.nseries || output.nseries != input.nseries) {
    throw std::invalid_argument(
        "CUDA FFA transform scratch/output nseries must match input nseries");
  }
  if (scratch.shape.rows != input.shape.rows ||
      scratch.shape.bins != input.shape.bins ||
      output.shape.rows != input.shape.rows ||
      output.shape.bins != input.shape.bins) {
    throw std::invalid_argument(
        "CUDA FFA transform scratch/output shapes must match input shape");
  }
  if (scratch.data == nullptr) {
    throw std::invalid_argument(
        "CUDA FFA transform scratch data must not be null");
  }
  if (output.data == nullptr) {
    throw std::invalid_argument(
        "CUDA FFA transform output data must not be null");
  }
  if (input.data == scratch.data || input.data == output.data ||
      scratch.data == output.data) {
    throw std::invalid_argument(
        "CUDA FFA transform input, scratch, and output must not alias");
  }

  const std::size_t series_elements =
      checked_multiply(input.shape.rows, input.shape.bins,
                       "CUDA FFA transform series element count overflow");
  if (series_elements > input.nsamples) {
    throw std::invalid_argument(
        "CUDA FFA transform rows * bins must be <= input nsamples");
  }
  if (scratch.stride < series_elements) {
    throw std::invalid_argument(
        "CUDA FFA transform scratch stride is too small");
  }
  if (output.stride < series_elements) {
    throw std::invalid_argument("CUDA FFA transform output stride is too small");
  }
  return series_elements;
}

void validate_transform_matches_task(CudaFfaInput input,
                                     CudaFfaBuffer scratch,
                                     CudaFfaBuffer output,
                                     const CudaFfaTaskLayout& task) {
  if (input.shape.rows != task.shape.rows || input.shape.bins != task.shape.bins ||
      scratch.shape.rows != task.shape.rows ||
      scratch.shape.bins != task.shape.bins ||
      output.shape.rows != task.shape.rows ||
      output.shape.bins != task.shape.bins) {
    throw std::invalid_argument(
        "CUDA FFA program transform shapes must match selected task");
  }
  if (input.nsamples < task.transform_elements) {
    throw std::invalid_argument(
        "CUDA FFA program transform input nsamples is too small");
  }
}

void launch_copy_level(const FfaCudaTransformLevel& level,
                       const FfaCudaDeviceTransformLevel& device_level,
                       CudaFfaInput input,
                       CudaFfaBuffer scratch,
                       CudaFfaBuffer output,
                       const CudaLaunchOptions& options) {
  if (level.copy_ops.empty()) {
    return;
  }

  const auto threads = static_cast<unsigned int>(options.threads_per_block);
  const std::size_t x_blocks =
      (device_level.max_copy_elements + threads - 1) / threads;
  const dim3 block(threads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA copy grid x overflow"),
      checked_grid_dim(level.copy_ops.size(), "CUDA FFA copy grid y overflow"),
      checked_grid_dim(input.nseries, "CUDA FFA copy grid z overflow"));
  ffa_copy_level_kernel<<<grid, block, 0, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins, device_level.copy_ops.data(),
      device_level.copy_op_count, device_level.max_copy_elements);
  check_cuda(cudaGetLastError(), "ffa_transform_block_cuda copy kernel launch");
}

void launch_merge_level(const FfaCudaTransformLevel& level,
                        const FfaCudaDeviceTransformLevel& device_level,
                        CudaFfaInput input,
                        CudaFfaBuffer scratch,
                        CudaFfaBuffer output,
                        const CudaLaunchOptions& options) {
  if (level.merge_ops.empty()) {
    return;
  }

  const std::size_t x_blocks =
      (device_level.max_merge_rows + kFfaMergeWarpsPerBlock - 1) /
      kFfaMergeWarpsPerBlock;
  const dim3 block(kFfaMergeThreads);
  const dim3 grid(
      checked_grid_dim(x_blocks, "CUDA FFA merge grid x overflow"),
      checked_grid_dim(level.merge_ops.size(), "CUDA FFA merge grid y overflow"),
      checked_grid_dim(input.nseries, "CUDA FFA merge grid z overflow"));
  ffa_merge_profiles_kernel<<<grid, block, 0, options.stream>>>(
      input.data, scratch.data, output.data, input.stride, scratch.stride,
      output.stride, input.shape.bins, device_level.merge_ops.data(),
      device_level.merge_op_count, device_level.max_merge_rows);
  check_cuda(cudaGetLastError(), "ffa_transform_block_cuda merge kernel launch");
}

std::vector<FfaCudaDeviceTransformLevel> materialize_device_levels(
    const std::vector<FfaCudaTransformLevel>& levels,
    FfaTransformShape shape) {
  std::vector<FfaCudaDeviceTransformLevel> device_levels;
  device_levels.reserve(levels.size());
  for (const auto& level : levels) {
    FfaCudaDeviceTransformLevel device_level{
        .copy_ops = CudaDeviceBuffer<FfaCudaCopyOp>(level.copy_ops.size()),
        .copy_op_count = level.copy_ops.size(),
        .max_copy_elements = max_copy_elements(level, shape.bins),
        .merge_ops = CudaDeviceBuffer<FfaCudaMergeOp>(level.merge_ops.size()),
        .merge_op_count = level.merge_ops.size(),
        .max_merge_rows = max_merge_rows(level),
    };
    if (!level.copy_ops.empty()) {
      check_cuda(cudaMemcpy(device_level.copy_ops.data(), level.copy_ops.data(),
                            level.copy_ops.size() * sizeof(FfaCudaCopyOp),
                            cudaMemcpyHostToDevice),
                 "ffa_transform_block_cuda copy ops H2D");
    }
    if (!level.merge_ops.empty()) {
      check_cuda(cudaMemcpy(device_level.merge_ops.data(),
                            level.merge_ops.data(),
                            level.merge_ops.size() * sizeof(FfaCudaMergeOp),
                            cudaMemcpyHostToDevice),
                 "ffa_transform_block_cuda merge ops H2D");
    }
    device_levels.push_back(std::move(device_level));
  }
  return device_levels;
}

void launch_transform_levels(
    const std::vector<FfaCudaTransformLevel>& levels,
    const std::vector<FfaCudaDeviceTransformLevel>& device_levels,
    CudaFfaInput input,
    CudaFfaBuffer scratch,
    CudaFfaBuffer output,
    const CudaLaunchOptions& options) {
  if (levels.size() != device_levels.size()) {
    throw std::invalid_argument(
        "CUDA FFA transform level metadata mismatch");
  }
  for (std::size_t index = 0; index < levels.size(); ++index) {
    launch_copy_level(levels[index], device_levels[index], input, scratch,
                      output, options);
    launch_merge_level(levels[index], device_levels[index], input, scratch,
                       output, options);
  }
  if (options.synchronize_after_call) {
    check_cuda(cudaStreamSynchronize(options.stream),
               "ffa_transform_block_cuda synchronize");
  }
}
