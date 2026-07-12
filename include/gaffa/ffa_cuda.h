#pragma once

#include "gaffa/cuda_memory.h"
#include "gaffa/ffa_detection.h"
#include "gaffa/ffa.h"
#include "gaffa/ffa_search.h"
#include "gaffa/launch_cuda.h"
#include "gaffa/time_series_cuda.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace gaffa {

namespace detail {
class CudaFfaTileRunner;
}

// Construction-time properties of a device-affine CUDA FFA program.
struct CudaFfaProgramOptions {
  int device_id = 0;
};

// Scheduling and temporary-storage policy fixed for one CudaFfaProgram.
struct CudaFfaExecutionOptions {
  // Number of independent time series processed together. In DM search this is
  // usually the number of DMs in one tile, but FFA itself only sees a series
  // batch.
  std::size_t series_tile_size = 16;

  std::size_t threads_per_block = 256;

  // 0 means auto. Non-zero limits temporary CUDA workspace allocated by the
  // FFA executor.
  std::size_t workspace_bytes_limit = 0;

  cudaStream_t stream = nullptr;

  [[nodiscard]] CudaLaunchOptions async_launch_options(
      int device_id) const noexcept {
    return CudaLaunchOptions{
        .device_id = device_id,
        .threads_per_block = threads_per_block,
        .stream = stream,
        .synchronize_after_call = false,
    };
  }
};

// A significant FFA peak with the identity of its source series in a CUDA
// tile. The index is tile-local; the caller may map it to DM or another
// domain-specific coordinate.
struct FfaBatchPeak {
  std::size_t series_index = 0;
  FfaPeak peak{};
};

struct FfaBatchSearchResult {
  std::vector<FfaBatchPeak> peaks;
};

struct CudaFfaWorkspaceShape {
  std::size_t series_tile_size = 0;
  std::size_t max_prepared_nsamples = 0;
  std::size_t max_task_elements = 0;
  std::size_t max_detection_slots_per_series = 0;
  std::size_t prepared_bytes = 0;
  std::size_t scratch_bytes = 0;
  std::size_t output_bytes = 0;
  std::size_t detection_raw_bytes = 0;
  std::size_t detection_compact_bytes = 0;
  std::size_t detection_flags_bytes = 0;
  std::size_t detection_select_temp_bytes = 0;
  std::size_t total_bytes = 0;
};

struct CudaFfaPrepareKey {
  std::size_t input_nsamples = 0;
  double downsample_factor = 1.0;
};

struct CudaFfaTaskLayout {
  FfaSearchTask task{};
  FfaTransformShape shape{};
  FfaDetectionPlan detection_plan{};
  std::size_t transform_elements = 0;
  std::size_t detection_slots_per_series = 0;
};

// Tasks in one group consume the same prepared time-series batch. They may
// differ in FFA transform shape, but must share the complete prepare key and
// exact prepared length.
struct CudaFfaPrepareGroup {
  CudaFfaPrepareKey prepare_key{};
  std::size_t prepared_nsamples = 0;
  std::vector<CudaFfaTaskLayout> tasks;
};

// Immutable CUDA execution layout derived from a logical FFA search plan.
// Workspace sizing must use this object rather than independently traversing
// FfaSearchPlan, so the layout and its cached maxima cannot drift apart.
class CudaFfaExecutionPlan {
 public:
  CudaFfaExecutionPlan(const CudaFfaExecutionPlan&) = default;
  CudaFfaExecutionPlan& operator=(const CudaFfaExecutionPlan&) = default;
  CudaFfaExecutionPlan(CudaFfaExecutionPlan&&) noexcept = default;
  CudaFfaExecutionPlan& operator=(CudaFfaExecutionPlan&&) noexcept = default;

  [[nodiscard]] std::span<const CudaFfaPrepareGroup> groups() const noexcept;
  [[nodiscard]] std::size_t max_prepared_nsamples() const noexcept;
  [[nodiscard]] std::size_t max_transform_elements() const noexcept;
  [[nodiscard]] std::size_t max_detection_slots_per_series() const noexcept;

 private:
  friend CudaFfaExecutionPlan make_ffa_cuda_execution_plan(
      const FfaSearchPlan& plan);

  CudaFfaExecutionPlan(std::vector<CudaFfaPrepareGroup> groups,
                       std::size_t max_prepared_nsamples,
                       std::size_t max_transform_elements,
                       std::size_t max_detection_slots_per_series);

  std::vector<CudaFfaPrepareGroup> groups_;
  std::size_t max_prepared_nsamples_ = 0;
  std::size_t max_transform_elements_ = 0;
  std::size_t max_detection_slots_per_series_ = 0;
};

struct CudaFfaProgramImpl;
struct CudaFfaInput;
struct CudaFfaBuffer;

// Owns GPU-resident metadata and reusable workspace for one FFA plan on one
// device. It is move-only, not thread-safe, and supports one active
// run_ffa_batch_cuda() call at a time.
class CudaFfaProgram {
 public:
  explicit CudaFfaProgram(CudaFfaExecutionPlan execution_plan,
                          const CudaFfaProgramOptions& program_options = {},
                          const CudaFfaExecutionOptions& execution_options = {});
  explicit CudaFfaProgram(const FfaSearchPlan& plan,
                          const CudaFfaProgramOptions& program_options = {},
                          const CudaFfaExecutionOptions& execution_options = {});
  ~CudaFfaProgram();

  CudaFfaProgram(CudaFfaProgram&&) noexcept;
  CudaFfaProgram& operator=(CudaFfaProgram&&) noexcept;

  CudaFfaProgram(const CudaFfaProgram&) = delete;
  CudaFfaProgram& operator=(const CudaFfaProgram&) = delete;

  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] const CudaFfaExecutionPlan& execution_plan() const;
  [[nodiscard]] int device_id() const;
  [[nodiscard]] std::size_t tile_capacity() const;
  [[nodiscard]] const CudaFfaWorkspaceShape& workspace_shape() const;
  [[nodiscard]] std::size_t device_metadata_bytes() const;

 private:
  friend class detail::CudaFfaTileRunner;
  friend FfaBatchSearchResult run_ffa_batch_cuda(
      CudaFfaProgram& program,
      CudaTimeSeriesBatchView batch,
      const FfaSearchOptions& options);
  friend void ffa_transform_block_cuda(const CudaFfaProgram& program,
                                       std::size_t group_index,
                                       std::size_t task_index,
                                       CudaFfaInput input,
                                       CudaFfaBuffer scratch,
                                       CudaFfaBuffer output,
                                       const CudaLaunchOptions& options);

  std::unique_ptr<CudaFfaProgramImpl> impl_;
};

struct CudaFfaInput {
  const float* data = nullptr;
  std::size_t nseries = 0;
  // Number of valid prepared samples in each series. The transform consumes
  // only the first rows * bins samples for the current task.
  std::size_t nsamples = 0;
  // Distance between consecutive series starts, in float elements.
  // Must be >= nsamples.
  std::size_t stride = 0;
  FfaTransformShape shape{};
  int device_id = 0;

  [[nodiscard]] std::size_t task_elements() const noexcept {
    return shape.rows * shape.bins;
  }
};

struct CudaFfaBuffer {
  float* data = nullptr;
  std::size_t nseries = 0;
  // Distance between consecutive series starts, in float elements.
  // Must be >= rows * bins.
  std::size_t stride = 0;
  FfaTransformShape shape{};
  int device_id = 0;

  [[nodiscard]] std::size_t task_elements() const noexcept {
    return shape.rows * shape.bins;
  }
};

CudaFfaWorkspaceShape estimate_ffa_cuda_workspace(
    const CudaFfaExecutionPlan& plan,
    const CudaFfaExecutionOptions& options = {});

// Creates CUDA execution groups from a logical FFA search plan. Tasks in each
// group can share exactly one prepared input batch in a future executor.
CudaFfaExecutionPlan make_ffa_cuda_execution_plan(
    const FfaSearchPlan& plan);

// Prepares one device-resident float time-series batch for a single FFA task.
// For downsample_factor == 1 this is a device copy/cast-free reshape into the
// task's prepared length. For downsample_factor > 1 it matches
// downsample_weighted_sum_cuda().
void prepare_ffa_input_cuda(CudaTimeSeriesBatchView input,
                            const FfaSearchTask& task,
                            CudaSpan<float> output,
                            const CudaLaunchOptions& options = {});

// Executes one materialized FFA transform block for a batch of independent
// device-resident prepared time series. Input layout is
// [series][prepared_sample] with input.stride between series. The transform uses
// only the first rows * bins samples of each prepared series. Scratch and output
// layouts are [series][row][bin] with their own strides. This materialized
// primitive owns temporary transform op buffers and therefore requires
// options.synchronize_after_call == true. Fully reusable asynchronous
// execution belongs to the Program metadata path.
void ffa_transform_block_cuda(CudaFfaInput input,
                              CudaFfaBuffer scratch,
                              CudaFfaBuffer output,
                              const CudaLaunchOptions& options = {});

// Executes one FFA transform task using GPU-resident transform metadata owned
// by program. It enqueues work on options.stream but never synchronizes; the
// caller must keep program and all buffers alive until that stream completes.
void ffa_transform_block_cuda(const CudaFfaProgram& program,
                              std::size_t group_index,
                              std::size_t task_index,
                              CudaFfaInput input,
                              CudaFfaBuffer scratch,
                              CudaFfaBuffer output,
                              const CudaLaunchOptions& options = {});

// Executes one already-preprocessed dense [series][sample] tile. The tile must
// contain no more than program.tile_capacity() series. Returned series_index
// values are local to this tile; no DM-specific semantics are assumed.
FfaBatchSearchResult run_ffa_batch_cuda(
    CudaFfaProgram& program,
    CudaTimeSeriesBatchView batch,
    const FfaSearchOptions& options = {});

// Single-series CUDA counterpart to search_ffa_cpu(). This is a thin wrapper
// around run_ffa_batch_cuda() with one series and reuses program workspace.
FfaSearchResult search_ffa_cuda(
    CudaFfaProgram& program,
    CudaSpan<const float> time_series,
    const FfaSearchOptions& options = {});

// One-shot convenience wrapper. Repeated tile execution should construct one
// CudaFfaProgram and call run_ffa_batch_cuda() instead.
FfaSearchResult search_ffa_cuda(
    CudaSpan<const float> time_series,
    const FfaSearchPlan& plan,
    const FfaSearchOptions& options = {},
    const CudaFfaProgramOptions& program_options = {},
    const CudaFfaExecutionOptions& execution_options = {});

}  // namespace gaffa
