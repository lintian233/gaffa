#pragma once

#include "gaffa/preprocessing.h"
#include "gaffa/time_series_cuda.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

namespace gaffa {

// Device-affine construction properties of a CUDA preprocessing program.
struct CudaPreprocessProgramOptions {
  int device_id = 0;
};

// Fixed workspace and stream policy for one CUDA preprocessing program.
struct CudaPreprocessExecutionOptions {
  std::size_t series_tile_size = 16;
  std::size_t max_nsamples = 0;
  std::size_t threads_per_block = 256;
  // 0 means unlimited. Non-zero rejects a program before CUDA allocation when
  // its exact temporary workspace would exceed this budget.
  std::size_t workspace_bytes_limit = 0;
  // Non-owning. When non-null, the stream must belong to the program device
  // and must remain alive until all submitted work has been synchronized.
  cudaStream_t stream = nullptr;
};

struct CudaPreprocessWorkspaceShape {
  std::size_t series_tile_size = 0;
  std::size_t max_nsamples = 0;
  std::size_t scrunched_bytes = 0;
  std::size_t baseline_bytes = 0;
  std::size_t partial_stats_bytes = 0;
  std::size_t series_stats_bytes = 0;
  std::size_t status_bytes = 0;
  std::size_t total_bytes = 0;
};

struct CudaPreprocessProgramImpl;

CudaPreprocessWorkspaceShape estimate_cuda_preprocess_workspace(
    const PreprocessPlan& plan,
    const CudaPreprocessExecutionOptions& options = {});

// Owns only preprocessing metadata and temporary workspace. It never owns a
// full-resolution input/output time-series buffer: callers provide mutable
// device storage and the transform is performed in place.
class CudaPreprocessProgram {
 public:
  CudaPreprocessProgram(
      PreprocessPlan plan,
      const CudaPreprocessProgramOptions& program_options = {},
      const CudaPreprocessExecutionOptions& execution_options = {});
  ~CudaPreprocessProgram();

  CudaPreprocessProgram(CudaPreprocessProgram&&) noexcept;
  CudaPreprocessProgram& operator=(CudaPreprocessProgram&&) noexcept;
  CudaPreprocessProgram(const CudaPreprocessProgram&) = delete;
  CudaPreprocessProgram& operator=(const CudaPreprocessProgram&) = delete;

  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] int device_id() const;
  [[nodiscard]] std::size_t tile_capacity() const;
  [[nodiscard]] std::size_t max_nsamples() const;
  [[nodiscard]] const CudaPreprocessWorkspaceShape& workspace_shape() const;

  // Waits for the configured stream and reports deferred input/data errors.
  // A program supports one active in-place run; call this before reusing it.
  // A CUDA runtime failure while submitting or synchronizing work poisons the
  // program because its workspace and stream state are no longer trustworthy.
  // Destroy and recreate a poisoned program instead of attempting reuse.
  void synchronize();

 private:
  friend void preprocess_time_series_batch_inplace_cuda(
      CudaPreprocessProgram& program, MutableCudaTimeSeriesBatchView batch);

  std::unique_ptr<CudaPreprocessProgramImpl> impl_;
};

// Enqueues every step in program's PreprocessPlan on its configured stream.
// The input is transformed in place. Only one call may be active for a Program
// at a time; call synchronize() before reusing that Program or its workspace.
// A CUDA runtime failure poisons the Program and requires reconstruction.
void preprocess_time_series_batch_inplace_cuda(
    CudaPreprocessProgram& program, MutableCudaTimeSeriesBatchView batch);

}  // namespace gaffa
