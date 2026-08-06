#include "gaffa/ffa_cuda.h"

#include "gaffa/cuda_memory.h"
#include "gaffa/time_series.h"
#include "gaffa/time_series_cuda.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gaffa {
namespace {

#include "ffa_cuda_kernels.cuh"
#include "ffa_cuda_transform.cuh"
#include "ffa_cuda_program.cuh"

class CudaDeviceScope {
 public:
  explicit CudaDeviceScope(int device_id) {
    check_cuda(cudaGetDevice(&previous_device_), "cudaGetDevice");
    if (previous_device_ != device_id) {
      check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
      restore_device_ = true;
    }
  }

  ~CudaDeviceScope() {
    if (restore_device_) {
      static_cast<void>(cudaSetDevice(previous_device_));
    }
  }

  CudaDeviceScope(const CudaDeviceScope&) = delete;
  CudaDeviceScope& operator=(const CudaDeviceScope&) = delete;

 private:
  int previous_device_ = 0;
  bool restore_device_ = false;
};

#include "ffa_cuda_executor.cuh"

}  // namespace gaffa
