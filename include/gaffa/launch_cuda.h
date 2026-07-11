#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace gaffa {

struct CudaLaunchOptions {
  int device_id = 0;
  std::size_t threads_per_block = 256;
  cudaStream_t stream = nullptr;
  bool synchronize_after_call = true;
};

}  // namespace gaffa
