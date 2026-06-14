#include "gaffa/vector_add_cuda.hpp"

#include "gaffa/vector_add.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) {
    check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc");
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  float* get() {
    return data_;
  }

 private:
  float* data_ = nullptr;
};

__global__ void vector_add_kernel(const float* lhs, const float* rhs, float* out, int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    out[index] = lhs[index] + rhs[index];
  }
}

}  // namespace

namespace gaffa {

std::vector<float> vector_add_cuda(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  const int size = static_cast<int>(lhs.size());
  std::vector<float> out(lhs.size());

  const std::size_t bytes = lhs.size() * sizeof(float);
  DeviceBuffer device_lhs(bytes);
  DeviceBuffer device_rhs(bytes);
  DeviceBuffer device_out(bytes);

  check_cuda(cudaMemcpy(device_lhs.get(), lhs.data(), bytes, cudaMemcpyHostToDevice), "copy lhs");
  check_cuda(cudaMemcpy(device_rhs.get(), rhs.data(), bytes, cudaMemcpyHostToDevice), "copy rhs");

  constexpr int threads_per_block = 256;
  const int blocks = (size + threads_per_block - 1) / threads_per_block;
  vector_add_kernel<<<blocks, threads_per_block>>>(device_lhs.get(), device_rhs.get(), device_out.get(), size);
  check_cuda(cudaGetLastError(), "vector_add_kernel");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  check_cuda(cudaMemcpy(out.data(), device_out.get(), bytes, cudaMemcpyDeviceToHost), "copy out");
  return out;
}

int cuda_runtime_version() {
  int version = 0;
  check_cuda(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion");
  return version;
}

int cuda_device_count() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice) {
    return 0;
  }
  check_cuda(status, "cudaGetDeviceCount");
  return count;
}

}  // namespace gaffa
