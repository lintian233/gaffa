#include "gaffa/cuda_memory.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace gaffa {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

}  // namespace

CudaDeviceMemory::CudaDeviceMemory(std::size_t bytes) : bytes_(bytes) {
  if (bytes_ != 0) {
    check_cuda(cudaGetDevice(&device_id_), "cudaGetDevice");
    check_cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
  }
}

CudaDeviceMemory::CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)),
      device_id_(std::exchange(other.device_id_, -1)) {}

CudaDeviceMemory& CudaDeviceMemory::operator=(
    CudaDeviceMemory&& other) noexcept {
  if (this != &other) {
    release_noexcept();
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    device_id_ = std::exchange(other.device_id_, -1);
  }
  return *this;
}

CudaDeviceMemory::~CudaDeviceMemory() {
  release_noexcept();
}

void CudaDeviceMemory::reset() {
  if (data_ == nullptr) {
    return;
  }

  int previous_device = -1;
  check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice");
  const bool switched_device = previous_device != device_id_;
  if (switched_device) {
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice for cudaFree");
  }

  const cudaError_t free_status = cudaFree(data_);
  if (free_status == cudaSuccess) {
    data_ = nullptr;
    bytes_ = 0;
    device_id_ = -1;
  }

  const cudaError_t restore_status =
      switched_device ? cudaSetDevice(previous_device) : cudaSuccess;
  if (restore_status != cudaSuccess) {
    if (free_status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaFree failed: ") + cudaGetErrorString(free_status) +
          "; cudaSetDevice restore failed: " +
          cudaGetErrorString(restore_status));
    }
    check_cuda(restore_status, "cudaSetDevice restore after cudaFree");
  }
  check_cuda(free_status, "cudaFree");
}

void* CudaDeviceMemory::data() noexcept {
  return data_;
}

const void* CudaDeviceMemory::data() const noexcept {
  return data_;
}

std::size_t CudaDeviceMemory::bytes() const noexcept {
  return bytes_;
}

int CudaDeviceMemory::device_id() const noexcept {
  return device_id_;
}

void CudaDeviceMemory::release_noexcept() noexcept {
  if (data_ != nullptr) {
    int previous_device = -1;
    if (cudaGetDevice(&previous_device) != cudaSuccess) {
      return;
    }
    const bool switched_device = previous_device != device_id_;
    if (switched_device && cudaSetDevice(device_id_) != cudaSuccess) {
      return;
    }
    if (cudaFree(data_) == cudaSuccess) {
      data_ = nullptr;
      bytes_ = 0;
      device_id_ = -1;
    }
    if (switched_device) {
      (void)cudaSetDevice(previous_device);
    }
  }
}

}  // namespace gaffa
