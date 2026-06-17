#include "gaffa/cuda_memory.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

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
    check_cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
  }
}

CudaDeviceMemory::CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_) {
  other.data_ = nullptr;
  other.bytes_ = 0;
}

CudaDeviceMemory& CudaDeviceMemory::operator=(
    CudaDeviceMemory&& other) noexcept {
  if (this != &other) {
    release();
    data_ = other.data_;
    bytes_ = other.bytes_;
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

CudaDeviceMemory::~CudaDeviceMemory() {
  release();
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

void CudaDeviceMemory::release() noexcept {
  if (data_ != nullptr) {
    (void)cudaFree(data_);
    data_ = nullptr;
    bytes_ = 0;
  }
}

}  // namespace gaffa
