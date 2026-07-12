#include "gaffa/cuda_memory.h"
#include "gaffa/vector_add.hpp"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

namespace {

class DeviceGuard {
 public:
  DeviceGuard() {
    EXPECT_EQ(cudaGetDevice(&device_id_), cudaSuccess);
  }

  ~DeviceGuard() {
    if (device_id_ >= 0) {
      (void)cudaSetDevice(device_id_);
    }
  }

 private:
  int device_id_ = -1;
};

}  // namespace

TEST(CudaDeviceMemory, TracksOwningDeviceAndExplicitReset) {
  if (gaffa::cuda_device_count() == 0) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  int current_device = -1;
  ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  gaffa::CudaDeviceMemory memory(256);
  EXPECT_EQ(memory.device_id(), current_device);
  EXPECT_EQ(memory.bytes(), 256);

  memory.reset();
  EXPECT_EQ(memory.data(), nullptr);
  EXPECT_EQ(memory.bytes(), 0);
  EXPECT_EQ(memory.device_id(), -1);
}

TEST(CudaDeviceMemory, ResetUsesOwnerDeviceAndRestoresCallerDevice) {
  if (gaffa::cuda_device_count() < 2) {
    GTEST_SKIP() << "at least two CUDA devices are required";
  }

  DeviceGuard guard;
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  gaffa::CudaDeviceMemory memory(256);
  ASSERT_EQ(memory.device_id(), 0);

  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  memory.reset();

  int current_device = -1;
  ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  EXPECT_EQ(current_device, 1);
  EXPECT_EQ(memory.device_id(), -1);
}

TEST(CudaDeviceMemory, MovePreservesOwnerDevice) {
  if (gaffa::cuda_device_count() == 0) {
    GTEST_SKIP() << "CUDA device is not visible";
  }

  int current_device = -1;
  ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  gaffa::CudaDeviceMemory source(256);
  gaffa::CudaDeviceMemory target(std::move(source));

  EXPECT_EQ(source.device_id(), -1);
  EXPECT_EQ(target.device_id(), current_device);
  target.reset();
}
