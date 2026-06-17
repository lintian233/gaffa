#pragma once

#include "gaffa/dedispersion.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace gaffa {

struct CudaDedispersionOptions {
  int device_id = 0;
  std::size_t threads_per_block = 256;
  std::size_t time_tile_samples = 81920;
};

class CudaDeviceMemory {
 public:
  CudaDeviceMemory() = default;
  explicit CudaDeviceMemory(std::size_t bytes);

  CudaDeviceMemory(const CudaDeviceMemory&) = delete;
  CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

  CudaDeviceMemory(CudaDeviceMemory&& other) noexcept;
  CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept;

  ~CudaDeviceMemory();

  void* data() noexcept;
  const void* data() const noexcept;
  std::size_t bytes() const noexcept;

 private:
  void release() noexcept;

  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

template <typename T>
class CudaDeviceBuffer {
 public:
  CudaDeviceBuffer() = default;

  explicit CudaDeviceBuffer(std::size_t count)
      : memory_(checked_bytes(count)), count_(count) {}

  T* data() noexcept {
    return static_cast<T*>(memory_.data());
  }

  const T* data() const noexcept {
    return static_cast<const T*>(memory_.data());
  }

  T* get() noexcept {
    return data();
  }

  const T* get() const noexcept {
    return data();
  }

  std::size_t size() const noexcept {
    return count_;
  }

  std::size_t bytes() const noexcept {
    return memory_.bytes();
  }

 private:
  static std::size_t checked_bytes(std::size_t count) {
    if (count != 0 &&
        count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("CUDA device buffer size overflow");
    }
    return count * sizeof(T);
  }

  CudaDeviceMemory memory_{};
  std::size_t count_ = 0;
};

template <typename T>
struct CudaDedispersedView {
  T* data = nullptr;
  DedispersedShape shape{};
  int device_id = 0;

  std::size_t size() const noexcept {
    return shape.ndm * shape.nsamples;
  }

  std::size_t bytes() const noexcept {
    return size() * sizeof(T);
  }
};

template <typename T>
struct CudaDedispersedResult {
  CudaDeviceBuffer<T> data;
  DedispersedShape shape{};
  int device_id = 0;

  CudaDedispersedView<T> view() noexcept {
    return CudaDedispersedView<T>{
        .data = data.data(),
        .shape = shape,
        .device_id = device_id,
    };
  }

  CudaDedispersedView<const T> view() const noexcept {
    return CudaDedispersedView<const T>{
        .data = data.data(),
        .shape = shape,
        .device_id = device_id,
    };
  }

  std::size_t size() const noexcept {
    return data.size();
  }

  std::size_t bytes() const noexcept {
    return data.bytes();
  }
};

DedispersedResult<std::uint32_t> dedisperse_single_dm_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

DedispersedResult<std::uint32_t> dedisperse_single_dm_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

DedispersedResult<float> dedisperse_single_dm_cuda(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

DedispersedResult<std::uint32_t> dedisperse_multi_dm_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

DedispersedResult<std::uint32_t> dedisperse_multi_dm_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

DedispersedResult<float> dedisperse_multi_dm_cuda(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

DedispersedResult<std::uint32_t> dedisperse_subband_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options = {},
    const CudaDedispersionOptions& cuda_options = {});

DedispersedResult<std::uint32_t> dedisperse_subband_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options = {},
    const CudaDedispersionOptions& cuda_options = {});

DedispersedResult<float> dedisperse_subband_cuda(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options = {},
    const CudaDedispersionOptions& cuda_options = {});

CudaDedispersedResult<std::uint32_t> dedisperse_single_dm_cuda_device(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

CudaDedispersedResult<std::uint32_t> dedisperse_single_dm_cuda_device(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

CudaDedispersedResult<float> dedisperse_single_dm_cuda_device(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

CudaDedispersedResult<std::uint32_t> dedisperse_multi_dm_cuda_device(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

CudaDedispersedResult<std::uint32_t> dedisperse_multi_dm_cuda_device(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

CudaDedispersedResult<float> dedisperse_multi_dm_cuda_device(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

CudaDedispersedResult<std::uint32_t> dedisperse_subband_cuda_device(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options = {},
    const CudaDedispersionOptions& cuda_options = {});

CudaDedispersedResult<std::uint32_t> dedisperse_subband_cuda_device(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options = {},
    const CudaDedispersionOptions& cuda_options = {});

CudaDedispersedResult<float> dedisperse_subband_cuda_device(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options = {},
    const CudaDedispersionOptions& cuda_options = {});

DedispersedResult<std::uint32_t> copy_to_host(
    const CudaDedispersedResult<std::uint32_t>& result);

DedispersedResult<float> copy_to_host(
    const CudaDedispersedResult<float>& result);

}  // namespace gaffa
