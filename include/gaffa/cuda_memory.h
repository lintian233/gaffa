#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace gaffa {

template <typename T>
struct CudaSpan {
  T* data = nullptr;
  std::size_t count = 0;
  int device_id = 0;

  std::size_t size() const noexcept {
    return count;
  }

  std::size_t bytes() const noexcept {
    return count * sizeof(T);
  }

  bool empty() const noexcept {
    return data == nullptr || count == 0;
  }
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

  CudaSpan<T> as_span(int device_id) noexcept {
    return CudaSpan<T>{
        .data = data(),
        .count = count_,
        .device_id = device_id,
    };
  }

  CudaSpan<const T> as_span(int device_id) const noexcept {
    return CudaSpan<const T>{
        .data = data(),
        .count = count_,
        .device_id = device_id,
    };
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

}  // namespace gaffa
