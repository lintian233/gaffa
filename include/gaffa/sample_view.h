#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <variant>

namespace gaffa {

enum class SampleDType {
  UInt8,
  UInt16,
  Float32,
};

struct SampleShape {
  std::size_t nsamples = 0;
  std::size_t nifs = 0;
  std::size_t nchans = 0;
};

inline std::size_t checked_sample_multiply(std::size_t lhs, std::size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error("sample shape size overflow");
  }
  return lhs * rhs;
}

inline std::size_t sample_element_count(const SampleShape& shape) {
  return checked_sample_multiply(
      checked_sample_multiply(shape.nsamples, shape.nifs), shape.nchans);
}

inline std::size_t sample_row_element_count(const SampleShape& shape) {
  return checked_sample_multiply(shape.nifs, shape.nchans);
}

template <typename T>
struct HostSampleView {
  std::span<const T> data;
  SampleShape shape{};

  std::size_t size() const {
    return sample_element_count(shape);
  }

  const T* data_ptr() const noexcept {
    return data.data();
  }

  bool empty() const noexcept {
    return data.empty();
  }
};

template <typename T>
struct MutableHostSampleView {
  std::span<T> data;
  SampleShape shape{};

  std::size_t size() const {
    return sample_element_count(shape);
  }

  T* data_ptr() const noexcept {
    return data.data();
  }

  bool empty() const noexcept {
    return data.empty();
  }
};

template <typename T>
HostSampleView<T> make_host_sample_view(std::span<const T> data,
                                        SampleShape shape) {
  if (data.size() != sample_element_count(shape)) {
    throw std::invalid_argument("sample view data size does not match shape");
  }
  return HostSampleView<T>{data, shape};
}

template <typename T>
MutableHostSampleView<T> make_mutable_host_sample_view(std::span<T> data,
                                                       SampleShape shape) {
  if (data.size() != sample_element_count(shape)) {
    throw std::invalid_argument("sample view data size does not match shape");
  }
  return MutableHostSampleView<T>{data, shape};
}

using AnyHostSampleView =
    std::variant<HostSampleView<std::uint8_t>,
                 HostSampleView<std::uint16_t>, HostSampleView<float>>;

}  // namespace gaffa
