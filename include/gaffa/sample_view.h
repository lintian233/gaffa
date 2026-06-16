#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
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
  const T* data = nullptr;
  SampleShape shape{};

  std::size_t size() const {
    return sample_element_count(shape);
  }

  bool empty() const {
    return data == nullptr || size() == 0;
  }
};

template <typename T>
struct MutableHostSampleView {
  T* data = nullptr;
  SampleShape shape{};

  std::size_t size() const {
    return sample_element_count(shape);
  }

  bool empty() const {
    return data == nullptr || size() == 0;
  }
};

using AnyHostSampleView =
    std::variant<HostSampleView<std::uint8_t>,
                 HostSampleView<std::uint16_t>, HostSampleView<float>>;

}  // namespace gaffa
