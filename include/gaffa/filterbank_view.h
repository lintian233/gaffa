#pragma once

#include "gaffa/filterbank.h"
#include "gaffa/sample_view.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace gaffa {

inline SampleShape sample_shape(const FilterbankHeader& header) {
  if (header.nsamples < 0 || header.nifs < 0 || header.nchans < 0) {
    throw std::runtime_error(
        "filterbank sample shape contains a negative dimension");
  }

  SampleShape shape{static_cast<std::size_t>(header.nsamples),
                    static_cast<std::size_t>(header.nifs),
                    static_cast<std::size_t>(header.nchans)};
  (void)sample_element_count(shape);
  return shape;
}

template <typename T>
HostSampleView<T> sample_view(const FilterbankData& filterbank) {
  const auto* samples = std::get_if<std::vector<T>>(&filterbank.samples);
  if (samples == nullptr) {
    throw std::runtime_error("filterbank sample dtype mismatch");
  }

  const SampleShape shape = sample_shape(filterbank.header);
  if (samples->size() != sample_element_count(shape)) {
    throw std::runtime_error("filterbank sample size does not match header shape");
  }

  return make_host_sample_view<T>(std::span<const T>(*samples), shape);
}

template <typename T>
MutableHostSampleView<T> mutable_sample_view(FilterbankData& filterbank) {
  auto* samples = std::get_if<std::vector<T>>(&filterbank.samples);
  if (samples == nullptr) {
    throw std::runtime_error("filterbank sample dtype mismatch");
  }

  const SampleShape shape = sample_shape(filterbank.header);
  if (samples->size() != sample_element_count(shape)) {
    throw std::runtime_error("filterbank sample size does not match header shape");
  }

  return make_mutable_host_sample_view<T>(std::span<T>(*samples), shape);
}

AnyHostSampleView sample_view(const FilterbankData& filterbank);

}  // namespace gaffa
