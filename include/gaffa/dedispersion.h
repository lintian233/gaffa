#pragma once

#include "gaffa/sample_view.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace gaffa {

struct SingleDmDedispersionPlan {
  double dm = 0.0;
  double ref_frequency_mhz = 0.0;
  double tsamp = 0.0;
  std::size_t chan_begin = 0;
  std::size_t chan_end = 0;
};

struct MultiDmDedispersionPlan {
  double dm_low = 0.0;
  double dm_step = 1.0;
  std::size_t ndm = 0;
  double ref_frequency_mhz = 0.0;
  double tsamp = 0.0;
  std::size_t chan_begin = 0;
  std::size_t chan_end = 0;
};

struct SubbandDedispersionOptions {
  std::size_t subband_channels = 32;
  std::size_t ndm_per_nominal = 32;
};

struct DedispersedShape {
  std::size_t ndm = 0;
  std::size_t nsamples = 0;
};

inline std::size_t dedispersed_element_count(DedispersedShape shape) {
  if (shape.ndm != 0 &&
      shape.nsamples > std::numeric_limits<std::size_t>::max() / shape.ndm) {
    throw std::overflow_error("dedispersed shape size overflow");
  }
  return shape.ndm * shape.nsamples;
}

template <typename T>
struct DedispersedResultView {
  std::span<const T> data;
  DedispersedShape shape{};

  [[nodiscard]] std::size_t size() const {
    return dedispersed_element_count(shape);
  }

  [[nodiscard]] bool empty() const noexcept {
    return data.empty();
  }

  [[nodiscard]] std::span<const T> dm_series(std::size_t dm_index) const {
    const std::size_t expected_size = size();
    if (data.size() != expected_size) {
      throw std::invalid_argument(
          "dedispersed result view data size does not match shape");
    }
    if (dm_index >= shape.ndm) {
      throw std::out_of_range("dedispersed result dm_index is out of range");
    }
    return data.subspan(dm_index * shape.nsamples, shape.nsamples);
  }
};

template <typename T>
struct MutableDedispersedResultView {
  std::span<T> data;
  DedispersedShape shape{};

  [[nodiscard]] std::size_t size() const {
    return dedispersed_element_count(shape);
  }

  [[nodiscard]] bool empty() const noexcept {
    return data.empty();
  }

  [[nodiscard]] std::span<T> dm_series(std::size_t dm_index) const {
    const std::size_t expected_size = size();
    if (data.size() != expected_size) {
      throw std::invalid_argument(
          "dedispersed result view data size does not match shape");
    }
    if (dm_index >= shape.ndm) {
      throw std::out_of_range("dedispersed result dm_index is out of range");
    }
    return data.subspan(dm_index * shape.nsamples, shape.nsamples);
  }
};

template <typename T>
struct DedispersedResult {
  std::vector<T> data;
  DedispersedShape shape{};

  [[nodiscard]] DedispersedResultView<T> view() const noexcept {
    return DedispersedResultView<T>{std::span<const T>(data), shape};
  }

  [[nodiscard]] MutableDedispersedResultView<T> mutable_view() noexcept {
    return MutableDedispersedResultView<T>{std::span<T>(data), shape};
  }
};

template <typename T>
struct DedispersedSpectrum {
  std::vector<T> data;
  SampleShape shape{};
  double dm = 0.0;
  double tsamp = 0.0;
  std::size_t chan_begin = 0;
  std::size_t chan_end = 0;

  [[nodiscard]] std::size_t nsamples() const noexcept {
    return shape.nsamples;
  }

  [[nodiscard]] std::size_t nchans() const noexcept {
    return shape.nchans;
  }

  [[nodiscard]] std::size_t size() const noexcept { return data.size(); }

  [[nodiscard]] HostSampleView<T> view() const {
    return make_host_sample_view<T>(std::span<const T>(data), shape);
  }

  [[nodiscard]] MutableHostSampleView<T> mutable_view() {
    return make_mutable_host_sample_view<T>(std::span<T>(data), shape);
  }
};

DedispersedSpectrum<std::uint8_t> dedisperse_spectrum_cpu(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan);

DedispersedSpectrum<std::uint16_t> dedisperse_spectrum_cpu(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan);

DedispersedSpectrum<float> dedisperse_spectrum_cpu(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan);

DedispersedResult<std::uint32_t> dedisperse_single_dm_cpu(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan);

DedispersedResult<std::uint32_t> dedisperse_single_dm_cpu(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan);

DedispersedResult<float> dedisperse_single_dm_cpu(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan);

DedispersedResult<std::uint32_t> dedisperse_multi_dm_cpu(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan);

DedispersedResult<std::uint32_t> dedisperse_multi_dm_cpu(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan);

DedispersedResult<float> dedisperse_multi_dm_cpu(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan);

DedispersedResult<std::uint32_t> dedisperse_subband_cpu(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options = {});

DedispersedResult<std::uint32_t> dedisperse_subband_cpu(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options = {});

DedispersedResult<float> dedisperse_subband_cpu(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options = {});

}  // namespace gaffa
