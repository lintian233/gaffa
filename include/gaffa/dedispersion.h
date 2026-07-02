#pragma once

#include "gaffa/sample_view.h"

#include <cstddef>
#include <cstdint>
#include <span>
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

template <typename T>
struct DedispersedResult {
  std::vector<T> data;
  DedispersedShape shape{};
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

  [[nodiscard]] HostSampleView<T> view() const noexcept {
    return HostSampleView<T>{data.data(), shape};
  }

  [[nodiscard]] MutableHostSampleView<T> mutable_view() noexcept {
    return MutableHostSampleView<T>{data.data(), shape};
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
