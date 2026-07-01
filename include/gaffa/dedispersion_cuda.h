#pragma once

#include "gaffa/cuda_memory.h"
#include "gaffa/dedispersion.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace gaffa {

struct CudaDedispersionOptions {
  int device_id = 0;
  std::size_t threads_per_block = 256;
  // Used by tiled CUDA algorithms such as subband dedispersion. Full-output
  // single-DM spectrum APIs currently materialize the complete aligned
  // spectrum and do not use this value to reduce output memory.
  std::size_t time_tile_samples = 81920;
};

template <typename T>
struct CudaDedispersedView {
  T* data = nullptr;
  DedispersedShape shape{};
  int device_id = 0;

  CudaSpan<T> as_span() const noexcept {
    return CudaSpan<T>{
        .data = data,
        .count = size(),
        .device_id = device_id,
    };
  }

  std::size_t size() const noexcept {
    return shape.ndm * shape.nsamples;
  }

  std::size_t bytes() const noexcept {
    return as_span().bytes();
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

template <typename T>
struct CudaDedispersedSpectrumView {
  T* data = nullptr;
  SampleShape shape{};
  int device_id = 0;

  CudaSpan<T> as_span() const noexcept {
    return CudaSpan<T>{
        .data = data,
        .count = size(),
        .device_id = device_id,
    };
  }

  std::size_t size() const noexcept {
    return shape.nsamples * shape.nchans;
  }

  std::size_t bytes() const noexcept {
    return as_span().bytes();
  }
};

template <typename T>
struct CudaDedispersedSpectrum {
  CudaDeviceBuffer<T> data;
  SampleShape shape{};
  double dm = 0.0;
  double tsamp = 0.0;
  std::size_t chan_begin = 0;
  std::size_t chan_end = 0;
  int device_id = 0;

  CudaDedispersedSpectrumView<T> view() noexcept {
    return CudaDedispersedSpectrumView<T>{
        .data = data.data(),
        .shape = shape,
        .device_id = device_id,
    };
  }

  CudaDedispersedSpectrumView<const T> view() const noexcept {
    return CudaDedispersedSpectrumView<const T>{
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

// Materializes the full single-DM aligned dynamic spectrum on the host.
// Output shape is (nsamples, 1, selected_nchans), and the dtype matches the
// input sample dtype. This is intended for diagnostics, visualization, and
// workflows that explicitly need the full aligned spectrum.
DedispersedSpectrum<std::uint8_t> dedisperse_spectrum_cuda(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

DedispersedSpectrum<std::uint16_t> dedisperse_spectrum_cuda(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

DedispersedSpectrum<float> dedisperse_spectrum_cuda(
    HostSampleView<float> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

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

// Materializes the full single-DM aligned dynamic spectrum on the selected
// CUDA device. Memory use is
//   output_nsamples * selected_nchans * sizeof(T)
// so large filterbanks can exceed device memory. For large production flows,
// prefer a future streaming/tile consumer API instead of this full-output API.
CudaDedispersedSpectrum<std::uint8_t> dedisperse_spectrum_cuda_device(
    HostSampleView<std::uint8_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

CudaDedispersedSpectrum<std::uint16_t> dedisperse_spectrum_cuda_device(
    HostSampleView<std::uint16_t> samples,
    std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const CudaDedispersionOptions& options = {});

CudaDedispersedSpectrum<float> dedisperse_spectrum_cuda_device(
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

DedispersedSpectrum<std::uint8_t> copy_to_host(
    const CudaDedispersedSpectrum<std::uint8_t>& result);

DedispersedSpectrum<std::uint16_t> copy_to_host(
    const CudaDedispersedSpectrum<std::uint16_t>& result);

DedispersedSpectrum<float> copy_to_host(
    const CudaDedispersedSpectrum<float>& result);

}  // namespace gaffa
