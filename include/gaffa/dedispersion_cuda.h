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
