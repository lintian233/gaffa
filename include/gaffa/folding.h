#pragma once

#include "gaffa/dedispersion.h"
#include "gaffa/sample_view.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gaffa {

struct FoldOptions {
  double period = 0.0;
  double tsamp = 0.0;
  std::size_t nbin = 0;
  std::size_t start_sample = 0;
};

struct FoldedProfile {
  std::vector<float> profile;
  std::vector<double> exposure;
  std::size_t nbin = 0;
};

struct FoldSpectrumOptions {
  double period = 0.0;
  double tsamp = 0.0;
  std::size_t nbin = 0;
  std::size_t subint_samples = 0;
  std::size_t start_sample = 0;
  // 0 keeps the input channel count. Non-zero values fold adjacent input
  // channels into this many output frequency bins without materializing an
  // intermediate downsampled spectrum.
  std::size_t output_channels = 0;
};

struct FoldedCube {
  // Layout is nsubint x nchans x nbin.
  std::vector<float> data;
  // Phase exposure is shared by all channels: nsubint x nbin.
  std::vector<double> exposure;
  std::size_t nsubint = 0;
  std::size_t nchans = 0;
  std::size_t nbin = 0;
};

struct FoldDedispersedSpectrumOptions {
  double period = 0.0;
  std::size_t nbin = 0;
  double tsubint = 10.0;
  // 0 keeps the current channel count. Non-zero values fold adjacent input
  // channels into this many output frequency bins.
  std::size_t output_channels = 0;
};

struct FoldDedispersedSpectrumResult {
  FoldedCube cube;
  // Layout is nbin.
  std::vector<float> profile;
  // Layout is nchans x nbin.
  std::vector<float> freq_phase;
  // Layout is nsubint x nbin.
  std::vector<float> time_phase;
  std::size_t nsubint = 0;
  std::size_t nchans = 0;
  std::size_t nbin = 0;
  double period = 0.0;
  double tsamp = 0.0;
  double tsubint = 0.0;
};

FoldedProfile fold_time_series_cpu(std::span<const std::uint32_t> input,
                                   const FoldOptions& options);

FoldedProfile fold_time_series_cpu(std::span<const float> input,
                                   const FoldOptions& options);

FoldedCube fold_spectrum_cpu(HostSampleView<std::uint8_t> samples,
                             const FoldSpectrumOptions& options);

FoldedCube fold_spectrum_cpu(HostSampleView<std::uint16_t> samples,
                             const FoldSpectrumOptions& options);

FoldedCube fold_spectrum_cpu(HostSampleView<float> samples,
                             const FoldSpectrumOptions& options);

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_view_cpu(
    HostSampleView<std::uint8_t> samples,
    double tsamp,
    const FoldDedispersedSpectrumOptions& options);

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_view_cpu(
    HostSampleView<std::uint16_t> samples,
    double tsamp,
    const FoldDedispersedSpectrumOptions& options);

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_view_cpu(
    HostSampleView<float> samples,
    double tsamp,
    const FoldDedispersedSpectrumOptions& options);

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_cpu(
    const DedispersedSpectrum<std::uint8_t>& spectrum,
    const FoldDedispersedSpectrumOptions& options);

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_cpu(
    const DedispersedSpectrum<std::uint16_t>& spectrum,
    const FoldDedispersedSpectrumOptions& options);

FoldDedispersedSpectrumResult fold_dedispersed_spectrum_cpu(
    const DedispersedSpectrum<float>& spectrum,
    const FoldDedispersedSpectrumOptions& options);

}  // namespace gaffa
