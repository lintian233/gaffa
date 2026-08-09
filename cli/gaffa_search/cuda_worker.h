#pragma once

#include "config.h"

#include "gaffa/periodic_peak.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gaffa_search {

struct CudaSearchResult {
  gaffa::DmPeaks peaks;
  bool complete = true;
  std::vector<std::string> warnings;
};

class CudaWorker {
 public:
  CudaWorker(int device_id, std::size_t tile_capacity);
  ~CudaWorker();

  CudaWorker(const CudaWorker&) = delete;
  CudaWorker& operator=(const CudaWorker&) = delete;
  CudaWorker(CudaWorker&&) noexcept;
  CudaWorker& operator=(CudaWorker&&) noexcept;

  [[nodiscard]] int device_id() const noexcept;

  void prepare_native(const SearchRangeConfig& search,
                      std::size_t source_nsamples, double tsamp,
                      float snr_threshold, std::size_t max_peaks,
                      const std::string& preprocess,
                      double running_median_seconds,
                      std::size_t max_peak_buffer_bytes,
                      const gaffa::PeakReductionOptions& reduction);

  CudaSearchResult run_native(
      std::span<const std::uint32_t> tile, std::size_t nseries,
      std::size_t source_nsamples, std::span<const double> dm_values,
      std::size_t global_dm_index_begin);

  CudaSearchResult run_native(
      std::span<const float> tile, std::size_t nseries,
      std::size_t source_nsamples, std::span<const double> dm_values,
      std::size_t global_dm_index_begin);

#ifdef GAFFA_SEARCH_ENABLE_LOKI
  void prepare_loki(const SearchRangeConfig& search,
                    std::size_t source_nsamples, double tsamp,
                    float snr_threshold, std::size_t max_peaks,
                    const std::string& preprocess,
                    double running_median_seconds,
                    const gaffa::PeakReductionOptions& reduction);

  CudaSearchResult run_loki(
      std::span<const std::uint32_t> tile, std::size_t nseries,
      std::size_t source_nsamples, std::span<const double> dm_values,
      std::size_t global_dm_index_begin);

  CudaSearchResult run_loki(
      std::span<const float> tile, std::size_t nseries,
      std::size_t source_nsamples, std::span<const double> dm_values,
      std::size_t global_dm_index_begin);
#endif

  // Synchronizes and releases the active backend program. The worker remains
  // usable and can be prepared for a different backend afterwards.
  void reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gaffa_search
