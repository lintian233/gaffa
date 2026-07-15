#pragma once

#include "gaffa/cuda_memory.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace gaffa {

// Inclusive numeric range used by Loki's Taylor-parameter search grid.
struct ClosedRange {
  double minimum = 0.0;
  double maximum = 0.0;
};

// Loki supports contiguous Taylor models only: frequency; acceleration plus
// frequency; jerk plus acceleration plus frequency; or snap through
// frequency. Values retain Loki's parameter convention. The current adapter
// rejects snap plans because Loki's public region planner cannot schedule the
// four-parameter model yet.
struct LokiTaylorSearchSpace {
  ClosedRange frequency_hz{};
  std::optional<ClosedRange> acceleration{};
  std::optional<ClosedRange> jerk{};
  std::optional<ClosedRange> snap{};
};

struct LokiPffaPlanOptions {
  std::size_t phase_bins_min = 256;
  std::size_t phase_bins_max = 256;
  double eta = 1.0;
  double duty_cycle_max = 0.20;
  double width_spacing = 1.5;
  float snr_threshold = 6.0F;
};

// A device-neutral definition of a Loki time-domain P-FFA search. It carries
// no Loki ABI types and no GPU allocation state. Region chunking is performed
// by LokiPffaProgram after its GPU memory budget is resolved.
class LokiPffaPlan {
 public:
  LokiPffaPlan(LokiPffaPlan&&) noexcept;
  LokiPffaPlan& operator=(LokiPffaPlan&&) noexcept;
  LokiPffaPlan(const LokiPffaPlan&) = default;
  LokiPffaPlan& operator=(const LokiPffaPlan&) = default;
  ~LokiPffaPlan() = default;

  [[nodiscard]] std::size_t input_nsamples() const noexcept;
  [[nodiscard]] double tsamp_seconds() const noexcept;
  [[nodiscard]] const LokiTaylorSearchSpace& search_space() const noexcept;
  [[nodiscard]] const LokiPffaPlanOptions& options() const noexcept;

 private:
  friend LokiPffaPlan make_loki_pffa_plan(
      std::size_t input_nsamples, double tsamp_seconds,
      LokiTaylorSearchSpace search_space, LokiPffaPlanOptions options);

  LokiPffaPlan(std::size_t input_nsamples, double tsamp_seconds,
               LokiTaylorSearchSpace search_space, LokiPffaPlanOptions options);

  std::size_t input_nsamples_ = 0;
  double tsamp_seconds_ = 0.0;
  LokiTaylorSearchSpace search_space_{};
  LokiPffaPlanOptions options_{};
};

LokiPffaPlan make_loki_pffa_plan(
    std::size_t input_nsamples, double tsamp_seconds,
    LokiTaylorSearchSpace search_space,
    LokiPffaPlanOptions options = {});

// Program-lifetime properties. The memory budget is fixed when the program
// first resolves its private execution layout, so it cannot change per run.
struct LokiPffaProgramOptions {
  int device_id = 0;
  // Zero resolves a conservative budget from free memory at first search.
  std::size_t memory_budget_bytes = 0;
  std::size_t memory_reserve_bytes = 512ULL * 1024ULL * 1024ULL;
};

struct LokiPffaExecutionOptions {
  // nullptr uses the default stream on the program's device. A non-null stream
  // must belong to that device. search() synchronizes before returning because
  // Loki's compact candidate count is host-visible.
  cudaStream_t stream = nullptr;
  // Total compact peaks returned across every Loki execution region. Exceeding
  // this limit fails rather than silently truncating candidates.
  std::size_t max_compact_peaks_total = 1'000'000;
};

struct LokiPffaPeak {
  float snr = 0.0F;
  double frequency_hz = 0.0;
  std::optional<double> loki_acceleration{};
  std::optional<double> loki_jerk{};
  std::optional<double> loki_snap{};
  std::size_t phase_bins = 0;
  std::size_t boxcar_width = 0;
  float duty_cycle = 0.0F;
};

// Reusable, device-affine time-domain Loki P-FFA executor. It accepts one
// normalized time series. The program owns an explicit unit-variance buffer;
// it does not infer or modify the caller's signal statistics. A Program is
// move-only and supports one active search at a time.
class LokiPffaProgram {
 public:
  LokiPffaProgram(LokiPffaPlan plan,
                  LokiPffaProgramOptions options = {});
  ~LokiPffaProgram();

  LokiPffaProgram(LokiPffaProgram&&) noexcept;
  LokiPffaProgram& operator=(LokiPffaProgram&&) noexcept;
  LokiPffaProgram(const LokiPffaProgram&) = delete;
  LokiPffaProgram& operator=(const LokiPffaProgram&) = delete;

  [[nodiscard]] int device_id() const noexcept;
  [[nodiscard]] const LokiPffaPlan& plan() const noexcept;

  std::vector<LokiPffaPeak> search(
      CudaSpan<const float> normalised_time_series,
      LokiPffaExecutionOptions options = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gaffa
