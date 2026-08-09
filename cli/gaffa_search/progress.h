#pragma once

#include "config.h"

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <mutex>

namespace gaffa_search {

enum class ProgressStage {
  Idle,
  Read,
  Dedispersion,
  Search,
  Candidate,
  Complete,
  Failed,
};

struct ProgressSummary {
  double elapsed_seconds = 0.0;
  std::size_t raw_peaks = 0;
  std::size_t candidate_groups = 0;
  std::size_t harmonic_relations = 0;
  std::size_t final_candidates = 0;
};

struct ProgressContext {
  std::filesystem::path input;
  std::size_t file_index = 0;
  std::size_t file_count = 0;
  std::size_t dm_range_index = 0;
  std::size_t search_range_index = 0;
  Backend backend = Backend::NativeCpu;
};

struct ProgressSnapshot {
  ProgressStage stage = ProgressStage::Idle;
  ProgressContext context{};

  std::size_t dm_ranges_completed = 0;
  std::size_t dm_ranges_total = 0;
  std::size_t search_runs_completed = 0;
  std::size_t search_runs_total = 0;
  std::size_t active_units_completed = 0;
  std::size_t active_units_total = 0;

  std::size_t observation_nsamples = 0;
  double tsamp_seconds = 0.0;
  double duration_seconds = 0.0;
  std::size_t raw_peaks = 0;
  ProgressSummary summary{};
};

// Execution-side state only. It never formats or writes output. Stage
// metadata is protected by a mutex; worker-facing counters are atomic so a
// CUDA tile can update progress without taking an output lock.
class ProgressTracker {
 public:
  void begin_file(const std::filesystem::path& input,
                  std::size_t file_index, std::size_t file_count);

  void begin_read();
  void finish_read(std::size_t observation_nsamples, double tsamp_seconds,
                   double duration_seconds);

  void begin_work(std::size_t total_dm_ranges,
                  std::size_t total_search_runs);
  void begin_dm_range(std::size_t dm_range_index);
  void complete_dm_range() noexcept;

  void begin_search_phase();
  void begin_search_run(std::size_t dm_range_index,
                        std::size_t search_range_index, Backend backend,
                        std::size_t total_units);
  void complete_search_units(std::size_t count = 1) noexcept;
  void finish_search_run(std::size_t raw_peak_count);

  void begin_candidate(std::size_t raw_peak_count);
  void finish_candidate(const ProgressSummary& summary);

  void finish_file();
  void mark_failed() noexcept;

  [[nodiscard]] ProgressSnapshot snapshot() const;

 private:
  mutable std::mutex metadata_mutex_;
  ProgressStage stage_ = ProgressStage::Idle;
  ProgressContext context_{};

  std::size_t dm_ranges_total_ = 0;
  std::size_t search_runs_total_ = 0;
  std::size_t active_units_total_ = 0;
  std::size_t observation_nsamples_ = 0;
  double tsamp_seconds_ = 0.0;
  double duration_seconds_ = 0.0;
  ProgressSummary summary_{};

  std::atomic<std::size_t> dm_ranges_completed_{0};
  std::atomic<std::size_t> search_runs_completed_{0};
  std::atomic<std::size_t> active_units_completed_{0};
  std::atomic<std::size_t> raw_peaks_{0};
};

}  // namespace gaffa_search
