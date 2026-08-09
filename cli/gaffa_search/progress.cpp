#include "progress.h"

#include <stdexcept>

namespace gaffa_search {

void ProgressTracker::begin_file(const std::filesystem::path& input,
                                 std::size_t file_index,
                                 std::size_t file_count) {
  std::lock_guard lock(metadata_mutex_);
  context_ = ProgressContext{
      .input = input,
      .file_index = file_index,
      .file_count = file_count,
  };
  stage_ = ProgressStage::Idle;
  dm_ranges_total_ = 0;
  search_runs_total_ = 0;
  active_units_total_ = 0;
  observation_nsamples_ = 0;
  tsamp_seconds_ = 0.0;
  duration_seconds_ = 0.0;
  summary_ = {};
  dm_ranges_completed_.store(0, std::memory_order_relaxed);
  search_runs_completed_.store(0, std::memory_order_relaxed);
  active_units_completed_.store(0, std::memory_order_relaxed);
  raw_peaks_.store(0, std::memory_order_relaxed);
}

void ProgressTracker::begin_read() {
  std::lock_guard lock(metadata_mutex_);
  stage_ = ProgressStage::Read;
}

void ProgressTracker::finish_read(std::size_t observation_nsamples,
                                  double tsamp_seconds,
                                  double duration_seconds) {
  std::lock_guard lock(metadata_mutex_);
  observation_nsamples_ = observation_nsamples;
  tsamp_seconds_ = tsamp_seconds;
  duration_seconds_ = duration_seconds;
}

void ProgressTracker::begin_work(std::size_t total_dm_ranges,
                                 std::size_t total_search_runs) {
  std::lock_guard lock(metadata_mutex_);
  dm_ranges_total_ = total_dm_ranges;
  search_runs_total_ = total_search_runs;
  active_units_total_ = 0;
  dm_ranges_completed_.store(0, std::memory_order_relaxed);
  search_runs_completed_.store(0, std::memory_order_relaxed);
  active_units_completed_.store(0, std::memory_order_relaxed);
}

void ProgressTracker::begin_dm_range(std::size_t dm_range_index) {
  std::lock_guard lock(metadata_mutex_);
  stage_ = ProgressStage::Dedispersion;
  context_.dm_range_index = dm_range_index;
  active_units_total_ = 0;
  active_units_completed_.store(0, std::memory_order_relaxed);
}

void ProgressTracker::complete_dm_range() noexcept {
  dm_ranges_completed_.fetch_add(1, std::memory_order_relaxed);
}

void ProgressTracker::begin_search_phase() {
  std::lock_guard lock(metadata_mutex_);
  stage_ = ProgressStage::Search;
  active_units_total_ = 0;
  active_units_completed_.store(0, std::memory_order_relaxed);
}

void ProgressTracker::begin_search_run(std::size_t dm_range_index,
                                       std::size_t search_range_index,
                                       Backend backend,
                                       std::size_t total_units) {
  std::lock_guard lock(metadata_mutex_);
  if (stage_ != ProgressStage::Search) {
    throw std::logic_error("search run started outside search stage");
  }
  context_.dm_range_index = dm_range_index;
  context_.search_range_index = search_range_index;
  context_.backend = backend;
  active_units_total_ = total_units;
  active_units_completed_.store(0, std::memory_order_relaxed);
}

void ProgressTracker::complete_search_units(std::size_t count) noexcept {
  active_units_completed_.fetch_add(count, std::memory_order_relaxed);
}

void ProgressTracker::finish_search_run(std::size_t raw_peak_count) {
  std::lock_guard lock(metadata_mutex_);
  search_runs_completed_.fetch_add(1, std::memory_order_relaxed);
  raw_peaks_.fetch_add(raw_peak_count, std::memory_order_relaxed);
}

void ProgressTracker::begin_candidate(std::size_t raw_peak_count) {
  std::lock_guard lock(metadata_mutex_);
  stage_ = ProgressStage::Candidate;
  raw_peaks_.store(raw_peak_count, std::memory_order_relaxed);
  active_units_total_ = 0;
  active_units_completed_.store(0, std::memory_order_relaxed);
}

void ProgressTracker::finish_candidate(const ProgressSummary& summary) {
  std::lock_guard lock(metadata_mutex_);
  summary_ = summary;
}

void ProgressTracker::finish_file() {
  std::lock_guard lock(metadata_mutex_);
  stage_ = ProgressStage::Complete;
}

void ProgressTracker::mark_failed() noexcept {
  std::lock_guard lock(metadata_mutex_);
  stage_ = ProgressStage::Failed;
}

ProgressSnapshot ProgressTracker::snapshot() const {
  std::lock_guard lock(metadata_mutex_);
  return ProgressSnapshot{
      .stage = stage_,
      .context = context_,
      .dm_ranges_completed =
          dm_ranges_completed_.load(std::memory_order_relaxed),
      .dm_ranges_total = dm_ranges_total_,
      .search_runs_completed =
          search_runs_completed_.load(std::memory_order_relaxed),
      .search_runs_total = search_runs_total_,
      .active_units_completed =
          active_units_completed_.load(std::memory_order_relaxed),
      .active_units_total = active_units_total_,
      .observation_nsamples = observation_nsamples_,
      .tsamp_seconds = tsamp_seconds_,
      .duration_seconds = duration_seconds_,
      .raw_peaks = raw_peaks_.load(std::memory_order_relaxed),
      .summary = summary_,
  };
}

}  // namespace gaffa_search
