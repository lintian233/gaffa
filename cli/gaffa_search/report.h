#pragma once

#include "search.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <vector>

namespace gaffa_search {

struct CandidateRow {
  std::size_t candidate_id = 0;
  std::size_t rank = 0;
  double dm = 0.0;
  float snr = 0.0F;
  double period_seconds = 0.0;
  double frequency_hz = 0.0;
  double acceleration_m_per_s2 = 0.0;
  double jerk_m_per_s3 = 0.0;
  double snap_m_per_s4 = 0.0;
  double reference_time_seconds = 0.0;
  gaffa::MotionOrder motion_order = gaffa::MotionOrder::Frequency;
  std::optional<std::size_t> phase_bin;
  std::size_t phase_bins = 0;
  std::size_t boxcar_width_bins = 0;
  double duty_cycle = 0.0;
  std::size_t member_count = 0;
  double dm_min = 0.0;
  double dm_max = 0.0;
  std::size_t dm_index_min = 0;
  std::size_t dm_index_max = 0;
};

struct Report {
  std::filesystem::path input;
  std::size_t observation_nsamples = 0;
  double tsamp_seconds = 0.0;
  double observation_seconds = 0.0;
  FileTiming timing;
  std::vector<SearchRunInfo> search_runs;
  std::size_t raw_peak_count = 0;
  std::size_t candidate_count = 0;
  std::size_t selected_count = 0;
  std::size_t harmonic_relation_count = 0;
  std::vector<CandidateRow> candidates;
};

Report make_report(const FileResult& result);

const char* motion_order_name(gaffa::MotionOrder order) noexcept;

}  // namespace gaffa_search
