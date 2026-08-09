#pragma once

#include "config.h"

#include "gaffa/candidate_analysis.h"
#include "gaffa/periodic_peak.h"

#include <filesystem>
#include <vector>

namespace gaffa_search {

class ProgressTracker;

struct FileTiming {
  double read_seconds = 0.0;
  double dedispersion_seconds = 0.0;
  double search_seconds = 0.0;
  double candidate_seconds = 0.0;
  double total_seconds = 0.0;
};

struct SearchCoordinate {
  // Search windows currently all start at the first observation sample. Keep
  // the origin explicit so a future non-zero window can be normalized before
  // candidate clustering.
  double start_time_seconds = 0.0;
  double valid_duration_seconds = 0.0;
  double tsamp_seconds = 0.0;

  std::size_t source_nsamples = 0;
  std::size_t prepared_nsamples = 0;
  bool padded = false;
};

struct SearchRunInfo {
  std::size_t dm_range_id = 0;
  std::size_t search_range_id = 0;
  Backend backend = Backend::NativeCpu;
  SearchCoordinate coordinate{};
  std::size_t raw_peak_count = 0;
};

struct FileResult {
  std::filesystem::path input;
  std::size_t observation_nsamples = 0;
  double tsamp_seconds = 0.0;
  double observation_seconds = 0.0;
  FileTiming timing;
  std::size_t raw_peak_count = 0;
  gaffa::CandidateResult candidates;
  std::vector<SearchRunInfo> search_runs;
};

struct RunResult {
  // Files are analyzed independently. Candidates are never merged across
  // files because their observation epochs and noise realizations differ.
  std::vector<FileResult> files;
};

RunResult execute(const Config& config);
RunResult execute(const Config& config, ProgressTracker& progress);

}  // namespace gaffa_search
