#pragma once

#include "config.h"

#include "gaffa/candidate_analysis.h"
#include "gaffa/periodic_peak.h"

#include <filesystem>
#include <functional>
#include <span>
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
  bool complete = true;
  std::vector<std::string> warnings;
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
  bool complete = true;
  std::vector<std::string> warnings;
};

using FileResultConsumer = std::function<void(FileResult)>;

// Discover regular filterbank files in a single-file or directory input.
// Directory results are sorted for deterministic execution and output order.
std::vector<std::filesystem::path> discover_inputs(
    const std::filesystem::path& input);

// Search one input at a time and immediately transfer its result to consumer.
// The consumer is called synchronously and must consume or move the result
// before returning. Results from different input files are never merged.
void search_each_file(const Config& config,
                      std::span<const std::filesystem::path> inputs,
                      ProgressTracker& progress,
                      FileResultConsumer consumer);

}  // namespace gaffa_search
