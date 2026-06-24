#pragma once

#include "gaffa/dedispersion.h"
#include "gaffa/ffa_candidate.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/ffa_search.h"
#include "gaffa/preprocessing.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gaffa {

struct DmCandidate {
  double dm = 0.0;
  std::size_t dm_index = 0;
  FfaCandidate ffa{};
};

struct DmSearchOptions {
  RiptideFfaPlanOptions plan{};
  PreprocessPlan preprocess{};
  float snr_threshold = 6.0F;
  std::size_t max_candidates = 1024;
};

struct DmSearchResult {
  std::vector<DmCandidate> candidates;
};

TimeSeries dm_time_series_cpu(const DedispersedResult<std::uint32_t>& input,
                              std::size_t dm_index,
                              double tsamp);

TimeSeries dm_time_series_cpu(const DedispersedResult<float>& input,
                              std::size_t dm_index,
                              double tsamp);

// Searches every DM row in an eager host dedispersion result. The FFA core stays
// one-dimensional; this wrapper only extracts each DM time series, optionally
// applies preprocessing, runs FFA search, and attaches DM metadata.
DmSearchResult search_dms_cpu(const DedispersedResult<std::uint32_t>& input,
                              std::span<const double> dms,
                              double tsamp,
                              const DmSearchOptions& options);

DmSearchResult search_dms_cpu(const DedispersedResult<float>& input,
                              std::span<const double> dms,
                              double tsamp,
                              const DmSearchOptions& options);

}  // namespace gaffa
