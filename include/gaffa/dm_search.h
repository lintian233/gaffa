#pragma once

#include "gaffa/dedispersion.h"
#include "gaffa/ffa_peak.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/ffa_search.h"
#include "gaffa/preprocessing.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gaffa {

struct DmPeak {
  double dm = 0.0;
  std::size_t dm_index = 0;
  FfaPeak peak{};
};

struct DmSearchOptions {
  RiptideFfaPlanOptions plan{};
  PreprocessPlan preprocess{};
  float snr_threshold = 6.0F;
  // 0 means unbounded. Applied per DM and per FFA block as a raw peak guard.
  std::size_t max_peaks = 0;
};

struct DmSearchResult {
  std::vector<DmPeak> peaks;
};

TimeSeries dm_time_series_cpu(const DedispersedResult<std::uint32_t>& input,
                              std::size_t dm_index,
                              double tsamp);

TimeSeries dm_time_series_cpu(const DedispersedResult<float>& input,
                              std::size_t dm_index,
                              double tsamp);

// Runs preprocessing and FFA peak search for every DM row in an eager host
// dedispersion result. Returned peaks retain their source DM metadata; final
// cross-DM clustering and candidate limiting belong to downstream stages.
DmSearchResult search_dedispersed_ffa_cpu(
    const DedispersedResult<std::uint32_t>& input,
    std::span<const double> dms,
    double tsamp,
    const DmSearchOptions& options);

DmSearchResult search_dedispersed_ffa_cpu(
    const DedispersedResult<float>& input,
    std::span<const double> dms,
    double tsamp,
    const DmSearchOptions& options);

}  // namespace gaffa
