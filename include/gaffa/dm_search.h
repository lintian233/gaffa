#pragma once

#include "gaffa/dedispersion.h"
#include "gaffa/ffa_peak.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/ffa_search.h"
#include "gaffa/peak_grouping.h"
#include "gaffa/periodic_peak.h"
#include "gaffa/preprocessing.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gaffa {

struct DmSearchOptions {
  RiptideFfaPlanOptions plan{};
  PreprocessPlan preprocess{};
  float snr_threshold = 6.0F;
  // 0 means unbounded. Applied per DM and per FFA block as a raw peak guard.
  std::size_t max_peaks = 0;
  DmPeakGroupingOptions grouping{};
};

struct DmSearchResult {
  std::vector<DmPeakGroups> peak_groups;
};

TimeSeries dm_time_series_cpu(const DedispersedResult<std::uint32_t>& input,
                              std::size_t dm_index,
                              double tsamp);

TimeSeries dm_time_series_cpu(const DedispersedResult<float>& input,
                              std::size_t dm_index,
                              double tsamp);

// Runs preprocessing and FFA peak search for every DM row in an eager host
// dedispersion result. Each returned entry owns all raw peaks for one DM trial
// plus its local peak groups; cross-DM clustering belongs downstream.
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
