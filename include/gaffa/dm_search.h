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

// Finds significant FFA peaks for every DM row in an eager host dedispersion
// result. This layer deliberately does not apply a final top-k cap; downstream
// clustering/candidate selection owns final candidate limiting.
DmSearchResult find_dm_peaks_cpu(const DedispersedResult<std::uint32_t>& input,
                                 std::span<const double> dms,
                                 double tsamp,
                                 const DmSearchOptions& options);

DmSearchResult find_dm_peaks_cpu(const DedispersedResult<float>& input,
                                 std::span<const double> dms,
                                 double tsamp,
                                 const DmSearchOptions& options);

}  // namespace gaffa
