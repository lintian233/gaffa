#pragma once

#include "gaffa/peak_grouping.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa::detail {

struct PeakGroupRecord {
  std::size_t dm_index = 0;
  std::size_t best_peak_index = 0;
  std::size_t best_member_index = 0;
  std::size_t member_begin = 0;
  std::size_t member_count = 0;
};

struct DmGroupRange {
  std::size_t dm_index = 0;
  std::size_t group_begin = 0;
  std::size_t group_count = 0;
  std::size_t member_begin = 0;
  std::size_t member_count = 0;
};

// Internal, non-owning grouping representation. raw_peaks must outlive this
// object; member_indices partitions the input without copying DmPeak values.
struct PeakGroupIndex {
  std::span<const DmPeak> raw_peaks;
  std::vector<std::size_t> member_indices;
  std::vector<PeakGroupRecord> groups;
  std::vector<DmGroupRange> dm_ranges;
};

PeakGroupIndex build_peak_group_index_cpu(
    std::span<const DmPeak> peaks,
    double searched_duration_seconds,
    const DmPeakGroupingOptions& options);

std::vector<DmPeakGroups> materialize_dm_peak_groups(
    const PeakGroupIndex& index);

}  // namespace gaffa::detail
