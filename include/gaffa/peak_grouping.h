#pragma once

#include "gaffa/periodic_peak.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct DmPeakGroupingOptions {
  // Maximum Taylor phase-trajectory separation within one DM trial, in cycles.
  double max_phase_distance_cycles = 0.05;
  // false requires identical folded-profile and boxcar widths.
  bool merge_widths = true;
};

struct DmPeakGroup {
  std::size_t best_index = 0;
  std::size_t member_begin = 0;
  std::size_t member_count = 0;
};

// Owns all raw periodic-search responses from one DM trial. Every input peak
// occurs exactly once in members; groups describe contiguous member ranges.
struct DmPeakGroups {
  std::vector<DmPeak> members;
  std::vector<DmPeakGroup> groups;

  [[nodiscard]] const DmPeak& best_of(std::size_t group_index) const;
  [[nodiscard]] std::span<const DmPeak> members_of(
      std::size_t group_index) const;
};

// Coalesces repeated search responses within exactly one DM trial. The input
// is never truncated: local groups retain every raw DmPeak.
DmPeakGroups group_dm_peaks_cpu(
    std::span<const DmPeak> peaks,
    double searched_duration_seconds,
    const DmPeakGroupingOptions& options = {});

// Partitions raw peaks by their source DM trial, then groups each trial
// independently. Returned entries are ordered by dm_index and retain every
// input peak exactly once.
std::vector<DmPeakGroups> group_dm_peak_batch_cpu(
    std::span<const DmPeak> peaks,
    double searched_duration_seconds,
    const DmPeakGroupingOptions& options = {});

}  // namespace gaffa
