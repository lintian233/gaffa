#include "gaffa/peak_grouping.h"

#include "detail/peak_group_index.h"

#include <stdexcept>
#include <utility>

namespace gaffa {
namespace {

std::span<const DmPeak> checked_members(const DmPeakGroups& groups,
                                        const DmPeakGroup& group) {
  if (group.member_begin > groups.members.size() ||
      group.member_count > groups.members.size() - group.member_begin) {
    throw std::out_of_range("DM peak group member range is outside members");
  }
  return std::span<const DmPeak>{groups.members}.subspan(group.member_begin,
                                                          group.member_count);
}

}  // namespace

const DmPeak& DmPeakGroups::best_of(std::size_t group_index) const {
  if (group_index >= groups.size()) {
    throw std::out_of_range("DM peak group index is out of range");
  }
  const DmPeakGroup& group = groups[group_index];
  (void)checked_members(*this, group);
  if (group.best_index < group.member_begin ||
      group.best_index >= group.member_begin + group.member_count) {
    throw std::out_of_range("DM peak group best index is outside its members");
  }
  return members[group.best_index];
}

std::span<const DmPeak> DmPeakGroups::members_of(
    std::size_t group_index) const {
  if (group_index >= groups.size()) {
    throw std::out_of_range("DM peak group index is out of range");
  }
  return checked_members(*this, groups[group_index]);
}

DmPeakGroups group_dm_peaks_cpu(
    std::span<const DmPeak> peaks,
    double searched_duration_seconds,
    const DmPeakGroupingOptions& options) {
  detail::PeakGroupIndex index = detail::build_peak_group_index_cpu(
      peaks, searched_duration_seconds, options);
  if (index.dm_ranges.size() > 1) {
    throw std::invalid_argument(
        "DM peak grouping input must contain exactly one DM trial");
  }
  std::vector<DmPeakGroups> materialized =
      detail::materialize_dm_peak_groups(index);
  return materialized.empty() ? DmPeakGroups{} : std::move(materialized.front());
}

std::vector<DmPeakGroups> group_dm_peak_batch_cpu(
    std::span<const DmPeak> peaks,
    double searched_duration_seconds,
    const DmPeakGroupingOptions& options) {
  const detail::PeakGroupIndex index = detail::build_peak_group_index_cpu(
      peaks, searched_duration_seconds, options);
  return detail::materialize_dm_peak_groups(index);
}

}  // namespace gaffa
