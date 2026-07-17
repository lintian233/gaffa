#include "gaffa/peak_grouping.h"

#include "gaffa/periodic_match.h"
#include "detail/phase_trajectory_grid.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace gaffa {
namespace {

constexpr std::size_t kNoIndex = std::numeric_limits<std::size_t>::max();

void validate_peak(const DmPeak& peak) {
  if (!std::isfinite(peak.dm) || !std::isfinite(peak.peak.duty_cycle) ||
      !std::isfinite(peak.peak.snr)) {
    throw std::invalid_argument("DM peak grouping requires finite values");
  }
  validate_periodic_motion(peak.peak.motion);
}

void validate_arguments(std::span<const DmPeak> peaks,
                        double searched_duration_seconds,
                        const DmPeakGroupingOptions& options) {
  if (!(searched_duration_seconds > 0.0) ||
      !std::isfinite(searched_duration_seconds)) {
    throw std::invalid_argument(
        "DM peak grouping searched_duration_seconds must be finite and > 0");
  }
  if (options.max_phase_distance_cycles < 0.0 ||
      !std::isfinite(options.max_phase_distance_cycles)) {
    throw std::invalid_argument(
        "DM peak grouping max_phase_distance_cycles must be finite and >= 0");
  }
  if (peaks.empty()) {
    return;
  }
  const DmPeak& first = peaks.front();
  validate_peak(first);
  for (const DmPeak& peak : peaks) {
    validate_peak(peak);
    if (peak.dm_index != first.dm_index || peak.dm != first.dm) {
      throw std::invalid_argument(
          "DM peak grouping input must contain exactly one DM trial");
    }
  }
}

bool is_better_peak(const DmPeak& lhs, const DmPeak& rhs) {
  if (lhs.peak.snr != rhs.peak.snr) {
    return lhs.peak.snr > rhs.peak.snr;
  }
  if (lhs.peak.motion.frequency_hz != rhs.peak.motion.frequency_hz) {
    return lhs.peak.motion.frequency_hz < rhs.peak.motion.frequency_hz;
  }
  if (lhs.peak.boxcar_width_bins != rhs.peak.boxcar_width_bins) {
    return lhs.peak.boxcar_width_bins < rhs.peak.boxcar_width_bins;
  }
  return lhs.peak.phase_bin < rhs.peak.phase_bin;
}

bool locally_compatible(const DmPeak& lhs,
                        const DmPeak& rhs,
                        double searched_duration_seconds,
                        const DmPeakGroupingOptions& options) {
  if (lhs.peak.motion.order != rhs.peak.motion.order) {
    return false;
  }
  if (!options.merge_widths &&
      (lhs.peak.phase_bins != rhs.peak.phase_bins ||
       lhs.peak.boxcar_width_bins != rhs.peak.boxcar_width_bins)) {
    return false;
  }
  const PhaseDrift drift = periodic_phase_drift(
      lhs.peak.motion, rhs.peak.motion, searched_duration_seconds);
  return drift.maximum_cycles <=
         options.max_phase_distance_cycles +
             detail::phase_trajectory_cell_tolerance(
                 options.max_phase_distance_cycles);
}

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
  validate_arguments(peaks, searched_duration_seconds, options);
  if (peaks.empty()) {
    return {};
  }

  const std::size_t count = peaks.size();
  std::vector<std::size_t> assignment(count, kNoIndex);
  std::vector<std::size_t> group_best_input;

  if (options.max_phase_distance_cycles == 0.0) {
    group_best_input.resize(count);
    std::iota(group_best_input.begin(), group_best_input.end(), std::size_t{0});
    std::iota(assignment.begin(), assignment.end(), std::size_t{0});
  } else {
    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs,
                                                      std::size_t rhs) {
      return is_better_peak(peaks[lhs], peaks[rhs]);
    });

    const double cell_width = options.max_phase_distance_cycles +
                              detail::phase_trajectory_cell_tolerance(
                                  options.max_phase_distance_cycles);
    std::unordered_map<detail::PhaseTrajectoryCell, std::vector<std::size_t>,
                       detail::PhaseTrajectoryCellHash>
        groups_by_cell;
    groups_by_cell.reserve(count);
    std::vector<std::size_t> unindexed_groups;

    for (const std::size_t input_index : order) {
      detail::PhaseTrajectoryCell cell{};
      const bool indexed = detail::make_phase_trajectory_cell(
          detail::make_phase_trajectory(peaks[input_index].peak.motion,
                                        searched_duration_seconds),
          cell_width, cell);
      std::size_t selected_group = kNoIndex;
      const auto consider = [&](std::size_t group_index) {
        const std::size_t best_input = group_best_input[group_index];
        if (!locally_compatible(peaks[input_index], peaks[best_input],
                                searched_duration_seconds, options)) {
          return;
        }
        if (selected_group == kNoIndex ||
            is_better_peak(peaks[best_input],
                           peaks[group_best_input[selected_group]]) ||
            (peaks[best_input].peak.snr ==
                 peaks[group_best_input[selected_group]].peak.snr &&
             group_index < selected_group)) {
          selected_group = group_index;
        }
      };

      for (const std::size_t group_index : unindexed_groups) {
        consider(group_index);
      }
      if (indexed) {
        detail::for_each_neighbor_phase_trajectory_cell(
            cell, [&](const detail::PhaseTrajectoryCell& neighbor) {
              const auto found = groups_by_cell.find(neighbor);
              if (found == groups_by_cell.end()) {
                return;
              }
              for (const std::size_t group_index : found->second) {
                consider(group_index);
              }
            });
      } else {
        for (std::size_t group_index = 0;
             group_index < group_best_input.size(); ++group_index) {
          consider(group_index);
        }
      }

      if (selected_group == kNoIndex) {
        selected_group = group_best_input.size();
        group_best_input.push_back(input_index);
        if (indexed) {
          groups_by_cell[cell].push_back(selected_group);
        } else {
          unindexed_groups.push_back(selected_group);
        }
      }
      assignment[input_index] = selected_group;
    }
  }

  std::vector<std::size_t> counts(group_best_input.size(), 0);
  for (const std::size_t group_index : assignment) {
    ++counts[group_index];
  }
  std::vector<std::size_t> begins(counts.size());
  std::size_t total = 0;
  for (std::size_t group_index = 0; group_index < counts.size(); ++group_index) {
    begins[group_index] = total;
    total += counts[group_index];
  }

  DmPeakGroups result;
  result.members.resize(count);
  result.groups.resize(counts.size());
  std::vector<std::size_t> cursors = begins;
  std::vector<std::size_t> output_index(count);
  for (std::size_t input_index = 0; input_index < count; ++input_index) {
    const std::size_t group_index = assignment[input_index];
    const std::size_t destination = cursors[group_index]++;
    result.members[destination] = peaks[input_index];
    output_index[input_index] = destination;
  }
  for (std::size_t group_index = 0; group_index < counts.size(); ++group_index) {
    result.groups[group_index] = DmPeakGroup{
        .best_index = output_index[group_best_input[group_index]],
        .member_begin = begins[group_index],
        .member_count = counts[group_index],
    };
  }
  return result;
}

std::vector<DmPeakGroups> group_dm_peak_batch_cpu(
    std::span<const DmPeak> peaks,
    double searched_duration_seconds,
    const DmPeakGroupingOptions& options) {
  std::map<std::size_t, std::vector<DmPeak>> by_dm;
  for (const DmPeak& peak : peaks) {
    // Validate before using the peak as a grouping input. The single-DM call
    // below remains the authority for the one-index/one-DM invariant.
    validate_peak(peak);
    by_dm[peak.dm_index].push_back(peak);
  }

  std::vector<DmPeakGroups> result;
  result.reserve(by_dm.size());
  for (auto& [dm_index, dm_peaks] : by_dm) {
    (void)dm_index;
    result.push_back(
        group_dm_peaks_cpu(dm_peaks, searched_duration_seconds, options));
  }
  return result;
}

}  // namespace gaffa
