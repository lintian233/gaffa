#include "peak_group_index.h"

#include "gaffa/periodic_match.h"
#include "phase_match.h"
#include "phase_trajectory_grid.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gaffa::detail {
namespace {

constexpr std::size_t kNoIndex = std::numeric_limits<std::size_t>::max();

struct DmRun {
  std::size_t dm_index = 0;
  std::size_t begin = 0;
  std::size_t count = 0;
};

struct LocalGrouping {
  std::vector<std::size_t> member_indices;
  std::vector<PeakGroupRecord> groups;
};

struct GroupingWorkspace {
  std::vector<std::size_t> assignment;
  std::vector<std::size_t> order;
  std::vector<std::size_t> group_best;
  std::vector<std::size_t> next_group;
  std::vector<std::size_t> unindexed_groups;
  std::vector<std::size_t> counts;
  std::vector<std::size_t> begins;
  std::vector<std::size_t> cursors;
  std::vector<std::size_t> output_index;
  std::vector<PhasePolynomial> phases;
  std::unordered_map<PhaseTrajectoryCell, std::size_t,
                     PhaseTrajectoryCellHash>
      cell_heads;

  void reset(std::size_t count) {
    assignment.assign(count, kNoIndex);
    order.resize(count);
    std::iota(order.begin(), order.end(), std::size_t{0});
    group_best.clear();
    next_group.clear();
    unindexed_groups.clear();
    counts.clear();
    begins.clear();
    cursors.clear();
    output_index.resize(count);
    phases.resize(count);
    cell_heads.clear();
    cell_heads.reserve(count);
  }
};

void validate_peak(const DmPeak& peak) {
  if (!std::isfinite(peak.dm) || !std::isfinite(peak.peak.duty_cycle) ||
      !std::isfinite(peak.peak.snr)) {
    throw std::invalid_argument("DM peak grouping requires finite values");
  }
  validate_periodic_motion(peak.peak.motion);
}

void validate_options(double searched_duration_seconds,
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
                        const PhasePolynomial& lhs_phase,
                        const DmPeak& rhs,
                        const PhasePolynomial& rhs_phase,
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
  const double maximum_cycles =
      options.max_phase_distance_cycles +
      phase_trajectory_cell_tolerance(options.max_phase_distance_cycles);
  if (!sampled_phase_within(lhs_phase, rhs_phase, maximum_cycles)) {
    return false;
  }
  const PhaseDrift drift = phase_drift(lhs_phase, 1.0L, rhs_phase, 1.0L,
                                       searched_duration_seconds);
  return drift.maximum_cycles <= maximum_cycles;
}

LocalGrouping group_run(std::span<const DmPeak> peaks,
                        std::span<const std::size_t> run_indices,
                        double searched_duration_seconds,
                        const DmPeakGroupingOptions& options,
                        GroupingWorkspace& workspace) {
  const std::size_t count = run_indices.size();
  workspace.reset(count);
  for (std::size_t local_index = 0; local_index < count; ++local_index) {
    workspace.phases[local_index] = make_phase_polynomial(
        peaks[run_indices[local_index]].peak.motion,
        searched_duration_seconds);
  }

  if (options.max_phase_distance_cycles == 0.0) {
    workspace.group_best = workspace.order;
    workspace.assignment = workspace.order;
  } else {
    std::stable_sort(workspace.order.begin(), workspace.order.end(),
                     [&](std::size_t lhs, std::size_t rhs) {
                       return is_better_peak(peaks[run_indices[lhs]],
                                             peaks[run_indices[rhs]]);
                     });

    const double cell_width = options.max_phase_distance_cycles +
                              phase_trajectory_cell_tolerance(
                                  options.max_phase_distance_cycles);
    for (const std::size_t local_index : workspace.order) {
      const DmPeak& peak = peaks[run_indices[local_index]];
      PhaseTrajectoryCell cell{};
      const bool indexed = make_phase_trajectory_cell(
          make_phase_trajectory(peak.peak.motion, searched_duration_seconds),
          cell_width, cell);
      std::size_t selected_group = kNoIndex;
      const auto consider = [&](std::size_t group_index) {
        const std::size_t best_local = workspace.group_best[group_index];
        const DmPeak& best = peaks[run_indices[best_local]];
        if (!locally_compatible(
                peak, workspace.phases[local_index], best,
                workspace.phases[best_local], searched_duration_seconds,
                options)) {
          return;
        }
        if (selected_group == kNoIndex ||
            is_better_peak(best,
                           peaks[run_indices[workspace.group_best[
                               selected_group]]]) ||
            (best.peak.snr ==
                 peaks[run_indices[workspace.group_best[selected_group]]]
                     .peak.snr &&
             group_index < selected_group)) {
          selected_group = group_index;
        }
      };

      for (const std::size_t group_index : workspace.unindexed_groups) {
        consider(group_index);
      }
      if (indexed) {
        for_each_neighbor_phase_trajectory_cell(
            cell, [&](const PhaseTrajectoryCell& neighbor) {
              const auto found = workspace.cell_heads.find(neighbor);
              if (found == workspace.cell_heads.end()) {
                return;
              }
              for (std::size_t group_index = found->second;
                   group_index != kNoIndex;
                   group_index = workspace.next_group[group_index]) {
                consider(group_index);
              }
            });
      } else {
        for (std::size_t group_index = 0;
             group_index < workspace.group_best.size(); ++group_index) {
          consider(group_index);
        }
      }

      if (selected_group == kNoIndex) {
        selected_group = workspace.group_best.size();
        workspace.group_best.push_back(local_index);
        workspace.next_group.push_back(kNoIndex);
        if (indexed) {
          const auto [entry, inserted] =
              workspace.cell_heads.try_emplace(cell, kNoIndex);
          (void)inserted;
          workspace.next_group.back() = entry->second;
          entry->second = selected_group;
        } else {
          workspace.unindexed_groups.push_back(selected_group);
        }
      }
      workspace.assignment[local_index] = selected_group;
    }
  }

  workspace.counts.assign(workspace.group_best.size(), 0);
  for (const std::size_t group_index : workspace.assignment) {
    ++workspace.counts[group_index];
  }
  workspace.begins.resize(workspace.counts.size());
  std::size_t total = 0;
  for (std::size_t group_index = 0;
       group_index < workspace.counts.size(); ++group_index) {
    workspace.begins[group_index] = total;
    total += workspace.counts[group_index];
  }
  workspace.cursors = workspace.begins;

  LocalGrouping result;
  result.member_indices.resize(count);
  result.groups.resize(workspace.counts.size());
  for (std::size_t local_index = 0; local_index < count; ++local_index) {
    const std::size_t group_index = workspace.assignment[local_index];
    const std::size_t destination = workspace.cursors[group_index]++;
    result.member_indices[destination] = run_indices[local_index];
    workspace.output_index[local_index] = destination;
  }
  for (std::size_t group_index = 0;
       group_index < workspace.counts.size(); ++group_index) {
    const std::size_t best_local = workspace.group_best[group_index];
    result.groups[group_index] = PeakGroupRecord{
        .dm_index = peaks[run_indices.front()].dm_index,
        .best_peak_index = run_indices[best_local],
        .best_member_index = workspace.output_index[best_local],
        .member_begin = workspace.begins[group_index],
        .member_count = workspace.counts[group_index],
    };
  }
  return result;
}

}  // namespace

PeakGroupIndex build_peak_group_index_cpu(
    std::span<const DmPeak> peaks,
    double searched_duration_seconds,
    const DmPeakGroupingOptions& options) {
  validate_options(searched_duration_seconds, options);
  for (const DmPeak& peak : peaks) {
    validate_peak(peak);
  }

  PeakGroupIndex result{.raw_peaks = peaks};
  if (peaks.empty()) {
    return result;
  }

  std::vector<std::size_t> peak_order(peaks.size());
  std::iota(peak_order.begin(), peak_order.end(), std::size_t{0});
  if (!std::is_sorted(peak_order.begin(), peak_order.end(),
                      [&](std::size_t lhs, std::size_t rhs) {
                        return peaks[lhs].dm_index < peaks[rhs].dm_index;
                      })) {
    std::stable_sort(peak_order.begin(), peak_order.end(),
                     [&](std::size_t lhs, std::size_t rhs) {
                       return peaks[lhs].dm_index < peaks[rhs].dm_index;
                     });
  }

  std::vector<DmRun> runs;
  for (std::size_t begin = 0; begin < peak_order.size();) {
    const DmPeak& first = peaks[peak_order[begin]];
    std::size_t end = begin + 1;
    while (end < peak_order.size() &&
           peaks[peak_order[end]].dm_index == first.dm_index) {
      if (peaks[peak_order[end]].dm != first.dm) {
        throw std::invalid_argument(
            "DM peak grouping found inconsistent physical DM for one trial");
      }
      ++end;
    }
    runs.push_back(
        {.dm_index = first.dm_index, .begin = begin, .count = end - begin});
    begin = end;
  }

  std::vector<LocalGrouping> local_results(runs.size());
  std::exception_ptr worker_error;
  std::atomic<bool> failed{false};
#ifdef _OPENMP
#pragma omp parallel if(runs.size() > 1)
  {
    GroupingWorkspace workspace;
#pragma omp for schedule(dynamic, 1)
    for (std::ptrdiff_t run_index = 0;
         run_index < static_cast<std::ptrdiff_t>(runs.size()); ++run_index) {
      if (failed.load(std::memory_order_relaxed)) {
        continue;
      }
      try {
        const DmRun& run = runs[static_cast<std::size_t>(run_index)];
        local_results[static_cast<std::size_t>(run_index)] = group_run(
            peaks,
            std::span<const std::size_t>{peak_order}.subspan(run.begin,
                                                              run.count),
            searched_duration_seconds, options, workspace);
      } catch (...) {
        failed.store(true, std::memory_order_relaxed);
#pragma omp critical(peak_grouping_error)
        {
          if (!worker_error) {
            worker_error = std::current_exception();
          }
        }
      }
    }
  }
#else
  GroupingWorkspace workspace;
  for (std::size_t run_index = 0; run_index < runs.size(); ++run_index) {
    const DmRun& run = runs[run_index];
    local_results[run_index] = group_run(
        peaks,
        std::span<const std::size_t>{peak_order}.subspan(run.begin, run.count),
        searched_duration_seconds, options, workspace);
  }
#endif
  if (worker_error) {
    std::rethrow_exception(worker_error);
  }

  result.member_indices.reserve(peaks.size());
  std::size_t group_count = 0;
  for (const LocalGrouping& local : local_results) {
    group_count += local.groups.size();
  }
  result.groups.reserve(group_count);
  result.dm_ranges.reserve(runs.size());
  for (std::size_t run_index = 0; run_index < runs.size(); ++run_index) {
    LocalGrouping& local = local_results[run_index];
    const std::size_t member_begin = result.member_indices.size();
    const std::size_t group_begin = result.groups.size();
    result.member_indices.insert(result.member_indices.end(),
                                 local.member_indices.begin(),
                                 local.member_indices.end());
    for (PeakGroupRecord record : local.groups) {
      record.member_begin += member_begin;
      record.best_member_index += member_begin;
      result.groups.push_back(record);
    }
    result.dm_ranges.push_back(DmGroupRange{
        .dm_index = runs[run_index].dm_index,
        .group_begin = group_begin,
        .group_count = local.groups.size(),
        .member_begin = member_begin,
        .member_count = local.member_indices.size(),
    });
  }
  return result;
}

std::vector<DmPeakGroups> materialize_dm_peak_groups(
    const PeakGroupIndex& index) {
  std::vector<DmPeakGroups> result;
  result.reserve(index.dm_ranges.size());
  for (const DmGroupRange& range : index.dm_ranges) {
    DmPeakGroups output;
    output.members.reserve(range.member_count);
    output.groups.reserve(range.group_count);
    for (std::size_t member = range.member_begin;
         member < range.member_begin + range.member_count; ++member) {
      output.members.push_back(index.raw_peaks[index.member_indices[member]]);
    }
    for (std::size_t group = range.group_begin;
         group < range.group_begin + range.group_count; ++group) {
      const PeakGroupRecord& source = index.groups[group];
      output.groups.push_back(DmPeakGroup{
          .best_index = source.best_member_index - range.member_begin,
          .member_begin = source.member_begin - range.member_begin,
          .member_count = source.member_count,
      });
    }
    result.push_back(std::move(output));
  }
  return result;
}

}  // namespace gaffa::detail
