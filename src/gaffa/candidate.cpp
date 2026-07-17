#include "gaffa/candidate.h"

#include "gaffa/periodic_match.h"
#include "detail/phase_trajectory_grid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gaffa {
namespace {

class DisjointSet {
 public:
  explicit DisjointSet(std::size_t size) : parent_(size), rank_(size, 0) {
    std::iota(parent_.begin(), parent_.end(), std::size_t{0});
  }

  std::size_t find(std::size_t item) {
    if (parent_[item] != item) {
      parent_[item] = find(parent_[item]);
    }
    return parent_[item];
  }

  void unite(std::size_t lhs, std::size_t rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs) {
      return;
    }
    if (rank_[lhs] < rank_[rhs]) {
      std::swap(lhs, rhs);
    }
    parent_[rhs] = lhs;
    if (rank_[lhs] == rank_[rhs]) {
      ++rank_[lhs];
    }
  }

 private:
  std::vector<std::size_t> parent_;
  std::vector<std::uint8_t> rank_;
};

struct GroupReference {
  const DmPeakGroups* source = nullptr;
  std::size_t group_index = 0;
};

void validate_options(double searched_duration_seconds,
                      const CandidateClusteringOptions& options) {
  if (!(searched_duration_seconds > 0.0) ||
      !std::isfinite(searched_duration_seconds)) {
    throw std::invalid_argument(
        "Candidate clustering searched_duration_seconds must be finite and > 0");
  }
  if (options.max_phase_distance_cycles < 0.0 ||
      !std::isfinite(options.max_phase_distance_cycles)) {
    throw std::invalid_argument(
        "Candidate max_phase_distance_cycles must be finite and >= 0");
  }
}

void validate_peak(const DmPeak& peak) {
  if (!std::isfinite(peak.dm) || !std::isfinite(peak.peak.duty_cycle) ||
      !std::isfinite(peak.peak.snr)) {
    throw std::invalid_argument(
        "Candidate clustering peaks must contain finite scientific values");
  }
  validate_periodic_motion(peak.peak.motion);
}

bool is_better_dm_peak(const DmPeak& lhs, const DmPeak& rhs) {
  if (lhs.peak.snr != rhs.peak.snr) {
    return lhs.peak.snr > rhs.peak.snr;
  }
  if (lhs.peak.motion.frequency_hz != rhs.peak.motion.frequency_hz) {
    return lhs.peak.motion.frequency_hz < rhs.peak.motion.frequency_hz;
  }
  if (lhs.peak.boxcar_width_bins != rhs.peak.boxcar_width_bins) {
    return lhs.peak.boxcar_width_bins < rhs.peak.boxcar_width_bins;
  }
  if (lhs.peak.phase_bin != rhs.peak.phase_bin) {
    return lhs.peak.phase_bin < rhs.peak.phase_bin;
  }
  if (lhs.dm_index != rhs.dm_index) {
    return lhs.dm_index < rhs.dm_index;
  }
  return lhs.dm < rhs.dm;
}

bool is_better_candidate(const Candidate& lhs, const Candidate& rhs) {
  if (is_better_dm_peak(lhs.best, rhs.best)) {
    return true;
  }
  if (is_better_dm_peak(rhs.best, lhs.best)) {
    return false;
  }
  if (lhs.member_count != rhs.member_count) {
    return lhs.member_count > rhs.member_count;
  }
  return lhs.member_begin < rhs.member_begin;
}

void validate_peak_groups(const DmPeakGroups& source,
                          std::unordered_set<std::size_t>& dm_indices,
                          std::vector<DmPeak>& representatives,
                          std::vector<GroupReference>& references,
                          std::size_t& raw_count) {
  if (source.groups.empty()) {
    if (!source.members.empty()) {
      throw std::invalid_argument(
          "Candidate clustering peak groups have members without groups");
    }
    return;
  }
  std::size_t expected_begin = 0;
  const DmPeak& first = source.members.front();
  validate_peak(first);
  if (!dm_indices.insert(first.dm_index).second) {
    throw std::invalid_argument(
        "Candidate clustering received duplicate DM peak-group inputs");
  }
  for (std::size_t group_index = 0; group_index < source.groups.size();
       ++group_index) {
    const DmPeakGroup& group = source.groups[group_index];
    if (group.member_begin != expected_begin ||
        group.member_begin > source.members.size() || group.member_count == 0 ||
        group.member_count > source.members.size() - group.member_begin ||
        group.best_index < group.member_begin ||
        group.best_index >= group.member_begin + group.member_count) {
      throw std::invalid_argument("Candidate clustering received invalid DM peak groups");
    }
    for (std::size_t index = group.member_begin;
         index < group.member_begin + group.member_count; ++index) {
      const DmPeak& peak = source.members[index];
      validate_peak(peak);
      if (peak.dm_index != first.dm_index || peak.dm != first.dm) {
        throw std::invalid_argument(
            "Candidate clustering DM peak group spans multiple DM trials");
      }
    }
    if (raw_count > std::numeric_limits<std::size_t>::max() -
                        group.member_count) {
      throw std::overflow_error("Candidate clustering raw peak count overflows size_t");
    }
    raw_count += group.member_count;
    representatives.push_back(source.members[group.best_index]);
    references.push_back({.source = &source, .group_index = group_index});
    expected_begin += group.member_count;
  }
  if (expected_begin != source.members.size()) {
    throw std::invalid_argument(
        "Candidate clustering DM peak groups do not cover all members");
  }
}

bool dm_compatible(const DmPeak& lhs,
                   const DmPeak& rhs,
                   std::size_t radius) {
  if (lhs.dm_index == rhs.dm_index) {
    return false;
  }
  const std::size_t lower = std::min(lhs.dm_index, rhs.dm_index);
  const std::size_t upper = std::max(lhs.dm_index, rhs.dm_index);
  return upper - lower <= radius;
}

bool width_compatible(const DmPeak& lhs,
                      const DmPeak& rhs,
                      bool cluster_across_widths) {
  return cluster_across_widths ||
         (lhs.peak.phase_bins == rhs.peak.phase_bins &&
          lhs.peak.boxcar_width_bins == rhs.peak.boxcar_width_bins);
}

bool peaks_link(const DmPeak& lhs,
                const DmPeak& rhs,
                double searched_duration_seconds,
                const CandidateClusteringOptions& options) {
  if (!dm_compatible(lhs, rhs, options.max_dm_index_distance) ||
      !width_compatible(lhs, rhs, options.cluster_across_widths)) {
    return false;
  }
  const PhaseDrift drift = periodic_phase_drift(
      lhs.peak.motion, rhs.peak.motion, searched_duration_seconds);
  return drift.maximum_cycles <=
         options.max_phase_distance_cycles +
             detail::phase_trajectory_cell_tolerance(
                 options.max_phase_distance_cycles);
}

void link_pairwise(std::span<const DmPeak> peaks,
                   double searched_duration_seconds,
                   const CandidateClusteringOptions& options,
                   DisjointSet& sets) {
  for (std::size_t current = 0; current < peaks.size(); ++current) {
    for (std::size_t previous = 0; previous < current; ++previous) {
      if (peaks_link(peaks[current], peaks[previous],
                     searched_duration_seconds, options)) {
        sets.unite(current, previous);
      }
    }
  }
}

bool link_with_phase_trajectory_grid(
    std::span<const DmPeak> peaks,
    double searched_duration_seconds,
    const CandidateClusteringOptions& options,
    DisjointSet& sets) {
  const double cell_width = options.max_phase_distance_cycles +
                            detail::phase_trajectory_cell_tolerance(
                                options.max_phase_distance_cycles);
  std::vector<detail::PhaseTrajectoryCell> cells(peaks.size());
  for (std::size_t index = 0; index < peaks.size(); ++index) {
    if (!detail::make_phase_trajectory_cell(
            detail::make_phase_trajectory(peaks[index].peak.motion,
                                          searched_duration_seconds),
            cell_width, cells[index])) {
      return false;
    }
  }

  std::unordered_map<detail::PhaseTrajectoryCell, std::vector<std::size_t>,
                     detail::PhaseTrajectoryCellHash>
      index;
  index.reserve(peaks.size());
  for (std::size_t current = 0; current < peaks.size(); ++current) {
    detail::for_each_neighbor_phase_trajectory_cell(
        cells[current], [&](const detail::PhaseTrajectoryCell& neighbor) {
          const auto found = index.find(neighbor);
          if (found == index.end()) {
            return;
          }
          for (const std::size_t previous : found->second) {
            if (peaks_link(peaks[current], peaks[previous],
                           searched_duration_seconds, options)) {
              sets.unite(current, previous);
            }
          }
        });
    index[cells[current]].push_back(current);
  }
  return true;
}

CandidateExtent make_extent(std::span<const DmPeak> members) {
  CandidateExtent extent{
      .dm_index_min = std::numeric_limits<std::size_t>::max(),
      .dm_index_max = 0,
      .frequency_hz = {.minimum = std::numeric_limits<double>::infinity(),
                       .maximum = -std::numeric_limits<double>::infinity()},
      .motion = {
          .acceleration_m_per_s2 = {
              .minimum = std::numeric_limits<double>::infinity(),
              .maximum = -std::numeric_limits<double>::infinity()},
          .jerk_m_per_s3 = {.minimum = std::numeric_limits<double>::infinity(),
                            .maximum = -std::numeric_limits<double>::infinity()},
          .snap_m_per_s4 = {.minimum = std::numeric_limits<double>::infinity(),
                            .maximum = -std::numeric_limits<double>::infinity()},
      },
  };
  for (const DmPeak& member : members) {
    extent.dm_index_min = std::min(extent.dm_index_min, member.dm_index);
    extent.dm_index_max = std::max(extent.dm_index_max, member.dm_index);
    extent.frequency_hz.minimum =
        std::min(extent.frequency_hz.minimum, member.peak.motion.frequency_hz);
    extent.frequency_hz.maximum =
        std::max(extent.frequency_hz.maximum, member.peak.motion.frequency_hz);
    extent.motion.acceleration_m_per_s2.minimum = std::min(
        extent.motion.acceleration_m_per_s2.minimum,
        member.peak.motion.acceleration_m_per_s2);
    extent.motion.acceleration_m_per_s2.maximum = std::max(
        extent.motion.acceleration_m_per_s2.maximum,
        member.peak.motion.acceleration_m_per_s2);
    extent.motion.jerk_m_per_s3.minimum = std::min(
        extent.motion.jerk_m_per_s3.minimum, member.peak.motion.jerk_m_per_s3);
    extent.motion.jerk_m_per_s3.maximum = std::max(
        extent.motion.jerk_m_per_s3.maximum, member.peak.motion.jerk_m_per_s3);
    extent.motion.snap_m_per_s4.minimum = std::min(
        extent.motion.snap_m_per_s4.minimum, member.peak.motion.snap_m_per_s4);
    extent.motion.snap_m_per_s4.maximum = std::max(
        extent.motion.snap_m_per_s4.maximum, member.peak.motion.snap_m_per_s4);
  }
  return extent;
}

CandidateSet make_candidate_set(std::span<const DmPeak> representatives,
                                std::span<const GroupReference> references,
                                DisjointSet& sets,
                                std::size_t raw_count) {
  std::map<std::size_t, std::vector<std::size_t>> components;
  for (std::size_t index = 0; index < representatives.size(); ++index) {
    components[sets.find(index)].push_back(index);
  }

  CandidateSet result;
  result.members.reserve(raw_count);
  result.candidates.reserve(components.size());
  for (const auto& [root, indices] : components) {
    (void)root;
    const std::size_t member_begin = result.members.size();
    std::size_t best = indices.front();
    for (const std::size_t index : indices) {
      const DmPeakGroups& source = *references[index].source;
      const auto members = source.members_of(references[index].group_index);
      result.members.insert(result.members.end(), members.begin(), members.end());
      if (is_better_dm_peak(representatives[index], representatives[best])) {
        best = index;
      }
    }
    const std::size_t member_count = result.members.size() - member_begin;
    const std::span<const DmPeak> members{
        result.members.data() + member_begin, member_count};
    result.candidates.push_back(Candidate{
        .best = representatives[best],
        .member_begin = member_begin,
        .member_count = member_count,
        .extent = make_extent(members),
    });
  }
  std::sort(result.candidates.begin(), result.candidates.end(),
            is_better_candidate);
  return result;
}

}  // namespace

std::span<const DmPeak> CandidateSet::members_of(
    const Candidate& candidate) const {
  if (candidate.member_begin > members.size() ||
      candidate.member_count > members.size() - candidate.member_begin) {
    throw std::out_of_range("Candidate member range is outside CandidateSet");
  }
  return std::span<const DmPeak>{members}.subspan(candidate.member_begin,
                                                  candidate.member_count);
}

CandidateSet cluster_dm_peak_groups_cpu(
    std::span<const DmPeakGroups> peak_groups,
    double searched_duration_seconds,
    const CandidateClusteringOptions& options) {
  validate_options(searched_duration_seconds, options);
  std::vector<DmPeak> representatives;
  std::vector<GroupReference> references;
  std::unordered_set<std::size_t> dm_indices;
  std::size_t raw_count = 0;
  for (const DmPeakGroups& groups : peak_groups) {
    validate_peak_groups(groups, dm_indices, representatives, references,
                         raw_count);
  }
  if (representatives.empty()) {
    return {};
  }

  DisjointSet sets(representatives.size());
  if (options.max_phase_distance_cycles == 0.0 ||
      !link_with_phase_trajectory_grid(representatives,
                                       searched_duration_seconds, options,
                                       sets)) {
    link_pairwise(representatives, searched_duration_seconds, options, sets);
  }
  return make_candidate_set(representatives, references, sets, raw_count);
}

std::vector<std::size_t> select_candidates_cpu(
    const CandidateSet& candidates,
    const CandidateSelectionOptions& options) {
  if (!std::isfinite(options.snr_min)) {
    throw std::invalid_argument("Candidate snr_min must be finite");
  }
  std::vector<std::size_t> selected;
  selected.reserve(candidates.candidates.size());
  for (std::size_t index = 0; index < candidates.candidates.size(); ++index) {
    if (candidates.candidates[index].best.peak.snr < options.snr_min) {
      continue;
    }
    selected.push_back(index);
    if (options.max_candidates != 0 &&
        selected.size() >= options.max_candidates) {
      break;
    }
  }
  return selected;
}

}  // namespace gaffa
