#include "gaffa/candidate.h"

#include "candidate_cluster.h"
#include "candidate_graph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gaffa {
namespace {

struct GroupView {
  const DmPeak* best = nullptr;
  const DmPeak* contiguous_members = nullptr;
  const DmPeak* raw_peaks = nullptr;
  const std::size_t* member_indices = nullptr;
  std::size_t member_count = 0;

  [[nodiscard]] const DmPeak& member(std::size_t index) const {
    return contiguous_members != nullptr
               ? contiguous_members[index]
               : raw_peaks[member_indices[index]];
  }
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
  if (options.max_dm_distance < 0.0 ||
      !std::isfinite(options.max_dm_distance)) {
    throw std::invalid_argument(
        "Candidate max_dm_distance must be finite and >= 0");
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

CandidateExtent empty_extent() {
  return CandidateExtent{
      .dm_index_min = std::numeric_limits<std::size_t>::max(),
      .dm_index_max = 0,
      .frequency_hz = {.minimum = std::numeric_limits<double>::infinity(),
                       .maximum = -std::numeric_limits<double>::infinity()},
      .motion = {
          .acceleration_m_per_s2 = {
              .minimum = std::numeric_limits<double>::infinity(),
              .maximum = -std::numeric_limits<double>::infinity()},
          .jerk_m_per_s3 = {
              .minimum = std::numeric_limits<double>::infinity(),
              .maximum = -std::numeric_limits<double>::infinity()},
          .snap_m_per_s4 = {
              .minimum = std::numeric_limits<double>::infinity(),
              .maximum = -std::numeric_limits<double>::infinity()},
      },
  };
}

void extend(CandidateExtent& extent, const DmPeak& member) {
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

CandidateSet materialize_candidates(std::span<const GroupView> groups,
                                    std::span<const std::size_t> labels,
                                    std::size_t raw_count) {
  std::vector<std::pair<std::size_t, std::size_t>> components;
  components.reserve(groups.size());
  for (std::size_t index = 0; index < groups.size(); ++index) {
    components.emplace_back(labels[index], index);
  }
  std::sort(components.begin(), components.end());

  CandidateSet result;
  result.members.reserve(raw_count);
  result.candidates.reserve(groups.size());
  for (std::size_t begin = 0; begin < components.size();) {
    std::size_t end = begin + 1;
    while (end < components.size() &&
           components[end].first == components[begin].first) {
      ++end;
    }

    const std::size_t member_begin = result.members.size();
    std::size_t best_group = components[begin].second;
    CandidateExtent extent = empty_extent();
    for (std::size_t component = begin; component < end; ++component) {
      const std::size_t group_index = components[component].second;
      const GroupView& group = groups[group_index];
      if (is_better_dm_peak(*group.best, *groups[best_group].best)) {
        best_group = group_index;
      }
      for (std::size_t member_index = 0;
           member_index < group.member_count; ++member_index) {
        const DmPeak& member = group.member(member_index);
        result.members.push_back(member);
        extend(extent, member);
      }
    }
    result.candidates.push_back(Candidate{
        .best = *groups[best_group].best,
        .member_begin = member_begin,
        .member_count = result.members.size() - member_begin,
        .extent = extent,
    });
    begin = end;
  }
  std::sort(result.candidates.begin(), result.candidates.end(),
            is_better_candidate);
  return result;
}

CandidateSet cluster_group_views(
    std::span<const GroupView> groups,
    std::size_t raw_count,
    double searched_duration_seconds,
    const CandidateClusteringOptions& options) {
  if (groups.empty()) {
    return {};
  }
  std::vector<const DmPeak*> representatives;
  representatives.reserve(groups.size());
  for (const GroupView& group : groups) {
    representatives.push_back(group.best);
  }
  const std::vector<std::size_t> labels =
      detail::build_candidate_components_cpu(
          representatives, searched_duration_seconds, options);
  return materialize_candidates(groups, labels, raw_count);
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
  std::vector<GroupView> views;
  std::unordered_set<std::size_t> dm_indices;
  std::size_t raw_count = 0;
  for (const DmPeakGroups& source : peak_groups) {
    if (source.groups.empty()) {
      if (!source.members.empty()) {
        throw std::invalid_argument(
            "Candidate clustering peak groups have members without groups");
      }
      continue;
    }
    if (source.members.empty()) {
      throw std::invalid_argument(
          "Candidate clustering peak groups have groups without members");
    }
    const DmPeak& first = source.members.front();
    validate_peak(first);
    if (!dm_indices.insert(first.dm_index).second) {
      throw std::invalid_argument(
          "Candidate clustering received duplicate DM peak-group inputs");
    }
    std::size_t expected_begin = 0;
    for (const DmPeakGroup& group : source.groups) {
      if (group.member_begin != expected_begin ||
          group.member_begin > source.members.size() ||
          group.member_count == 0 ||
          group.member_count > source.members.size() - group.member_begin ||
          group.best_index < group.member_begin ||
          group.best_index >= group.member_begin + group.member_count) {
        throw std::invalid_argument(
            "Candidate clustering received invalid DM peak groups");
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
        throw std::overflow_error(
            "Candidate clustering raw peak count overflows size_t");
      }
      raw_count += group.member_count;
      views.push_back(GroupView{
          .best = &source.members[group.best_index],
          .contiguous_members = source.members.data() + group.member_begin,
          .member_count = group.member_count,
      });
      expected_begin += group.member_count;
    }
    if (expected_begin != source.members.size()) {
      throw std::invalid_argument(
          "Candidate clustering DM peak groups do not cover all members");
    }
  }
  return cluster_group_views(views, raw_count, searched_duration_seconds,
                             options);
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

namespace detail {

CandidateSet cluster_peak_group_index_cpu(
    const PeakGroupIndex& index,
    double searched_duration_seconds,
    const CandidateClusteringOptions& options) {
  validate_options(searched_duration_seconds, options);
  std::vector<GroupView> views;
  views.reserve(index.groups.size());
  for (const PeakGroupRecord& group : index.groups) {
    views.push_back(GroupView{
        .best = &index.raw_peaks[group.best_peak_index],
        .raw_peaks = index.raw_peaks.data(),
        .member_indices = index.member_indices.data() + group.member_begin,
        .member_count = group.member_count,
    });
  }
  return cluster_group_views(views, index.raw_peaks.size(),
                             searched_duration_seconds, options);
}

}  // namespace detail

}  // namespace gaffa
