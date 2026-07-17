#pragma once

#include "gaffa/peak_grouping.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct Candidate {
  // The best-scoring member is the candidate's physical source of truth.
  DmPeak best{};
  std::size_t member_begin = 0;
  std::size_t member_count = 0;
  CandidateExtent extent{};
};

// Owns every raw peak exactly once. Candidate member ranges refer to contiguous
// slices of members and remain valid while this object is not modified.
struct CandidateSet {
  std::vector<DmPeak> members;
  std::vector<Candidate> candidates;

  [[nodiscard]] std::span<const DmPeak> members_of(
      const Candidate& candidate) const;
};

struct CandidateClusteringOptions {
  // Maximum phase-trajectory separation over the observation, in cycles.
  double max_phase_distance_cycles = 0.1;
  // DM graph-edge radius in trial-index units, not physical DM units.
  std::size_t max_dm_index_distance = 1;
  bool cluster_across_widths = true;
};

struct CandidateSelectionOptions {
  float snr_min = 0.0F;
  // 0 means unbounded. This cap is applied after clustering and ranking.
  std::size_t max_candidates = 0;
};

// Builds cross-DM connected components from local DM peak groups. Only each
// group's best peak participates in graph construction; every raw DmPeak from
// selected groups is retained in the returned CandidateSet.
CandidateSet cluster_dm_peak_groups_cpu(
    std::span<const DmPeakGroups> peak_groups,
    double searched_duration_seconds,
    const CandidateClusteringOptions& options = {});

// Returns ranked indices into CandidateSet::candidates without copying members.
std::vector<std::size_t> select_candidates_cpu(
    const CandidateSet& candidates,
    const CandidateSelectionOptions& options = {});

}  // namespace gaffa
