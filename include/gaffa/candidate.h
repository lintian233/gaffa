#pragma once

#include "gaffa/periodic_peak.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct Candidate {
  // The best-scoring member is the candidate's physical source of truth.
  DmPeak best{};
  std::size_t peak_count = 0;
  CandidateExtent extent{};
};

struct CandidateSelectionOptions {
  // Frequency clustering radius in Fourier-bin units. This is converted to Hz as
  // frequency_cluster_radius / observation_seconds.
  double frequency_cluster_radius = 0.1;
  // DM clustering radius in trial-index units, not physical DM units.
  std::size_t dm_cluster_radius = 1;
  bool cluster_across_widths = true;
  // 0 means unbounded. This is a final candidate cap after clustering/ranking.
  std::size_t max_candidates = 0;
};

std::vector<Candidate> select_candidates_cpu(
    std::span<const DmPeak> peaks,
    double observation_seconds,
    const CandidateSelectionOptions& options = {});

}  // namespace gaffa
