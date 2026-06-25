#pragma once

#include "gaffa/dm_search.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct Candidate {
  double dm = 0.0;
  std::size_t dm_index = 0;

  double period = 0.0;
  double frequency = 0.0;
  std::size_t width = 0;
  double duty_cycle = 0.0;
  float snr = 0.0F;

  std::size_t peak_count = 0;
  std::size_t dm_index_min = 0;
  std::size_t dm_index_max = 0;
  double frequency_min = 0.0;
  double frequency_max = 0.0;

  DmPeak best_peak{};
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
