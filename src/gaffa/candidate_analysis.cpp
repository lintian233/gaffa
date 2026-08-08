#include "gaffa/candidate_analysis.h"

#include "detail/candidate_cluster.h"
#include "detail/peak_group_index.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace gaffa {

CandidateResult make_candidates_cpu(
    std::span<const DmPeak> peaks,
    const HarmonicContext& context,
    const CandidateOptions& options) {
  if (!std::isfinite(options.selection.snr_min)) {
    throw std::invalid_argument("Candidate snr_min must be finite");
  }
  const auto profile_start = std::chrono::steady_clock::now();
  const detail::PeakGroupIndex groups = detail::build_peak_group_index_cpu(
      peaks, context.observation_seconds, options.grouping);
  const auto profile_grouped = std::chrono::steady_clock::now();
  CandidateSet candidate_set = detail::cluster_peak_group_index_cpu(
      groups, context.observation_seconds, options.clustering);
  const auto profile_clustered = std::chrono::steady_clock::now();
  std::vector<HarmonicRelation> harmonic_relations = flag_harmonics_cpu(
      candidate_set, context, options.harmonic);
  const auto profile_harmonics = std::chrono::steady_clock::now();
  std::fprintf(stderr, "candidate_profile grouping=%.6f clustering=%.6f harmonic=%.6f\n",
               std::chrono::duration<double>(profile_grouped - profile_start).count(),
               std::chrono::duration<double>(profile_clustered - profile_grouped).count(),
               std::chrono::duration<double>(profile_harmonics - profile_clustered).count());

  std::vector<std::size_t> selected = remove_harmonics_cpu(
      candidate_set, harmonic_relations);
  selected.erase(
      std::remove_if(selected.begin(), selected.end(),
                     [&](std::size_t index) {
                       return candidate_set.candidates[index].best.peak.snr <
                              options.selection.snr_min;
                     }),
      selected.end());
  if (options.selection.max_candidates != 0 &&
      selected.size() > options.selection.max_candidates) {
    selected.resize(options.selection.max_candidates);
  }

  return CandidateResult{
      .candidate_set = std::move(candidate_set),
      .harmonic_relations = std::move(harmonic_relations),
      .selected = std::move(selected),
  };
}

}  // namespace gaffa
