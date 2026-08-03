#include "gaffa/candidate_analysis.h"

#include <algorithm>
#include <utility>

namespace gaffa {

CandidateResult make_candidates_cpu(
    std::span<const DmPeak> peaks,
    const HarmonicContext& context,
    const CandidateOptions& options) {
  const std::vector<DmPeakGroups> groups = group_dm_peak_batch_cpu(
      peaks, context.observation_seconds, options.grouping);
  CandidateSet candidate_set = cluster_dm_peak_groups_cpu(
      groups, context.observation_seconds, options.clustering);
  std::vector<HarmonicRelation> harmonic_relations = flag_harmonics_cpu(
      candidate_set, context, options.harmonic);

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
