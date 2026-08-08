#include "gaffa/candidate_analysis.h"

#include "detail/candidate_cluster.h"
#include "detail/peak_group_index.h"

#include <algorithm>
#include <cmath>
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
  const detail::PeakGroupIndex groups = detail::build_peak_group_index_cpu(
      peaks, context.observation_seconds, options.grouping);
  CandidateSet candidate_set = detail::cluster_peak_group_index_cpu(
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
