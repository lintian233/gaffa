#pragma once

#include "gaffa/harmonic.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

// Options for the complete raw-DM-peak to candidate analysis path.
struct CandidateOptions {
  DmPeakGroupingOptions grouping{};
  CandidateClusteringOptions clustering{};
  HarmonicOptions harmonic{};
  CandidateSelectionOptions selection{};
};

// The complete result retains diagnostic candidates and harmonic relations.
// selected contains the final ranked non-harmonic candidate indices.
struct CandidateResult {
  CandidateSet candidate_set;
  std::vector<HarmonicRelation> harmonic_relations;
  std::vector<std::size_t> selected;
};

// Builds candidates from raw peaks from one or more DM trials. It groups
// repeated per-DM responses, clusters them across DMs, records harmonic
// relations, then selects non-harmonic candidates by S/N and final cap.
CandidateResult make_candidates_cpu(
    std::span<const DmPeak> peaks,
    const HarmonicContext& context,
    const CandidateOptions& options = {});

}  // namespace gaffa
