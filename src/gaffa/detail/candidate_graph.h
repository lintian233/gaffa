#pragma once

#include "gaffa/candidate.h"

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa::detail {

// Builds canonical connected-component labels for prevalidated local-group
// representatives. Labels are group indices and are independent of OpenMP
// scheduling.
std::vector<std::size_t> build_candidate_components_cpu(
    std::span<const DmPeak* const> representatives,
    double searched_duration_seconds,
    const CandidateClusteringOptions& options);

}  // namespace gaffa::detail
