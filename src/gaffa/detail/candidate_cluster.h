#pragma once

#include "gaffa/candidate.h"
#include "peak_group_index.h"

namespace gaffa::detail {

CandidateSet cluster_peak_group_index_cpu(
    const PeakGroupIndex& index,
    double searched_duration_seconds,
    const CandidateClusteringOptions& options);

}  // namespace gaffa::detail
