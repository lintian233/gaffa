#pragma once

#include "gaffa/dedispersion.h"
#include "gaffa/ffa_peak.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/ffa_search.h"
#include "gaffa/periodic_peak.h"
#include "gaffa/preprocessing.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace gaffa {

struct DmFfaOptions {
  PreprocessPlan preprocess{};
  FfaSearchOptions search{};
};

// Runs preprocessing and FFA peak search for every DM row in an eager host
// dedispersion result. The result contains raw DmPeak responses; grouping,
// cross-DM clustering, and candidate filtering belong downstream.
DmPeaks search_dm_ffa_cpu(
    DedispersedResultView<std::uint32_t> input,
    DmTrialView trials,
    const FfaSearchPlan& plan,
    const DmFfaOptions& options = {});

DmPeaks search_dm_ffa_cpu(
    DedispersedResultView<float> input,
    DmTrialView trials,
    const FfaSearchPlan& plan,
    const DmFfaOptions& options = {});

DmPeaks search_dm_ffa_cpu(
    const DedispersedResult<std::uint32_t>& input,
    DmTrialView trials,
    const FfaSearchPlan& plan,
    const DmFfaOptions& options = {});

DmPeaks search_dm_ffa_cpu(
    const DedispersedResult<float>& input,
    DmTrialView trials,
    const FfaSearchPlan& plan,
    const DmFfaOptions& options = {});

}  // namespace gaffa
