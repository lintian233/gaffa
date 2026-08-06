#pragma once

#include "gaffa/loki_pffa.h"
#include "gaffa/periodic_peak.h"
#include "gaffa/time_series_cuda.h"

namespace gaffa {

// Searches one already-normalised, plan-length DM tile with Loki P-FFA and
// attaches physical DM identity. Truncation or zero-padding belongs before
// this call and must happen after preprocessing.
DmPeaks search_dm_pffa_cuda(
    LokiPffaProgram& pffa,
    CudaTimeSeriesBatchView normalised_batch,
    DmTrialView trials,
    LokiPffaExecutionOptions options = {});

}  // namespace gaffa
