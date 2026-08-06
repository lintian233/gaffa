#pragma once

#include "gaffa/ffa_cuda.h"
#include "gaffa/periodic_peak.h"
#include "gaffa/preprocessing_cuda.h"
#include "gaffa/time_series_cuda.h"

namespace gaffa {

// Preprocesses one dense device-resident DM tile in place, searches every
// series with the native CUDA FFA backend, and attaches physical DM identity.
// Both programs must use the same device and stream. The function is
// synchronous because compact peaks are returned in host memory.
DmPeaks search_dm_ffa_cuda(
    CudaPreprocessProgram& preprocess,
    CudaFfaProgram& ffa,
    MutableCudaTimeSeriesBatchView batch,
    DmTrialView trials,
    const FfaSearchOptions& options = {});

}  // namespace gaffa
