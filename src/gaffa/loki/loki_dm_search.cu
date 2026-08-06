#include "gaffa/loki_dm_search.h"

#include <stdexcept>

namespace gaffa {

DmPeaks search_dm_pffa_cuda(LokiPffaProgram& pffa,
                            CudaTimeSeriesBatchView normalised_batch,
                            DmTrialView trials,
                            LokiPffaExecutionOptions options) {
  if (trials.values.size() != normalised_batch.nseries) {
    throw std::invalid_argument(
        "Loki DM trial count must match input batch nseries");
  }
  validate_dm_trials(trials);
  SeriesPeaks peaks = pffa.search_batch(normalised_batch, options);
  return attach_dm_trials(peaks, trials);
}

}  // namespace gaffa
