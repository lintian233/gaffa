#include "gaffa/dm_search_cuda.h"

#include <exception>
#include <stdexcept>
#include <vector>

namespace gaffa {
namespace {

void validate_dm_ffa_cuda_inputs(const CudaPreprocessProgram& preprocess,
                                 const CudaFfaProgram& ffa,
                                 MutableCudaTimeSeriesBatchView batch,
                                 DmTrialView trials) {
  if (batch.data == nullptr || batch.nseries == 0 || batch.nsamples == 0) {
    throw std::invalid_argument("CUDA DM FFA batch must not be empty");
  }
  if (trials.values.size() != batch.nseries) {
    throw std::invalid_argument(
        "CUDA DM FFA trial count must match batch nseries");
  }
  validate_dm_trials(trials);
  if (preprocess.device_id() != batch.device_id ||
      ffa.device_id() != batch.device_id) {
    throw std::invalid_argument(
        "CUDA DM FFA batch and programs must use the same device");
  }
  if (preprocess.stream() != ffa.stream()) {
    throw std::invalid_argument(
        "CUDA DM FFA preprocessing and search must use the same stream");
  }
  if (batch.nseries > preprocess.tile_capacity() ||
      batch.nseries > ffa.tile_capacity()) {
    throw std::invalid_argument(
        "CUDA DM FFA batch exceeds a program tile capacity");
  }
  if (batch.nsamples > preprocess.max_nsamples() ||
      batch.nsamples != ffa.execution_plan().observation().nsamples) {
    throw std::invalid_argument(
        "CUDA DM FFA batch length does not match program contracts");
  }
}

}  // namespace

DmPeaks search_dm_ffa_cuda(CudaPreprocessProgram& preprocess,
                           CudaFfaProgram& ffa,
                           MutableCudaTimeSeriesBatchView batch,
                           DmTrialView trials,
                           const FfaSearchOptions& options) {
  validate_dm_ffa_cuda_inputs(preprocess, ffa, batch, trials);
  preprocess_time_series_batch_inplace_cuda(preprocess, batch);

  SeriesPeaks periodic;
  try {
    periodic = search_ffa_batch_cuda(ffa, batch.as_const(), options);
  } catch (...) {
    const std::exception_ptr search_error = std::current_exception();
    preprocess.synchronize();
    std::rethrow_exception(search_error);
  }
  preprocess.synchronize();

  return attach_dm_trials(periodic, trials);
}

}  // namespace gaffa
