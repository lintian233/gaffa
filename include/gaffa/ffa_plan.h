#pragma once

#include <cstddef>
#include <vector>

namespace gaffa {

// Physical time coordinates of the complete input series searched by an FFA
// plan. Every task in the plan is derived from this one observation.
struct FfaObservation {
  std::size_t nsamples = 0;
  double tsamp_seconds = 0.0;

  [[nodiscard]] double duration_seconds() const noexcept;
  [[nodiscard]] double reference_time_seconds() const noexcept;
};

struct FfaSearchTask {
  double downsample_factor = 1.0;
  double effective_tsamp = 0.0;

  // Number of samples after preparing this task's input. This equals
  // observation.nsamples for downsample_factor == 1 and downsampled_size
  // otherwise.
  std::size_t prepared_nsamples = 0;

  // FFA transform shape for the prepared samples.
  std::size_t bins = 0;
  std::size_t rows = 0;
  // Number of transform rows to expose to detection. The full transform still
  // uses rows x bins internally so trial-period mapping stays correct.
  std::size_t rows_eval = 0;

  double period_begin = 0.0;
  double period_end = 0.0;
};

struct FfaSearchPlan {
  FfaObservation observation{};
  std::vector<FfaSearchTask> tasks;
  std::vector<std::size_t> width_trials;
};

struct RiptideFfaPlanOptions {
  double period_min = 0.0;
  double period_max = 0.0;

  std::size_t bins_min = 0;
  std::size_t bins_max = 0;

  std::size_t min_periods = 1;

  double duty_cycle_max = 0.20;
  double width_trial_spacing = 1.5;

  std::size_t max_tasks = 1'000'000;
};

FfaSearchPlan make_riptide_ffa_plan(
    std::size_t nsamples,
    double tsamp,
    const RiptideFfaPlanOptions& options);

// Validates the immutable-by-contract observation and all task-local values.
// CPU and CUDA execution paths share this validation before allocating or
// launching work.
void validate_ffa_search_plan(const FfaSearchPlan& plan);

}  // namespace gaffa
