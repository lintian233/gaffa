#pragma once

#include <cstddef>
#include <vector>

namespace gaffa {

struct FfaSearchTask {
  double downsample_factor = 1.0;
  double effective_tsamp = 0.0;

  std::size_t input_nsamples = 0;
  std::size_t nsamples = 0;

  std::size_t bins = 0;
  std::size_t rows = 0;
  std::size_t rows_eval = 0;

  double period_begin = 0.0;
  double period_end = 0.0;
};

struct FfaSearchPlan {
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

}  // namespace gaffa
