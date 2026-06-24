#include "gaffa/ffa_plan.h"

#include "gaffa/time_series.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gaffa {
namespace {

void check_arg(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

void validate_riptide_options(std::size_t nsamples,
                              double tsamp,
                              const RiptideFfaPlanOptions& options) {
  check_arg(nsamples > 0, "FFA plan nsamples must be > 0");
  check_arg(tsamp > 0.0, "FFA plan tsamp must be > 0");
  check_arg(options.period_min > 0.0, "FFA plan period_min must be > 0");
  check_arg(options.period_max > options.period_min,
            "FFA plan period_max must be > period_min");
  check_arg(options.bins_min > 1, "FFA plan bins_min must be > 1");
  check_arg(options.bins_max >= options.bins_min,
            "FFA plan bins_max must be >= bins_min");
  check_arg(options.min_periods > 0, "FFA plan min_periods must be > 0");
  check_arg(options.duty_cycle_max > 0.0 && options.duty_cycle_max < 1.0,
            "FFA plan duty_cycle_max must be in (0, 1)");
  check_arg(options.width_trial_spacing > 1.0,
            "FFA plan width_trial_spacing must be > 1");
  check_arg(options.max_tasks > 0, "FFA plan max_tasks must be > 0");
  check_arg(options.period_min >=
                tsamp * static_cast<double>(options.bins_min),
            "FFA plan period_min must be >= tsamp * bins_min");
}

std::size_t prepared_sample_count(std::size_t nsamples, double factor) {
  if (factor == 1.0) {
    return nsamples;
  }
  return downsampled_size(nsamples, factor);
}

std::size_t ceil_shift(std::size_t rows,
                       std::size_t bins,
                       double period_max_samples) {
  const double value =
      static_cast<double>(bins) * (static_cast<double>(rows) - 1.0) *
      (1.0 - static_cast<double>(bins) / period_max_samples);
  return static_cast<std::size_t>(std::ceil(value));
}

std::vector<std::size_t> generate_width_trials(std::size_t bins_min,
                                               double duty_cycle_max,
                                               double width_trial_spacing) {
  std::vector<std::size_t> widths;
  const auto width_max = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::floor(duty_cycle_max * static_cast<double>(bins_min))));

  for (std::size_t width = 1; width <= width_max;) {
    widths.push_back(width);
    const auto scaled = static_cast<std::size_t>(
        std::floor(width_trial_spacing * static_cast<double>(width)));
    width = std::max(width + 1, scaled);
  }

  return widths;
}

double evaluated_period_end(double effective_tsamp,
                            std::size_t bins,
                            std::size_t rows,
                            std::size_t rows_eval) {
  if (rows == 1 || rows_eval == 1) {
    return effective_tsamp * static_cast<double>(bins);
  }

  const double bins_value = static_cast<double>(bins);
  const double shift =
      static_cast<double>(rows_eval - 1) / static_cast<double>(rows - 1);
  return effective_tsamp * bins_value * bins_value / (bins_value - shift);
}

}  // namespace

FfaSearchPlan make_riptide_ffa_plan(
    std::size_t nsamples,
    double tsamp,
    const RiptideFfaPlanOptions& options) {
  validate_riptide_options(nsamples, tsamp, options);

  const double observation_time = static_cast<double>(nsamples) * tsamp;
  const double effective_period_max =
      std::min(options.period_max,
               observation_time / static_cast<double>(options.min_periods));
  if (effective_period_max <= options.period_min) {
    throw std::invalid_argument(
        "FFA search range is empty after min_periods cap");
  }

  FfaSearchPlan plan;
  plan.width_trials = generate_width_trials(
      options.bins_min, options.duty_cycle_max, options.width_trial_spacing);

  const double initial_downsample_factor =
      options.period_min /
      (tsamp * static_cast<double>(options.bins_min));
  const double downsample_growth =
      (static_cast<double>(options.bins_max) + 1.0) /
      static_cast<double>(options.bins_min);
  const auto cycle_count = static_cast<std::size_t>(
      std::ceil(std::log(effective_period_max / options.period_min) /
                std::log(downsample_growth)));

  for (std::size_t cycle = 0; cycle < cycle_count; ++cycle) {
    const double factor =
        initial_downsample_factor *
        std::pow(downsample_growth, static_cast<double>(cycle));
    const double effective_tsamp = factor * tsamp;
    const std::size_t prepared_nsamples =
        prepared_sample_count(nsamples, factor);
    const double period_max_samples =
        effective_period_max / effective_tsamp;

    const auto bins_stop = std::min(
        {options.bins_max, prepared_nsamples,
         static_cast<std::size_t>(std::floor(period_max_samples))});
    if (bins_stop < options.bins_min) {
      continue;
    }

    for (std::size_t bins = options.bins_min; bins <= bins_stop; ++bins) {
      const std::size_t rows = prepared_nsamples / bins;
      if (rows < options.min_periods) {
        continue;
      }

      const double period_ceil_samples =
          std::min(period_max_samples, static_cast<double>(bins) + 1.0);
      const std::size_t rows_eval =
          std::min(rows, ceil_shift(rows, bins, period_ceil_samples));
      if (rows_eval == 0) {
        continue;
      }

      if (plan.tasks.size() >= options.max_tasks) {
        throw std::runtime_error("FFA search plan exceeds max_tasks");
      }

      plan.tasks.push_back(FfaSearchTask{
          .downsample_factor = factor,
          .effective_tsamp = effective_tsamp,
          .input_nsamples = nsamples,
          .prepared_nsamples = prepared_nsamples,
          .bins = bins,
          .rows = rows,
          .rows_eval = rows_eval,
          .period_begin = effective_tsamp * static_cast<double>(bins),
          .period_end =
              evaluated_period_end(effective_tsamp, bins, rows, rows_eval),
      });
    }
  }

  return plan;
}

}  // namespace gaffa
