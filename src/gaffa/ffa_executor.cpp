#include "gaffa/ffa_executor.h"

#include "gaffa/time_series.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gaffa {
namespace {

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error("FFA executor shape size overflow");
  }
  return lhs * rhs;
}

bool is_no_downsample(double factor) {
  return factor == 1.0;
}

struct PreparedSampleCache {
  bool valid = false;
  double downsample_factor = 0.0;
  std::vector<float> downsampled;
};

void validate_task(const FfaSearchTask& task, std::size_t input_nsamples) {
  if (task.input_nsamples != input_nsamples) {
    throw std::invalid_argument(
        "FFA executor task input_nsamples must match time series size");
  }
  if (!std::isfinite(task.downsample_factor) ||
      task.downsample_factor < 1.0) {
    throw std::invalid_argument(
        "FFA executor task downsample_factor must be finite and >= 1");
  }
  if (!(task.effective_tsamp > 0.0) || !std::isfinite(task.effective_tsamp)) {
    throw std::invalid_argument(
        "FFA executor task effective_tsamp must be finite and > 0");
  }
  if (task.bins <= 1) {
    throw std::invalid_argument("FFA executor task bins must be > 1");
  }
  if (task.rows == 0) {
    throw std::invalid_argument("FFA executor task rows must be > 0");
  }
  if (task.rows_eval == 0 || task.rows_eval > task.rows) {
    throw std::invalid_argument(
        "FFA executor task rows_eval must satisfy 0 < rows_eval <= rows");
  }
  if (task.prepared_nsamples == 0) {
    throw std::invalid_argument(
        "FFA executor task prepared_nsamples must be > 0");
  }

  const std::size_t full_size = checked_multiply(task.rows, task.bins);
  if (full_size > task.prepared_nsamples) {
    throw std::invalid_argument(
        "FFA executor task rows * bins must be <= prepared_nsamples");
  }

  if (is_no_downsample(task.downsample_factor)) {
    if (task.prepared_nsamples != input_nsamples) {
      throw std::invalid_argument(
          "FFA executor no-downsample task prepared_nsamples must match input "
          "size");
    }
    return;
  }

  if (task.prepared_nsamples != downsampled_size(input_nsamples,
                                        task.downsample_factor)) {
    throw std::invalid_argument(
        "FFA executor task prepared_nsamples must match downsampled_size");
  }
}

void validate_inputs(std::span<const float> time_series,
                     const FfaSearchPlan& plan,
                     const FfaBlockConsumer& consumer) {
  if (time_series.empty()) {
    throw std::invalid_argument("FFA executor time series must not be empty");
  }
  if (plan.tasks.empty()) {
    throw std::invalid_argument("FFA executor plan must contain at least one task");
  }
  if (!consumer) {
    throw std::invalid_argument("FFA executor consumer must be callable");
  }
  for (const auto& task : plan.tasks) {
    validate_task(task, time_series.size());
  }
}

std::span<const float> prepare_task_samples(std::span<const float> time_series,
                                            const FfaSearchTask& task,
                                            PreparedSampleCache& cache) {
  if (is_no_downsample(task.downsample_factor)) {
    return time_series.first(task.prepared_nsamples);
  }

  if (!cache.valid || cache.downsample_factor != task.downsample_factor ||
      cache.downsampled.size() != task.prepared_nsamples) {
    cache.downsampled.resize(task.prepared_nsamples);
    downsample_weighted_sum_cpu(time_series, task.downsample_factor,
                                cache.downsampled);
    cache.downsample_factor = task.downsample_factor;
    cache.valid = true;
  }
  return cache.downsampled;
}

float task_stdnoise(std::size_t input_nsamples, const FfaSearchTask& task) {
  double variance = 1.0;
  if (!is_no_downsample(task.downsample_factor)) {
    variance = downsampled_variance(input_nsamples, task.downsample_factor);
  }
  return static_cast<float>(
      std::sqrt(static_cast<double>(task.rows) * variance));
}

}  // namespace

void for_each_ffa_block_cpu(std::span<const float> time_series,
                            const FfaSearchPlan& plan,
                            const FfaBlockConsumer& consumer) {
  validate_inputs(time_series, plan, consumer);

  PreparedSampleCache prepared_cache;
  std::vector<float> scratch;
  std::vector<float> transform;

  for (const auto& task : plan.tasks) {
    const std::span<const float> prepared =
        prepare_task_samples(time_series, task, prepared_cache);

    const std::size_t full_size = checked_multiply(task.rows, task.bins);
    scratch.resize(full_size);
    transform.resize(full_size);

    const FfaTransformShape full_shape{
        .rows = task.rows,
        .bins = task.bins,
    };
    ffa_transform_block_cpu(prepared.first(full_size), full_shape, scratch,
                            transform);

    const std::size_t exposed_size =
        checked_multiply(task.rows_eval, task.bins);
    consumer(FfaBlockView{
        .task = &task,
        .shape = FfaTransformShape{.rows = task.rows_eval, .bins = task.bins},
        .transform = std::span<const float>(transform).first(exposed_size),
        .stdnoise = task_stdnoise(time_series.size(), task),
    });
  }
}

}  // namespace gaffa
