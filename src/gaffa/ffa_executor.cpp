#include "gaffa/ffa_executor.h"

#include "gaffa/ffa_detection.h"
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

void validate_inputs(std::span<const float> time_series,
                     const FfaSearchPlan& plan,
                     bool consumer_is_callable) {
  if (time_series.empty()) {
    throw std::invalid_argument("FFA executor time series must not be empty");
  }
  if (!consumer_is_callable) {
    throw std::invalid_argument("FFA executor consumer must be callable");
  }
  validate_ffa_search_plan(plan);
  if (time_series.size() != plan.observation.nsamples) {
    throw std::invalid_argument(
        "FFA executor time series size must match plan observation nsamples");
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

}  // namespace

void for_each_ffa_block_cpu(std::span<const float> time_series,
                            const FfaSearchPlan& plan,
                            const FfaBlockConsumer& consumer) {
  validate_inputs(time_series, plan, static_cast<bool>(consumer));

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
        .stdnoise = ffa_task_stdnoise(plan.observation, task),
    });
  }
}

void for_each_ffa_row_cpu(std::span<const float> time_series,
                          const FfaSearchPlan& plan,
                          const FfaRowConsumer& consumer) {
  validate_inputs(time_series, plan, static_cast<bool>(consumer));

  PreparedSampleCache prepared_cache;
  std::vector<float> scratch;
  std::vector<float> work;
  std::vector<float> row_buffer;

  for (const auto& task : plan.tasks) {
    const std::span<const float> prepared =
        prepare_task_samples(time_series, task, prepared_cache);

    const std::size_t full_size = checked_multiply(task.rows, task.bins);
    scratch.resize(full_size);
    work.resize(full_size);
    row_buffer.resize(task.bins);

    const FfaTransformShape full_shape{
        .rows = task.rows,
        .bins = task.bins,
    };
    const float stdnoise = ffa_task_stdnoise(plan.observation, task);
    for_each_ffa_transform_row_cpu(
        prepared.first(full_size), full_shape, task.rows_eval, scratch, work,
        row_buffer, [&](const FfaTransformRowView& row) {
          consumer(FfaRowView{
              .task = &task,
              .shift = row.shift,
              .profile = row.profile,
              .stdnoise = stdnoise,
          });
        });
  }
}

}  // namespace gaffa
