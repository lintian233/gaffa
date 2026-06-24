#include "gaffa/preprocessing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gaffa {
namespace {

void validate_running_median_options(DetrendRunningMedianOptions options) {
  if (options.window_samples == 0 || options.window_samples % 2 == 0) {
    throw std::invalid_argument(
        "running median detrend window_samples must be odd and > 0");
  }
  if (options.min_points == 0 || options.min_points % 2 == 0) {
    throw std::invalid_argument(
        "running median detrend min_points must be odd and > 0");
  }
}

void validate_preprocess_plan(const PreprocessPlan& plan) {
  if (plan.steps.empty()) {
    throw std::invalid_argument("preprocess plan must contain at least one step");
  }
}

void validate_positive_finite(double value, const char* message) {
  if (!(value > 0.0) || !std::isfinite(value)) {
    throw std::invalid_argument(message);
  }
}

void validate_finite_samples(std::span<const float> input) {
  for (const float value : input) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "running median detrend input samples must be finite");
    }
  }
}

float median_of_window(std::span<const float> window_input,
                       std::vector<float>& window) {
  window.assign(window_input.begin(), window_input.end());
  const std::size_t middle = window.size() / 2;
  std::nth_element(window.begin(), window.begin() + middle, window.end());
  const float upper = window[middle];
  return upper;
}

void exact_running_median_cpu(std::span<const float> input,
                              std::size_t window_samples,
                              std::span<float> output) {
  if (window_samples >= input.size()) {
    throw std::invalid_argument(
        "running median detrend window_samples must be < input size");
  }

  const std::size_t half_window = window_samples / 2;
  std::vector<float> padded(input.size() + 2 * half_window);
  std::fill_n(padded.begin(), half_window, input.front());
  std::copy(input.begin(), input.end(),
            padded.begin() + static_cast<std::ptrdiff_t>(half_window));
  std::fill(padded.begin() + static_cast<std::ptrdiff_t>(half_window +
                                                         input.size()),
            padded.end(), input.back());

  std::vector<float> window;
  window.reserve(window_samples);
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = median_of_window(
        std::span<const float>(padded).subspan(index, window_samples), window);
  }
}

std::vector<float> mean_scrunch_cpu(std::span<const float> input,
                                    std::size_t factor) {
  const std::size_t output_size = input.size() / factor;
  std::vector<float> output(output_size);
  for (std::size_t output_index = 0; output_index < output_size;
       ++output_index) {
    double sum = 0.0;
    const std::size_t input_begin = output_index * factor;
    for (std::size_t index = 0; index < factor; ++index) {
      sum += static_cast<double>(input[input_begin + index]);
    }
    output[output_index] = static_cast<float>(sum / static_cast<double>(factor));
  }
  return output;
}

void interpolate_scrunched_median(std::span<const float> low_res,
                                  std::size_t scrunch_factor,
                                  std::span<float> output) {
  if (low_res.empty()) {
    throw std::invalid_argument(
        "running median detrend low-resolution median must not be empty");
  }

  const double offset = 0.5 * static_cast<double>(scrunch_factor - 1);
  for (std::size_t index = 0; index < output.size(); ++index) {
    const double position =
        (static_cast<double>(index) - offset) /
        static_cast<double>(scrunch_factor);
    if (position <= 0.0) {
      output[index] = low_res.front();
      continue;
    }
    const auto lower = static_cast<std::size_t>(std::floor(position));
    if (lower >= low_res.size() - 1) {
      output[index] = low_res.back();
      continue;
    }

    const double fraction = position - static_cast<double>(lower);
    output[index] = static_cast<float>(
        (1.0 - fraction) * static_cast<double>(low_res[lower]) +
        fraction * static_cast<double>(low_res[lower + 1]));
  }
}

void fast_running_median_cpu(std::span<const float> input,
                             DetrendRunningMedianOptions options,
                             std::span<float> output) {
  const std::size_t scrunch_factor = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             static_cast<double>(options.window_samples) /
             static_cast<double>(options.min_points)));
  if (scrunch_factor == 1) {
    exact_running_median_cpu(input, options.window_samples, output);
    return;
  }

  const std::vector<float> scrunched = mean_scrunch_cpu(input, scrunch_factor);
  if (scrunched.size() <= options.min_points) {
    throw std::invalid_argument(
        "running median detrend low-resolution sample count must be > "
        "min_points");
  }
  std::vector<float> low_res_median(scrunched.size());
  exact_running_median_cpu(scrunched, options.min_points, low_res_median);
  interpolate_scrunched_median(low_res_median, scrunch_factor, output);
}

void apply_preprocess_step(const PreprocessStep& step,
                           std::span<const float> input,
                           std::vector<float>& output) {
  output.resize(input.size());
  switch (step.kind) {
    case PreprocessStepKind::DetrendRunningMedian:
      detrend_running_median_cpu(input, step.detrend_running_median, output);
      return;
    case PreprocessStepKind::Normalise:
      normalise_cpu(input, output, step.normalise);
      return;
  }
  throw std::invalid_argument("unknown preprocess step kind");
}

}  // namespace

DetrendRunningMedianOptions running_median_window_seconds(
    double window_seconds,
    double tsamp) {
  validate_positive_finite(
      window_seconds,
      "running median window_seconds must be finite and > 0");
  validate_positive_finite(tsamp, "running median tsamp must be finite and > 0");

  const double rounded_samples = std::round(window_seconds / tsamp);
  constexpr auto max_size = static_cast<double>(
      std::numeric_limits<std::size_t>::max());
  if (!std::isfinite(rounded_samples) || rounded_samples > max_size - 1.0) {
    throw std::invalid_argument("running median window_samples is too large");
  }

  auto window_samples =
      static_cast<std::size_t>(std::max(1.0, rounded_samples));
  if (window_samples % 2 == 0) {
    ++window_samples;
  }

  return DetrendRunningMedianOptions{
      .window_samples = window_samples,
  };
}

PreprocessPlan make_riptide_preprocess_plan(
    double tsamp,
    const RiptidePreprocessOptions& options) {
  PreprocessPlan plan;
  plan.steps.push_back(PreprocessStep{
      .kind = PreprocessStepKind::DetrendRunningMedian,
      .detrend_running_median = running_median_window_seconds(
          options.running_median_width_seconds, tsamp),
  });
  plan.steps.back().detrend_running_median.min_points =
      options.running_median_min_points;

  if (options.normalise) {
    plan.steps.push_back(PreprocessStep{
        .kind = PreprocessStepKind::Normalise,
        .normalise = options.normalise_options,
    });
  }

  return plan;
}

void detrend_running_median_cpu(std::span<const float> input,
                                DetrendRunningMedianOptions options,
                                std::span<float> output) {
  validate_running_median_options(options);
  if (input.empty()) {
    throw std::invalid_argument(
        "running median detrend input must not be empty");
  }
  if (output.size() != input.size()) {
    throw std::invalid_argument(
        "running median detrend output size must match input size");
  }
  validate_finite_samples(input);

  std::vector<float> running_median(input.size());
  fast_running_median_cpu(input, options, running_median);
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index] - running_median[index];
  }
}

std::vector<float> detrend_running_median_cpu(
    std::span<const float> input,
    DetrendRunningMedianOptions options) {
  std::vector<float> output(input.size());
  detrend_running_median_cpu(input, options, output);
  return output;
}

TimeSeries preprocess_time_series_cpu(const TimeSeries& input,
                                      const PreprocessPlan& plan) {
  validate_time_series(input);
  validate_preprocess_plan(plan);

  std::vector<float> current = input.data;
  std::vector<float> next;
  for (const auto& step : plan.steps) {
    apply_preprocess_step(step, current, next);
    current.swap(next);
  }

  return TimeSeries{
      .data = std::move(current),
      .tsamp = input.tsamp,
  };
}

void preprocess_time_series_inplace_cpu(TimeSeries& input,
                                        const PreprocessPlan& plan) {
  input = preprocess_time_series_cpu(input, plan);
}

}  // namespace gaffa
