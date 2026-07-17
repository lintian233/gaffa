#include "gaffa/loki_pffa.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace gaffa {
namespace {

bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

void validate_range(const ValueRange& range, const char* name,
                    bool positive, bool require_nonzero_width) {
  if (!std::isfinite(range.minimum) || !std::isfinite(range.maximum) ||
      range.maximum < range.minimum ||
      (require_nonzero_width && range.maximum == range.minimum) ||
      (positive && range.minimum <= 0.0)) {
    throw std::invalid_argument(std::string("invalid Loki ") + name +
                                " search range");
  }
}

}  // namespace

LokiPffaPlan::LokiPffaPlan(std::size_t input_nsamples, double tsamp_seconds,
                           LokiTaylorSearchSpace search_space,
                           LokiPffaPlanOptions options)
    : input_nsamples_(input_nsamples),
      tsamp_seconds_(tsamp_seconds),
      search_space_(std::move(search_space)),
      options_(options) {}

LokiPffaPlan::LokiPffaPlan(LokiPffaPlan&&) noexcept = default;
LokiPffaPlan& LokiPffaPlan::operator=(LokiPffaPlan&&) noexcept = default;

std::size_t LokiPffaPlan::input_nsamples() const noexcept {
  return input_nsamples_;
}

double LokiPffaPlan::tsamp_seconds() const noexcept {
  return tsamp_seconds_;
}

const LokiTaylorSearchSpace& LokiPffaPlan::search_space() const noexcept {
  return search_space_;
}

const LokiPffaPlanOptions& LokiPffaPlan::options() const noexcept {
  return options_;
}

LokiPffaPlan make_loki_pffa_plan(std::size_t input_nsamples,
                                  double tsamp_seconds,
                                  LokiTaylorSearchSpace search_space,
                                  LokiPffaPlanOptions options) {
  if (!is_power_of_two(input_nsamples)) {
    throw std::invalid_argument(
        "Loki P-FFA input_nsamples must be a non-zero power of two");
  }
  if (!(tsamp_seconds > 0.0) || !std::isfinite(tsamp_seconds)) {
    throw std::invalid_argument(
        "Loki P-FFA tsamp_seconds must be finite and > 0");
  }
  validate_range(search_space.frequency_hz, "frequency", true,
                 /*require_nonzero_width=*/true);
  if (search_space.jerk_m_per_s3.has_value() &&
      !search_space.acceleration_m_per_s2.has_value()) {
    throw std::invalid_argument("Loki jerk search requires acceleration search");
  }
  if (search_space.snap_m_per_s4.has_value()) {
    throw std::invalid_argument(
        "Loki snap search is not supported by the current Loki region planner");
  }
  if (search_space.acceleration_m_per_s2.has_value()) {
    validate_range(*search_space.acceleration_m_per_s2,
                   "acceleration_m_per_s2", false,
                   /*require_nonzero_width=*/true);
  }
  if (search_space.jerk_m_per_s3.has_value()) {
    validate_range(*search_space.jerk_m_per_s3, "jerk_m_per_s3", false,
                   /*require_nonzero_width=*/true);
  }
  if (options.phase_bins_min < 2 ||
      options.phase_bins_max < options.phase_bins_min || !(options.eta > 0.0) ||
      !(options.duty_cycle_max > 0.0 && options.duty_cycle_max <= 1.0) ||
      !(options.width_spacing > 1.0) || !(options.snr_threshold > 0.0F) ||
      !std::isfinite(options.eta) || !std::isfinite(options.duty_cycle_max) ||
      !std::isfinite(options.width_spacing) ||
      !std::isfinite(options.snr_threshold)) {
    throw std::invalid_argument("invalid Loki P-FFA plan options");
  }
  return LokiPffaPlan(input_nsamples, tsamp_seconds, std::move(search_space),
                      options);
}

}  // namespace gaffa
