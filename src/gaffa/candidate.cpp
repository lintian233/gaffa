#include "gaffa/candidate.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gaffa {
namespace {

void validate_options(double observation_seconds,
                      const CandidateSelectionOptions& options) {
  if (!(observation_seconds > 0.0) || !std::isfinite(observation_seconds)) {
    throw std::invalid_argument(
        "Candidate selection observation_seconds must be finite and > 0");
  }
  if (options.frequency_cluster_radius < 0.0 ||
      !std::isfinite(options.frequency_cluster_radius)) {
    throw std::invalid_argument(
        "Candidate selection frequency_cluster_radius must be finite and >= 0");
  }
}

void validate_peak(const DmPeak& peak) {
  if (!std::isfinite(peak.dm) || !std::isfinite(peak.peak.period) ||
      !std::isfinite(peak.peak.frequency) ||
      !std::isfinite(peak.peak.duty_cycle) ||
      !std::isfinite(peak.peak.snr)) {
    throw std::invalid_argument(
        "Candidate selection peaks must contain finite scientific values");
  }
}

bool is_better_dm_peak_for_candidate(const DmPeak& lhs, const DmPeak& rhs) {
  if (is_better_ffa_peak(lhs.peak, rhs.peak)) {
    return true;
  }
  if (is_better_ffa_peak(rhs.peak, lhs.peak)) {
    return false;
  }
  if (lhs.dm_index != rhs.dm_index) {
    return lhs.dm_index < rhs.dm_index;
  }
  return lhs.dm < rhs.dm;
}

bool is_better_candidate(const Candidate& lhs, const Candidate& rhs) {
  if (is_better_dm_peak_for_candidate(lhs.best_peak, rhs.best_peak)) {
    return true;
  }
  if (is_better_dm_peak_for_candidate(rhs.best_peak, lhs.best_peak)) {
    return false;
  }
  if (lhs.peak_count != rhs.peak_count) {
    return lhs.peak_count > rhs.peak_count;
  }
  if (lhs.frequency != rhs.frequency) {
    return lhs.frequency < rhs.frequency;
  }
  return lhs.dm_index < rhs.dm_index;
}

bool peak_order(const DmPeak& lhs, const DmPeak& rhs) {
  if (lhs.peak.frequency != rhs.peak.frequency) {
    return lhs.peak.frequency < rhs.peak.frequency;
  }
  if (lhs.peak.width != rhs.peak.width) {
    return lhs.peak.width < rhs.peak.width;
  }
  if (lhs.dm_index != rhs.dm_index) {
    return lhs.dm_index < rhs.dm_index;
  }
  return is_better_dm_peak_for_candidate(lhs, rhs);
}

bool dm_group_order(const DmPeak& lhs, const DmPeak& rhs) {
  if (lhs.dm_index != rhs.dm_index) {
    return lhs.dm_index < rhs.dm_index;
  }
  if (lhs.peak.width != rhs.peak.width) {
    return lhs.peak.width < rhs.peak.width;
  }
  return lhs.peak.frequency < rhs.peak.frequency;
}

bool width_then_dm_group_order(const DmPeak& lhs, const DmPeak& rhs) {
  if (lhs.peak.width != rhs.peak.width) {
    return lhs.peak.width < rhs.peak.width;
  }
  return dm_group_order(lhs, rhs);
}

bool same_dm_cluster(std::size_t previous_dm_index,
                     std::size_t dm_index,
                     std::size_t dm_cluster_radius) {
  return dm_index <= previous_dm_index ||
         dm_index - previous_dm_index <= dm_cluster_radius;
}

Candidate make_candidate(std::vector<DmPeak>::const_iterator begin,
                         std::vector<DmPeak>::const_iterator end) {
  if (begin == end) {
    throw std::logic_error("Candidate selection cannot build an empty cluster");
  }

  auto best = begin;
  std::size_t dm_index_min = std::numeric_limits<std::size_t>::max();
  std::size_t dm_index_max = 0;
  double frequency_min = std::numeric_limits<double>::infinity();
  double frequency_max = -std::numeric_limits<double>::infinity();

  for (auto current = begin; current != end; ++current) {
    if (is_better_dm_peak_for_candidate(*current, *best)) {
      best = current;
    }
    dm_index_min = std::min(dm_index_min, current->dm_index);
    dm_index_max = std::max(dm_index_max, current->dm_index);
    frequency_min = std::min(frequency_min, current->peak.frequency);
    frequency_max = std::max(frequency_max, current->peak.frequency);
  }

  return Candidate{
      .dm = best->dm,
      .dm_index = best->dm_index,
      .period = best->peak.period,
      .frequency = best->peak.frequency,
      .width = best->peak.width,
      .duty_cycle = best->peak.duty_cycle,
      .snr = best->peak.snr,
      .peak_count = static_cast<std::size_t>(end - begin),
      .dm_index_min = dm_index_min,
      .dm_index_max = dm_index_max,
      .frequency_min = frequency_min,
      .frequency_max = frequency_max,
      .best_peak = *best,
  };
}

void append_dm_clusters(std::vector<DmPeak>::iterator begin,
                        std::vector<DmPeak>::iterator end,
                        const CandidateSelectionOptions& options,
                        std::vector<Candidate>& candidates) {
  if (begin == end) {
    return;
  }

  if (options.cluster_across_widths) {
    std::sort(begin, end, dm_group_order);
  } else {
    std::sort(begin, end, width_then_dm_group_order);
  }

  auto cluster_begin = begin;
  std::size_t previous_dm_index = begin->dm_index;
  std::size_t previous_width = begin->peak.width;
  for (auto current = std::next(begin); current != end; ++current) {
    const bool same_width =
        options.cluster_across_widths || current->peak.width == previous_width;
    const bool same_dm =
        same_dm_cluster(previous_dm_index, current->dm_index,
                        options.dm_cluster_radius);
    if (!same_width || !same_dm) {
      candidates.push_back(make_candidate(cluster_begin, current));
      cluster_begin = current;
    }
    previous_dm_index = current->dm_index;
    previous_width = current->peak.width;
  }
  candidates.push_back(make_candidate(cluster_begin, end));
}

}  // namespace

std::vector<Candidate> select_candidates_cpu(
    std::span<const DmPeak> peaks,
    double observation_seconds,
    const CandidateSelectionOptions& options) {
  validate_options(observation_seconds, options);
  if (peaks.empty()) {
    return {};
  }

  std::vector<DmPeak> sorted_peaks(peaks.begin(), peaks.end());
  for (const DmPeak& peak : sorted_peaks) {
    validate_peak(peak);
  }
  std::sort(sorted_peaks.begin(), sorted_peaks.end(), peak_order);

  const double frequency_radius_hz =
      options.frequency_cluster_radius / observation_seconds;
  std::vector<Candidate> candidates;

  auto frequency_begin = sorted_peaks.begin();
  double previous_frequency = frequency_begin->peak.frequency;
  for (auto current = std::next(sorted_peaks.begin());
       current != sorted_peaks.end(); ++current) {
    if (current->peak.frequency - previous_frequency > frequency_radius_hz) {
      append_dm_clusters(frequency_begin, current, options, candidates);
      frequency_begin = current;
    }
    previous_frequency = current->peak.frequency;
  }
  append_dm_clusters(frequency_begin, sorted_peaks.end(), options, candidates);

  std::sort(candidates.begin(), candidates.end(), is_better_candidate);
  if (options.max_candidates != 0 &&
      candidates.size() > options.max_candidates) {
    candidates.resize(options.max_candidates);
  }
  return candidates;
}

}  // namespace gaffa
