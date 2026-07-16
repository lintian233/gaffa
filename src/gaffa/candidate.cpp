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
  if (!std::isfinite(peak.dm) || !std::isfinite(peak.peak.duty_cycle) ||
      !std::isfinite(peak.peak.snr)) {
    throw std::invalid_argument(
        "Candidate selection peaks must contain finite scientific values");
  }
  validate_periodic_motion(peak.peak.motion);
  if (peak.peak.motion.order != MotionOrder::Frequency) {
    throw std::invalid_argument(
        "Candidate selection currently supports frequency-only periodic peaks");
  }
}

bool is_better_dm_peak_for_candidate(const DmPeak& lhs, const DmPeak& rhs) {
  if (lhs.peak.snr != rhs.peak.snr) {
    return lhs.peak.snr > rhs.peak.snr;
  }
  if (lhs.peak.motion.frequency_hz != rhs.peak.motion.frequency_hz) {
    return lhs.peak.motion.frequency_hz < rhs.peak.motion.frequency_hz;
  }
  if (lhs.peak.boxcar_width_bins != rhs.peak.boxcar_width_bins) {
    return lhs.peak.boxcar_width_bins < rhs.peak.boxcar_width_bins;
  }
  if (lhs.peak.phase_bin != rhs.peak.phase_bin) {
    return lhs.peak.phase_bin < rhs.peak.phase_bin;
  }
  if (lhs.dm_index != rhs.dm_index) {
    return lhs.dm_index < rhs.dm_index;
  }
  return lhs.dm < rhs.dm;
}

bool is_better_candidate(const Candidate& lhs, const Candidate& rhs) {
  if (is_better_dm_peak_for_candidate(lhs.best, rhs.best)) {
    return true;
  }
  if (is_better_dm_peak_for_candidate(rhs.best, lhs.best)) {
    return false;
  }
  if (lhs.peak_count != rhs.peak_count) {
    return lhs.peak_count > rhs.peak_count;
  }
  if (lhs.best.peak.motion.frequency_hz != rhs.best.peak.motion.frequency_hz) {
    return lhs.best.peak.motion.frequency_hz <
           rhs.best.peak.motion.frequency_hz;
  }
  return lhs.best.dm_index < rhs.best.dm_index;
}

bool peak_order(const DmPeak& lhs, const DmPeak& rhs) {
  if (lhs.peak.motion.frequency_hz != rhs.peak.motion.frequency_hz) {
    return lhs.peak.motion.frequency_hz < rhs.peak.motion.frequency_hz;
  }
  if (lhs.peak.boxcar_width_bins != rhs.peak.boxcar_width_bins) {
    return lhs.peak.boxcar_width_bins < rhs.peak.boxcar_width_bins;
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
  if (lhs.peak.boxcar_width_bins != rhs.peak.boxcar_width_bins) {
    return lhs.peak.boxcar_width_bins < rhs.peak.boxcar_width_bins;
  }
  return lhs.peak.motion.frequency_hz < rhs.peak.motion.frequency_hz;
}

bool width_then_dm_group_order(const DmPeak& lhs, const DmPeak& rhs) {
  if (lhs.peak.boxcar_width_bins != rhs.peak.boxcar_width_bins) {
    return lhs.peak.boxcar_width_bins < rhs.peak.boxcar_width_bins;
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
  CandidateExtent extent{
      .dm_index_min = std::numeric_limits<std::size_t>::max(),
      .dm_index_max = 0,
      .frequency_hz = {.minimum = std::numeric_limits<double>::infinity(),
                       .maximum = -std::numeric_limits<double>::infinity()},
      .motion = {
          .acceleration_m_per_s2 = {
              .minimum = std::numeric_limits<double>::infinity(),
              .maximum = -std::numeric_limits<double>::infinity()},
          .jerk_m_per_s3 = {.minimum = std::numeric_limits<double>::infinity(),
                            .maximum = -std::numeric_limits<double>::infinity()},
          .snap_m_per_s4 = {.minimum = std::numeric_limits<double>::infinity(),
                            .maximum = -std::numeric_limits<double>::infinity()},
      },
  };

  for (auto current = begin; current != end; ++current) {
    if (is_better_dm_peak_for_candidate(*current, *best)) {
      best = current;
    }
    extent.dm_index_min = std::min(extent.dm_index_min, current->dm_index);
    extent.dm_index_max = std::max(extent.dm_index_max, current->dm_index);
    extent.frequency_hz.minimum = std::min(
        extent.frequency_hz.minimum, current->peak.motion.frequency_hz);
    extent.frequency_hz.maximum = std::max(
        extent.frequency_hz.maximum, current->peak.motion.frequency_hz);
    extent.motion.acceleration_m_per_s2.minimum = std::min(
        extent.motion.acceleration_m_per_s2.minimum,
        current->peak.motion.acceleration_m_per_s2);
    extent.motion.acceleration_m_per_s2.maximum = std::max(
        extent.motion.acceleration_m_per_s2.maximum,
        current->peak.motion.acceleration_m_per_s2);
    extent.motion.jerk_m_per_s3.minimum = std::min(
        extent.motion.jerk_m_per_s3.minimum,
        current->peak.motion.jerk_m_per_s3);
    extent.motion.jerk_m_per_s3.maximum = std::max(
        extent.motion.jerk_m_per_s3.maximum,
        current->peak.motion.jerk_m_per_s3);
    extent.motion.snap_m_per_s4.minimum = std::min(
        extent.motion.snap_m_per_s4.minimum,
        current->peak.motion.snap_m_per_s4);
    extent.motion.snap_m_per_s4.maximum = std::max(
        extent.motion.snap_m_per_s4.maximum,
        current->peak.motion.snap_m_per_s4);
  }

  return Candidate{
      .best = *best,
      .peak_count = static_cast<std::size_t>(end - begin),
      .extent = extent,
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
  std::size_t previous_width = begin->peak.boxcar_width_bins;
  for (auto current = std::next(begin); current != end; ++current) {
    const bool same_width =
        options.cluster_across_widths ||
        current->peak.boxcar_width_bins == previous_width;
    const bool same_dm =
        same_dm_cluster(previous_dm_index, current->dm_index,
                        options.dm_cluster_radius);
    if (!same_width || !same_dm) {
      candidates.push_back(make_candidate(cluster_begin, current));
      cluster_begin = current;
    }
    previous_dm_index = current->dm_index;
    previous_width = current->peak.boxcar_width_bins;
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
  double previous_frequency = frequency_begin->peak.motion.frequency_hz;
  for (auto current = std::next(sorted_peaks.begin());
       current != sorted_peaks.end(); ++current) {
    if (current->peak.motion.frequency_hz - previous_frequency >
        frequency_radius_hz) {
      append_dm_clusters(frequency_begin, current, options, candidates);
      frequency_begin = current;
    }
    previous_frequency = current->peak.motion.frequency_hz;
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
