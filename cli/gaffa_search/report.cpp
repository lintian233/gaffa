#include "report.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gaffa_search {
namespace {

CandidateRow make_candidate_row(const gaffa::CandidateSet& candidates,
                                std::size_t candidate_id,
                                std::size_t selected_index) {
  if (selected_index >= candidates.candidates.size()) {
    throw std::out_of_range("selected candidate index is outside CandidateSet");
  }
  const gaffa::Candidate& candidate = candidates.candidates[selected_index];
  const auto members = candidates.members_of(candidate);
  if (members.empty()) {
    throw std::invalid_argument("selected candidate has no raw members");
  }

  double dm_min = std::numeric_limits<double>::infinity();
  double dm_max = -std::numeric_limits<double>::infinity();
  for (const gaffa::DmPeak& member : members) {
    dm_min = std::min(dm_min, member.dm);
    dm_max = std::max(dm_max, member.dm);
  }

  const auto& peak = candidate.best.peak;
  return CandidateRow{
      .candidate_id = candidate_id,
      .dm = candidate.best.dm,
      .snr = peak.snr,
      .period_seconds = peak.period_seconds(),
      .frequency_hz = peak.motion.frequency_hz,
      .acceleration_m_per_s2 = peak.motion.acceleration_m_per_s2,
      .jerk_m_per_s3 = peak.motion.jerk_m_per_s3,
      .snap_m_per_s4 = peak.motion.snap_m_per_s4,
      .reference_time_seconds = peak.motion.reference_time_seconds,
      .motion_order = peak.motion.order,
      .phase_bin = peak.phase_bin,
      .phase_bins = peak.phase_bins,
      .boxcar_width_bins = peak.boxcar_width_bins,
      .duty_cycle = peak.duty_cycle,
      .member_count = members.size(),
      .dm_min = dm_min,
      .dm_max = dm_max,
      .dm_index_min = candidate.extent.dm_index_min,
      .dm_index_max = candidate.extent.dm_index_max,
  };
}

bool candidate_row_less(const CandidateRow& lhs, const CandidateRow& rhs) {
  if (lhs.snr != rhs.snr) {
    return lhs.snr > rhs.snr;
  }
  if (lhs.dm != rhs.dm) {
    return lhs.dm < rhs.dm;
  }
  if (lhs.frequency_hz != rhs.frequency_hz) {
    return lhs.frequency_hz < rhs.frequency_hz;
  }
  return lhs.candidate_id < rhs.candidate_id;
}

}  // namespace

const char* motion_order_name(gaffa::MotionOrder order) noexcept {
  switch (order) {
    case gaffa::MotionOrder::Frequency:
      return "frequency";
    case gaffa::MotionOrder::Acceleration:
      return "acceleration";
    case gaffa::MotionOrder::Jerk:
      return "jerk";
    case gaffa::MotionOrder::Snap:
      return "snap";
  }
  return "unknown";
}

Report make_report(const FileResult& result) {
  Report report;
  report.input = result.input;
  report.observation_nsamples = result.observation_nsamples;
  report.tsamp_seconds = result.tsamp_seconds;
  report.observation_seconds = result.observation_seconds;
  report.timing = result.timing;
  report.search_runs = result.search_runs;

  std::size_t candidate_id = 0;
  const auto& candidate_result = result.candidates;
  const auto& candidate_set = candidate_result.candidate_set;
  report.raw_peak_count = result.raw_peak_count;
  report.candidate_count = candidate_set.candidates.size();
  report.selected_count = candidate_result.selected.size();
  report.harmonic_relation_count = candidate_result.harmonic_relations.size();
  report.complete = result.complete;
  report.warnings = result.warnings;

  for (const std::size_t selected_index : candidate_result.selected) {
    report.candidates.push_back(
        make_candidate_row(candidate_set, candidate_id++, selected_index));
  }

  std::sort(report.candidates.begin(), report.candidates.end(),
            candidate_row_less);
  for (std::size_t index = 0; index < report.candidates.size(); ++index) {
    report.candidates[index].rank = index + 1;
  }
  return report;
}

}  // namespace gaffa_search
