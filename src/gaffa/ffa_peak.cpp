#include "gaffa/ffa_peak.h"

#include <algorithm>
#include <cmath>

namespace gaffa {
namespace {

bool is_finite_peak(const FfaPeak& peak) {
  return std::isfinite(peak.snr) && std::isfinite(peak.period) &&
         std::isfinite(peak.frequency);
}

}  // namespace

bool is_better_ffa_peak(const FfaPeak& lhs, const FfaPeak& rhs) {
  const bool lhs_finite = is_finite_peak(lhs);
  const bool rhs_finite = is_finite_peak(rhs);
  if (lhs_finite != rhs_finite) {
    return lhs_finite;
  }
  if (lhs.snr != rhs.snr) {
    return lhs.snr > rhs.snr;
  }
  if (lhs.period != rhs.period) {
    return lhs.period < rhs.period;
  }
  if (lhs.frequency != rhs.frequency) {
    return lhs.frequency < rhs.frequency;
  }
  if (lhs.width != rhs.width) {
    return lhs.width < rhs.width;
  }
  if (lhs.phase != rhs.phase) {
    return lhs.phase < rhs.phase;
  }
  if (lhs.shift != rhs.shift) {
    return lhs.shift < rhs.shift;
  }
  return lhs.bins < rhs.bins;
}

void sort_ffa_peaks(std::vector<FfaPeak>& peaks) {
  std::sort(peaks.begin(), peaks.end(), is_better_ffa_peak);
}

}  // namespace gaffa
