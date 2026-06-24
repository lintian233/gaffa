#include "gaffa/ffa_candidate.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gaffa {
namespace {

struct WorstCandidateFirst {
  bool operator()(const FfaCandidate& lhs, const FfaCandidate& rhs) const {
    return is_better_ffa_candidate(lhs, rhs);
  }
};

bool is_finite_candidate(const FfaCandidate& candidate) {
  return std::isfinite(candidate.snr) && std::isfinite(candidate.period);
}

}  // namespace

bool is_better_ffa_candidate(const FfaCandidate& lhs,
                             const FfaCandidate& rhs) {
  const bool lhs_finite = is_finite_candidate(lhs);
  const bool rhs_finite = is_finite_candidate(rhs);
  if (lhs_finite != rhs_finite) {
    return lhs_finite;
  }
  if (lhs.snr != rhs.snr) {
    return lhs.snr > rhs.snr;
  }
  if (lhs.period != rhs.period) {
    return lhs.period < rhs.period;
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

void sort_ffa_candidates(std::vector<FfaCandidate>& candidates) {
  std::sort(candidates.begin(), candidates.end(), is_better_ffa_candidate);
}

std::vector<FfaCandidate> top_ffa_candidates(
    std::span<const FfaCandidate> candidates,
    std::size_t max_candidates) {
  FfaCandidateTopK top_k(max_candidates);
  top_k.consider(candidates);
  return std::move(top_k).sorted();
}

FfaCandidateTopK::FfaCandidateTopK(std::size_t max_candidates)
    : max_candidates_(max_candidates) {
  if (max_candidates == 0) {
    throw std::invalid_argument("FFA candidate max_candidates must be > 0");
  }
  candidates_.reserve(max_candidates);
}

void FfaCandidateTopK::consider(const FfaCandidate& candidate) {
  if (candidates_.size() < max_candidates_) {
    candidates_.push_back(candidate);
    std::push_heap(candidates_.begin(), candidates_.end(),
                   WorstCandidateFirst{});
    return;
  }
  if (is_better_ffa_candidate(candidate, candidates_.front())) {
    std::pop_heap(candidates_.begin(), candidates_.end(),
                  WorstCandidateFirst{});
    candidates_.back() = candidate;
    std::push_heap(candidates_.begin(), candidates_.end(),
                   WorstCandidateFirst{});
  }
}

void FfaCandidateTopK::consider(std::span<const FfaCandidate> candidates) {
  for (const auto& candidate : candidates) {
    consider(candidate);
  }
}

std::vector<FfaCandidate> FfaCandidateTopK::sorted() && {
  sort_ffa_candidates(candidates_);
  return std::move(candidates_);
}

}  // namespace gaffa
