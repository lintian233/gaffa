#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace gaffa {

struct FfaCandidate {
  double period = 0.0;
  std::size_t width = 0;
  std::size_t phase = 0;
  std::size_t shift = 0;
  std::size_t bins = 0;
  float snr = 0.0F;
};

bool is_better_ffa_candidate(const FfaCandidate& lhs,
                             const FfaCandidate& rhs);

void sort_ffa_candidates(std::vector<FfaCandidate>& candidates);

std::vector<FfaCandidate> top_ffa_candidates(
    std::span<const FfaCandidate> candidates,
    std::size_t max_candidates);

class FfaCandidateTopK {
 public:
  explicit FfaCandidateTopK(std::size_t max_candidates);

  void consider(const FfaCandidate& candidate);
  void consider(std::span<const FfaCandidate> candidates);

  std::vector<FfaCandidate> sorted() &&;

 private:
  std::size_t max_candidates_ = 0;
  std::vector<FfaCandidate> candidates_;
};

}  // namespace gaffa
