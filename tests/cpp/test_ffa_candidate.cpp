#include "gaffa/ffa_candidate.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

gaffa::FfaCandidate candidate(float snr,
                              double period,
                              std::size_t width,
                              std::size_t phase,
                              std::size_t shift,
                              std::size_t bins) {
  return gaffa::FfaCandidate{
      .period = period,
      .width = width,
      .phase = phase,
      .shift = shift,
      .bins = bins,
      .snr = snr,
  };
}

}  // namespace

TEST(FfaCandidate, OrdersBySnrDescendingThenStableTieBreakers) {
  EXPECT_TRUE(gaffa::is_better_ffa_candidate(
      candidate(2.0F, 10.0, 1, 0, 0, 8),
      candidate(1.0F, 10.0, 1, 0, 0, 8)));
  EXPECT_TRUE(gaffa::is_better_ffa_candidate(
      candidate(1.0F, 9.0, 1, 0, 0, 8),
      candidate(1.0F, 10.0, 1, 0, 0, 8)));
  EXPECT_TRUE(gaffa::is_better_ffa_candidate(
      candidate(1.0F, 10.0, 1, 0, 0, 8),
      candidate(1.0F, 10.0, 2, 0, 0, 8)));
  EXPECT_TRUE(gaffa::is_better_ffa_candidate(
      candidate(1.0F, 10.0, 1, 0, 0, 8),
      candidate(1.0F, 10.0, 1, 1, 0, 8)));
  EXPECT_TRUE(gaffa::is_better_ffa_candidate(
      candidate(1.0F, 10.0, 1, 0, 0, 8),
      candidate(1.0F, 10.0, 1, 0, 1, 8)));
  EXPECT_TRUE(gaffa::is_better_ffa_candidate(
      candidate(1.0F, 10.0, 1, 0, 0, 4),
      candidate(1.0F, 10.0, 1, 0, 0, 8)));
}

TEST(FfaCandidate, SortsCandidates) {
  std::vector<gaffa::FfaCandidate> candidates{
      candidate(1.0F, 10.0, 2, 0, 0, 8),
      candidate(3.0F, 12.0, 1, 0, 0, 8),
      candidate(3.0F, 11.0, 1, 0, 0, 8),
  };

  gaffa::sort_ffa_candidates(candidates);

  EXPECT_FLOAT_EQ(candidates[0].snr, 3.0F);
  EXPECT_DOUBLE_EQ(candidates[0].period, 11.0);
  EXPECT_DOUBLE_EQ(candidates[1].period, 12.0);
  EXPECT_FLOAT_EQ(candidates[2].snr, 1.0F);
}

TEST(FfaCandidate, SortsNonFiniteCandidatesAfterFiniteCandidates) {
  std::vector<gaffa::FfaCandidate> candidates{
      candidate(NAN, 1.0, 1, 0, 0, 8),
      candidate(2.0F, INFINITY, 1, 0, 0, 8),
      candidate(1.0F, 10.0, 1, 0, 0, 8),
      candidate(3.0F, 11.0, 1, 0, 0, 8),
  };

  gaffa::sort_ffa_candidates(candidates);

  EXPECT_FLOAT_EQ(candidates[0].snr, 3.0F);
  EXPECT_DOUBLE_EQ(candidates[0].period, 11.0);
  EXPECT_FLOAT_EQ(candidates[1].snr, 1.0F);
  EXPECT_DOUBLE_EQ(candidates[1].period, 10.0);
  EXPECT_FALSE(std::isfinite(candidates[2].snr) &&
               std::isfinite(candidates[2].period));
  EXPECT_FALSE(std::isfinite(candidates[3].snr) &&
               std::isfinite(candidates[3].period));
}

TEST(FfaCandidate, TopCandidatesKeepsOnlyBest) {
  const std::vector<gaffa::FfaCandidate> candidates{
      candidate(1.0F, 10.0, 1, 0, 0, 8),
      candidate(5.0F, 10.0, 1, 0, 0, 8),
      candidate(4.0F, 9.0, 1, 0, 0, 8),
      candidate(5.0F, 9.0, 1, 0, 0, 8),
  };

  const auto top = gaffa::top_ffa_candidates(candidates, 2);

  ASSERT_EQ(top.size(), 2);
  EXPECT_FLOAT_EQ(top[0].snr, 5.0F);
  EXPECT_DOUBLE_EQ(top[0].period, 9.0);
  EXPECT_FLOAT_EQ(top[1].snr, 5.0F);
  EXPECT_DOUBLE_EQ(top[1].period, 10.0);
}

TEST(FfaCandidate, TopKAcceptsCandidateSpans) {
  const std::vector<gaffa::FfaCandidate> candidates{
      candidate(1.0F, 10.0, 1, 0, 0, 8),
      candidate(2.0F, 10.0, 1, 0, 0, 8),
      candidate(3.0F, 10.0, 1, 0, 0, 8),
  };
  gaffa::FfaCandidateTopK top_k(2);

  top_k.consider(candidates);
  const auto top = std::move(top_k).sorted();

  ASSERT_EQ(top.size(), 2);
  EXPECT_FLOAT_EQ(top[0].snr, 3.0F);
  EXPECT_FLOAT_EQ(top[1].snr, 2.0F);
}

TEST(FfaCandidate, RejectsZeroTopKLimit) {
  EXPECT_THROW((void)gaffa::FfaCandidateTopK(0), std::invalid_argument);
  EXPECT_THROW((void)gaffa::top_ffa_candidates({}, 0), std::invalid_argument);
}
