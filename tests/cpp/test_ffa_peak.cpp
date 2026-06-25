#include "gaffa/ffa_peak.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

gaffa::FfaPeak peak(float snr,
                    double period,
                    std::size_t width,
                    std::size_t phase,
                    std::size_t shift,
                    std::size_t bins) {
  return gaffa::FfaPeak{
      .period = period,
      .frequency = 1.0 / period,
      .width = width,
      .duty_cycle = static_cast<double>(width) / static_cast<double>(bins),
      .phase = phase,
      .shift = shift,
      .bins = bins,
      .snr = snr,
  };
}

}  // namespace

TEST(FfaPeak, OrdersBySnrDescendingThenStableTieBreakers) {
  EXPECT_TRUE(gaffa::is_better_ffa_peak(peak(2.0F, 10.0, 1, 0, 0, 8),
                                        peak(1.0F, 10.0, 1, 0, 0, 8)));
  EXPECT_TRUE(gaffa::is_better_ffa_peak(peak(1.0F, 9.0, 1, 0, 0, 8),
                                        peak(1.0F, 10.0, 1, 0, 0, 8)));
  EXPECT_TRUE(gaffa::is_better_ffa_peak(peak(1.0F, 10.0, 1, 0, 0, 8),
                                        peak(1.0F, 10.0, 2, 0, 0, 8)));
  EXPECT_TRUE(gaffa::is_better_ffa_peak(peak(1.0F, 10.0, 1, 0, 0, 8),
                                        peak(1.0F, 10.0, 1, 1, 0, 8)));
  EXPECT_TRUE(gaffa::is_better_ffa_peak(peak(1.0F, 10.0, 1, 0, 0, 8),
                                        peak(1.0F, 10.0, 1, 0, 1, 8)));
  EXPECT_TRUE(gaffa::is_better_ffa_peak(peak(1.0F, 10.0, 1, 0, 0, 4),
                                        peak(1.0F, 10.0, 1, 0, 0, 8)));
}

TEST(FfaPeak, SortsPeaks) {
  std::vector<gaffa::FfaPeak> peaks{
      peak(1.0F, 10.0, 2, 0, 0, 8),
      peak(3.0F, 12.0, 1, 0, 0, 8),
      peak(3.0F, 11.0, 1, 0, 0, 8),
  };

  gaffa::sort_ffa_peaks(peaks);

  EXPECT_FLOAT_EQ(peaks[0].snr, 3.0F);
  EXPECT_DOUBLE_EQ(peaks[0].period, 11.0);
  EXPECT_DOUBLE_EQ(peaks[1].period, 12.0);
  EXPECT_FLOAT_EQ(peaks[2].snr, 1.0F);
}

TEST(FfaPeak, SortsNonFinitePeaksAfterFinitePeaks) {
  std::vector<gaffa::FfaPeak> peaks{
      peak(NAN, 1.0, 1, 0, 0, 8),
      peak(2.0F, INFINITY, 1, 0, 0, 8),
      peak(1.0F, 10.0, 1, 0, 0, 8),
      peak(3.0F, 11.0, 1, 0, 0, 8),
  };

  gaffa::sort_ffa_peaks(peaks);

  EXPECT_FLOAT_EQ(peaks[0].snr, 3.0F);
  EXPECT_DOUBLE_EQ(peaks[0].period, 11.0);
  EXPECT_FLOAT_EQ(peaks[1].snr, 1.0F);
  EXPECT_DOUBLE_EQ(peaks[1].period, 10.0);
  EXPECT_FALSE(std::isfinite(peaks[2].snr) &&
               std::isfinite(peaks[2].period));
  EXPECT_FALSE(std::isfinite(peaks[3].snr) &&
               std::isfinite(peaks[3].period));
}
