#pragma once

#include <cstddef>
#include <vector>

namespace gaffa {

struct FfaPeak {
  double period = 0.0;
  double frequency = 0.0;
  std::size_t width = 0;
  double duty_cycle = 0.0;
  std::size_t width_index = 0;
  std::size_t period_index = 0;
  std::size_t phase = 0;
  std::size_t shift = 0;
  std::size_t bins = 0;
  float snr = 0.0F;
};

bool is_better_ffa_peak(const FfaPeak& lhs, const FfaPeak& rhs);

void sort_ffa_peaks(std::vector<FfaPeak>& peaks);

}  // namespace gaffa
