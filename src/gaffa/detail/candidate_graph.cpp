#include "candidate_graph.h"

#include "phase_match.h"
#include "phase_trajectory_grid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gaffa::detail {
namespace {

constexpr std::size_t kEdgeBatchSize = 8192;

class DisjointSet {
 public:
  explicit DisjointSet(std::size_t size) : parent_(size), rank_(size, 0) {
    std::iota(parent_.begin(), parent_.end(), std::size_t{0});
  }

  std::size_t find(std::size_t item) {
    if (parent_[item] != item) {
      parent_[item] = find(parent_[item]);
    }
    return parent_[item];
  }

  void unite(std::size_t lhs, std::size_t rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs) {
      return;
    }
    if (rank_[lhs] < rank_[rhs]) {
      std::swap(lhs, rhs);
    }
    parent_[rhs] = lhs;
    if (rank_[lhs] == rank_[rhs]) {
      ++rank_[lhs];
    }
  }

 private:
  std::vector<std::size_t> parent_;
  std::vector<std::uint8_t> rank_;
};

using PhaseCell = std::array<std::int64_t, 2>;
using SampledTrajectory = std::array<long double, 4>;

struct PhaseCellHash {
  std::size_t operator()(const PhaseCell& cell) const noexcept {
    std::size_t result = std::hash<std::int64_t>{}(cell[0]);
    const std::size_t second = std::hash<std::int64_t>{}(cell[1]);
    result ^= second + 0x9e3779b97f4a7c15ULL + (result << 6U) +
              (result >> 2U);
    return result;
  }
};

struct TrajectoryPoint {
  const DmPeak* representative = nullptr;
  SampledTrajectory samples{};
  PhasePolynomial polynomial{};
  PhaseCell cell{};
};

struct TrajectoryIndex {
  std::vector<TrajectoryPoint> points;
  std::unordered_map<PhaseCell, std::vector<std::size_t>, PhaseCellHash>
      buckets;
  std::array<std::size_t, 2> indexed_coordinates{};
  bool usable = true;
};

bool width_compatible(const DmPeak& lhs,
                      const DmPeak& rhs,
                      bool cluster_across_widths) {
  return cluster_across_widths ||
         (lhs.peak.phase_bins == rhs.peak.phase_bins &&
          lhs.peak.boxcar_width_bins == rhs.peak.boxcar_width_bins);
}

long double evaluate_phase(const PhasePolynomial& polynomial,
                           long double x) {
  long double value = polynomial.back();
  for (std::size_t index = polynomial.size() - 1; index != 0; --index) {
    value = value * x + polynomial[index - 1];
  }
  return value;
}

SampledTrajectory sample_phase_polynomial(
    const PhasePolynomial& polynomial) {
  constexpr std::array<long double, 4> kSamples{
      0.25L, 0.5L, 0.75L, 1.0L};
  SampledTrajectory result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = evaluate_phase(polynomial, kSamples[index]);
  }
  return result;
}

bool sampled_trajectory_within(const SampledTrajectory& lhs,
                               const SampledTrajectory& rhs,
                               double maximum_cycles) {
  for (std::size_t coordinate = 0; coordinate < lhs.size(); ++coordinate) {
    const long double difference =
        std::abs(lhs[coordinate] - rhs[coordinate]);
    const long double tolerance =
        64.0L * std::numeric_limits<long double>::epsilon() *
        std::max(1.0L, difference);
    if (difference > static_cast<long double>(maximum_cycles) + tolerance) {
      return false;
    }
  }
  return true;
}

bool groups_link(const TrajectoryPoint& lhs,
                 const TrajectoryPoint& rhs,
                 double searched_duration_seconds,
                 const CandidateClusteringOptions& options,
                 double maximum_cycles) {
  const DmPeak& lhs_peak = *lhs.representative;
  const DmPeak& rhs_peak = *rhs.representative;
  if (lhs_peak.dm_index == rhs_peak.dm_index ||
      std::abs(lhs_peak.dm - rhs_peak.dm) > options.max_dm_distance ||
      !width_compatible(lhs_peak, rhs_peak,
                        options.cluster_across_widths) ||
      !sampled_trajectory_within(lhs.samples, rhs.samples, maximum_cycles)) {
    return false;
  }
  const PhaseDrift drift = phase_drift(
      lhs.polynomial, 1.0L, rhs.polynomial, 1.0L,
      searched_duration_seconds);
  return drift.maximum_cycles <= maximum_cycles;
}

std::array<std::size_t, 2> choose_indexed_coordinates(
    std::span<const TrajectoryPoint> points) {
  std::array<std::pair<long double, std::size_t>, 4> spreads{};
  for (std::size_t coordinate = 0; coordinate < spreads.size(); ++coordinate) {
    long double minimum = std::numeric_limits<long double>::infinity();
    long double maximum = -std::numeric_limits<long double>::infinity();
    for (const TrajectoryPoint& point : points) {
      minimum = std::min(minimum, point.samples[coordinate]);
      maximum = std::max(maximum, point.samples[coordinate]);
    }
    spreads[coordinate] = {maximum - minimum, coordinate};
  }
  std::stable_sort(spreads.begin(), spreads.end(),
                   [](const auto& lhs, const auto& rhs) {
                     return lhs.first > rhs.first;
                   });
  return {spreads[0].second, spreads[1].second};
}

bool make_phase_cell(long double coordinate,
                     long double cell_width,
                     std::int64_t& cell) {
  const long double lower =
      static_cast<long double>(std::numeric_limits<std::int64_t>::min() + 1);
  const long double upper =
      static_cast<long double>(std::numeric_limits<std::int64_t>::max() - 1);
  const long double scaled = std::floor(coordinate / cell_width);
  if (!std::isfinite(scaled) || scaled < lower || scaled > upper) {
    return false;
  }
  cell = static_cast<std::int64_t>(scaled);
  return true;
}

bool offset_cell_coordinate(std::int64_t value,
                            int offset,
                            std::int64_t& result) {
  if ((offset > 0 &&
       value > std::numeric_limits<std::int64_t>::max() - offset) ||
      (offset < 0 &&
       value < std::numeric_limits<std::int64_t>::min() - offset)) {
    return false;
  }
  result = value + offset;
  return true;
}

TrajectoryIndex build_trajectory_index(
    std::span<const DmPeak* const> representatives,
    double searched_duration_seconds,
    double cell_width) {
  TrajectoryIndex result;
  result.points.reserve(representatives.size());
  for (const DmPeak* representative : representatives) {
    const PhasePolynomial polynomial = make_phase_polynomial(
        representative->peak.motion, searched_duration_seconds);
    result.points.push_back(
        TrajectoryPoint{.representative = representative,
                        .samples = sample_phase_polynomial(polynomial),
                        .polynomial = polynomial});
  }
  result.indexed_coordinates = choose_indexed_coordinates(result.points);

  std::vector<std::size_t> dm_order(result.points.size());
  std::iota(dm_order.begin(), dm_order.end(), std::size_t{0});
  std::stable_sort(dm_order.begin(), dm_order.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     return result.points[lhs].representative->dm <
                            result.points[rhs].representative->dm;
                   });

  result.buckets.reserve(result.points.size());
  for (const std::size_t index : dm_order) {
    TrajectoryPoint& point = result.points[index];
    for (std::size_t dimension = 0; dimension < point.cell.size();
         ++dimension) {
      if (!make_phase_cell(
              point.samples[result.indexed_coordinates[dimension]],
              static_cast<long double>(cell_width),
              point.cell[dimension])) {
        result.usable = false;
        result.buckets.clear();
        return result;
      }
    }
    result.buckets[point.cell].push_back(index);
  }
  return result;
}

template <typename Callback>
void for_each_neighbor_bucket(const TrajectoryIndex& index,
                              const PhaseCell& center,
                              Callback&& callback) {
  for (int first = -1; first <= 1; ++first) {
    for (int second = -1; second <= 1; ++second) {
      PhaseCell neighbor{};
      if (!offset_cell_coordinate(center[0], first, neighbor[0]) ||
          !offset_cell_coordinate(center[1], second, neighbor[1])) {
        continue;
      }
      const auto found = index.buckets.find(neighbor);
      if (found != index.buckets.end()) {
        callback(found->second);
      }
    }
  }
}

void find_indexed_edges(const TrajectoryIndex& index,
                        std::size_t current,
                        double searched_duration_seconds,
                        const CandidateClusteringOptions& options,
                        double maximum_cycles,
                        std::vector<std::size_t>& edges) {
  const TrajectoryPoint& point = index.points[current];
  const double dm_min =
      point.representative->dm - options.max_dm_distance;
  const double dm_max =
      point.representative->dm + options.max_dm_distance;
  for_each_neighbor_bucket(
      index, point.cell, [&](const std::vector<std::size_t>& bucket) {
        const auto first = std::lower_bound(
            bucket.begin(), bucket.end(), dm_min,
            [&](std::size_t group_index, double minimum_dm) {
              return index.points[group_index].representative->dm < minimum_dm;
            });
        const auto last = std::upper_bound(
            first, bucket.end(), dm_max,
            [&](double maximum_dm, std::size_t group_index) {
              return maximum_dm <
                     index.points[group_index].representative->dm;
            });
        for (auto previous = first; previous != last; ++previous) {
          if (*previous >= current) {
            continue;
          }
          if (groups_link(point, index.points[*previous],
                          searched_duration_seconds, options,
                          maximum_cycles)) {
            edges.push_back(*previous);
          }
        }
      });
}

void find_pairwise_edges(const TrajectoryIndex& index,
                         std::size_t current,
                         double searched_duration_seconds,
                         const CandidateClusteringOptions& options,
                         double maximum_cycles,
                         std::vector<std::size_t>& edges) {
  for (std::size_t previous = 0; previous < current; ++previous) {
    if (groups_link(index.points[current], index.points[previous],
                    searched_duration_seconds, options, maximum_cycles)) {
      edges.push_back(previous);
    }
  }
}

std::vector<std::size_t> canonical_labels(DisjointSet& sets,
                                          std::size_t size) {
  std::vector<std::size_t> root_minimum(
      size, std::numeric_limits<std::size_t>::max());
  for (std::size_t index = 0; index < size; ++index) {
    const std::size_t root = sets.find(index);
    root_minimum[root] = std::min(root_minimum[root], index);
  }
  std::vector<std::size_t> labels(size);
  for (std::size_t index = 0; index < size; ++index) {
    labels[index] = root_minimum[sets.find(index)];
  }
  return labels;
}

}  // namespace

std::vector<std::size_t> build_candidate_components_cpu(
    std::span<const DmPeak* const> representatives,
    double searched_duration_seconds,
    const CandidateClusteringOptions& options) {
  if (representatives.empty()) {
    return {};
  }
  const double maximum_cycles =
      options.max_phase_distance_cycles +
      phase_trajectory_cell_tolerance(options.max_phase_distance_cycles);
  const TrajectoryIndex index = build_trajectory_index(
      representatives, searched_duration_seconds, maximum_cycles);
  DisjointSet sets(representatives.size());
  std::vector<std::vector<std::size_t>> batch_edges(
      std::min(kEdgeBatchSize, representatives.size()));

  for (std::size_t batch_begin = 0; batch_begin < representatives.size();
       batch_begin += kEdgeBatchSize) {
    const std::size_t batch_count =
        std::min(kEdgeBatchSize, representatives.size() - batch_begin);
    for (std::size_t offset = 0; offset < batch_count; ++offset) {
      batch_edges[offset].clear();
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(batch_count > 1)
#endif
    for (std::ptrdiff_t offset = 0;
         offset < static_cast<std::ptrdiff_t>(batch_count); ++offset) {
      const std::size_t current = batch_begin +
                                  static_cast<std::size_t>(offset);
      std::vector<std::size_t>& edges =
          batch_edges[static_cast<std::size_t>(offset)];
      if (index.usable) {
        find_indexed_edges(index, current, searched_duration_seconds, options,
                           maximum_cycles, edges);
      } else {
        find_pairwise_edges(index, current, searched_duration_seconds, options,
                            maximum_cycles, edges);
      }
    }

    for (std::size_t offset = 0; offset < batch_count; ++offset) {
      const std::size_t current = batch_begin + offset;
      for (const std::size_t previous : batch_edges[offset]) {
        sets.unite(current, previous);
      }
    }
  }
  return canonical_labels(sets, representatives.size());
}

}  // namespace gaffa::detail
