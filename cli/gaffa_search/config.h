#pragma once

#include "gaffa/periodic_peak.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gaffa_search {

enum class Backend {
  NativeCpu,
  NativeCuda,
  LokiCuda,
};

enum class DedispersionBackend {
  CpuSubband,
  CudaSubband,
};

enum class WindowMode {
  Truncate,
  ZeroPad,
};

struct DmRangeConfig {
  double dm_low = 0.0;
  double dm_step = 1.0;
  std::size_t ndm = 0;
};

struct MotionRangeConfig {
  // Values use SI units: accel is m/s^2 and jerk is m/s^3. An absent range
  // means that the corresponding Taylor coefficient is not searched.
  std::optional<gaffa::ValueRange> accel;
  std::optional<gaffa::ValueRange> jerk;
};

struct SearchRangeConfig {
  std::size_t id = 0;
  Backend backend = Backend::NativeCpu;
  double period_min = 0.0;
  double period_max = 0.0;
  std::size_t bins_min = 0;
  std::size_t bins_max = 0;
  WindowMode window_mode = WindowMode::Truncate;
  MotionRangeConfig motion{};
};

struct Config {
  std::filesystem::path input;
  std::vector<DmRangeConfig> dm_ranges;
  std::vector<SearchRangeConfig> search_ranges;

  DedispersionBackend dedispersion_backend =
      DedispersionBackend::CudaSubband;
  int dedispersion_device = 0;
  std::vector<int> native_cuda_devices;
  std::vector<int> loki_cuda_devices;

  std::size_t dm_tile_size = 16;
  std::string preprocess = "riptide";
  double running_median_seconds = 5.0;
  std::size_t subband_channels = 32;
  std::size_t ndm_per_nominal = 32;

  float snr_threshold = 7.5F;
  std::size_t max_peaks = 0;
  std::size_t max_total_raw_peaks = 0;
  std::size_t max_candidates = 0;
  std::size_t print_candidates = 64;
  // Physical DM distance in pc cm^-3 used for candidate clustering.
  double candidate_dm_radius = 25.0;

  // When set, this is an output prefix for a single input file, or an output
  // directory for directory input. The CLI writes both .cand and .out.
  std::optional<std::filesystem::path> candidate_output;
  bool overwrite_output = false;
};

Config parse_arguments(int argc, char** argv);
void print_usage(const char* program);
void validate_config(const Config& config);

const char* backend_name(Backend backend) noexcept;
const char* window_mode_name(WindowMode mode) noexcept;

}  // namespace gaffa_search
