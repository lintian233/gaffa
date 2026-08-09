#include "config.h"

#include "config_yaml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace gaffa_search {
namespace {

template <typename T>
T parse_number(std::string_view value, const char* name) {
  try {
    std::string text(value);
    if constexpr (std::is_same_v<T, std::size_t>) {
      std::size_t position = 0;
      const unsigned long long parsed = std::stoull(text, &position);
      if (position != text.size()) {
        throw std::invalid_argument("trailing characters");
      }
      return static_cast<std::size_t>(parsed);
    } else if constexpr (std::is_same_v<T, double>) {
      std::size_t position = 0;
      const double parsed = std::stod(text, &position);
      if (position != text.size()) {
        throw std::invalid_argument("trailing characters");
      }
      return parsed;
    } else {
      static_assert(std::is_same_v<T, float>);
      std::size_t position = 0;
      const float parsed = std::stof(text, &position);
      if (position != text.size()) {
        throw std::invalid_argument("trailing characters");
      }
      return parsed;
    }
  } catch (const std::exception& error) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " +
                                error.what());
  }
}

std::vector<std::string_view> split(std::string_view value, char delimiter) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const std::size_t end = value.find(delimiter, begin);
    result.push_back(value.substr(begin, end == std::string_view::npos
                                           ? value.size() - begin
                                           : end - begin));
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  return result;
}

Backend parse_backend(std::string_view value) {
  if (value == "native-cpu") {
    return Backend::NativeCpu;
  }
  if (value == "native" || value == "native-cuda") {
    return Backend::NativeCuda;
  }
  if (value == "loki-cuda" || value == "loki") {
    return Backend::LokiCuda;
  }
  throw std::invalid_argument("unknown search backend: " +
                              std::string(value));
}

WindowMode parse_window_mode(std::string_view value) {
  if (value == "truncate") {
    return WindowMode::Truncate;
  }
  if (value == "zero-pad") {
    return WindowMode::ZeroPad;
  }
  throw std::invalid_argument("window mode must be truncate or zero-pad");
}

std::vector<int> parse_devices(std::string_view value) {
  const auto parts = split(value, ',');
  if (parts.empty()) {
    throw std::invalid_argument("device list must not be empty");
  }
  std::vector<int> devices;
  devices.reserve(parts.size());
  for (const auto part : parts) {
    const std::size_t parsed = parse_number<std::size_t>(part, "device id");
    if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("device id is too large");
    }
    devices.push_back(static_cast<int>(parsed));
  }
  std::sort(devices.begin(), devices.end());
  devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
  return devices;
}

DmRangeConfig parse_dm_range(std::string_view value) {
  const auto parts = split(value, ':');
  if (parts.size() != 3) {
    throw std::invalid_argument(
        "--dm-range expects dm_low:dm_step:ndm");
  }
  return DmRangeConfig{
      .dm_low = parse_number<double>(parts[0], "dm_low"),
      .dm_step = parse_number<double>(parts[1], "dm_step"),
      .ndm = parse_number<std::size_t>(parts[2], "ndm"),
  };
}

SearchRangeConfig parse_search_range(std::string_view value,
                                     std::size_t id) {
  const auto parts = split(value, ':');
  if (parts.size() != 5 && parts.size() != 6) {
    throw std::invalid_argument(
        "--search expects backend:period_min:period_max:bins_min:bins_max[:window]");
  }
  return SearchRangeConfig{
      .id = id,
      .backend = parse_backend(parts[0]),
      .period_min = parse_number<double>(parts[1], "period_min"),
      .period_max = parse_number<double>(parts[2], "period_max"),
      .bins_min = parse_number<std::size_t>(parts[3], "bins_min"),
      .bins_max = parse_number<std::size_t>(parts[4], "bins_max"),
      .window_mode = parts.size() == 6 ? parse_window_mode(parts[5])
                                       : WindowMode::Truncate,
  };
}

enum class MotionTerm { Accel, Jerk };

struct MotionOverride {
  std::size_t search_id = 0;
  MotionTerm term = MotionTerm::Accel;
  gaffa::ValueRange range{};
};

struct ReductionOverride {
  std::size_t search_id = 0;
  bool top_k = false;
  std::size_t value = 0;
};

void apply_motion_overrides(Config& config,
                            const std::vector<MotionOverride>& overrides) {
  for (const MotionOverride& item : overrides) {
    if (item.search_id >= config.search_ranges.size()) {
      throw std::invalid_argument("motion search id is out of range");
    }
    SearchRangeConfig& search = config.search_ranges[item.search_id];
    std::optional<gaffa::ValueRange>* target =
        item.term == MotionTerm::Accel ? &search.motion.accel
                                       : &search.motion.jerk;
    if (target->has_value()) {
      throw std::invalid_argument("motion range was specified more than once");
    }
    *target = item.range;
  }
}

void apply_reduction_overrides(
    Config& config, const std::vector<ReductionOverride>& overrides) {
  for (const ReductionOverride& item : overrides) {
    if (item.search_id >= config.search_ranges.size()) {
      throw std::invalid_argument("reduction search id is out of range");
    }
    auto& reduction = config.search_ranges[item.search_id].reduction;
    std::size_t* target =
        item.top_k ? &reduction.top_k_per_group
                   : &reduction.max_groups_per_series;
    if (*target != 0) {
      throw std::invalid_argument(
          "reduction option was specified more than once");
    }
    *target = item.value;
  }
}

bool has_motion(const MotionRangeConfig& motion) {
  return motion.accel.has_value() || motion.jerk.has_value();
}

void validate_motion_range(const std::optional<gaffa::ValueRange>& range,
                           const char* name) {
  if (!range) {
    return;
  }
  if (!std::isfinite(range->minimum) || !std::isfinite(range->maximum) ||
      range->maximum <= range->minimum) {
    throw std::invalid_argument(std::string(name) +
                                " range must be finite with max > min");
  }
}

void validate_config_impl(const Config& config) {
  if (config.input.empty()) {
    throw std::invalid_argument("--input is required");
  }
  if (config.dm_ranges.empty()) {
    throw std::invalid_argument("at least one --dm-range is required");
  }
  if (config.search_ranges.empty()) {
    throw std::invalid_argument("at least one --search is required");
  }
  if (config.dm_tile_size == 0 || config.subband_channels == 0 ||
      config.ndm_per_nominal == 0) {
    throw std::invalid_argument("tile and subband sizes must be > 0");
  }
  if (config.resources.native_cuda.max_peak_memory_bytes == 0) {
    throw std::invalid_argument(
        "native CUDA max peak memory must be greater than zero");
  }
  if (!std::isfinite(config.snr_threshold)) {
    throw std::invalid_argument("snr threshold must be finite");
  }
  if (config.candidate_output && config.candidate_output->empty()) {
    throw std::invalid_argument("--cand path must not be empty");
  }
  if (config.dedispersion_device < 0) {
    throw std::invalid_argument("dedispersion device must be non-negative");
  }
  if (config.overwrite_output && !config.candidate_output) {
    throw std::invalid_argument("--overwrite requires --cand");
  }
  if (config.preprocess != "none" && config.preprocess != "normalise" &&
      config.preprocess != "riptide") {
    throw std::invalid_argument(
        "--preprocess must be none, normalise, or riptide");
  }
  if (!(config.running_median_seconds > 0.0) ||
      !std::isfinite(config.running_median_seconds)) {
    throw std::invalid_argument(
        "running median seconds must be finite and > 0");
  }

  double previous_end = -std::numeric_limits<double>::infinity();
  std::size_t previous_global_end = 0;
  for (std::size_t index = 0; index < config.dm_ranges.size(); ++index) {
    const auto& range = config.dm_ranges[index];
    if (range.ndm == 0 || range.dm_low < 0.0 || !(range.dm_step > 0.0) ||
        !std::isfinite(range.dm_low) || !std::isfinite(range.dm_step)) {
      throw std::invalid_argument(
          "DM ranges must have finite non-negative lows and positive steps");
    }
    const double range_end =
        range.dm_low + static_cast<double>(range.ndm) * range.dm_step;
    if (!std::isfinite(range_end)) {
      throw std::invalid_argument("DM range endpoint must be finite");
    }
    if (index != 0 && range.dm_low < previous_end) {
      throw std::invalid_argument("DM ranges must not overlap");
    }
    if (previous_global_end >
        std::numeric_limits<std::size_t>::max() - range.ndm) {
      throw std::overflow_error("DM trial index range overflows size_t");
    }
    previous_global_end += range.ndm;
    previous_end = range_end;
  }
  if (!std::isfinite(config.candidate_dm_radius) ||
      config.candidate_dm_radius < 0.0) {
    throw std::invalid_argument(
        "candidate DM radius must be finite and non-negative");
  }

  for (const auto& search : config.search_ranges) {
    if (!(search.period_min > 0.0) ||
        !(search.period_max > search.period_min) ||
        !std::isfinite(search.period_min) ||
        !std::isfinite(search.period_max) || search.bins_min <= 1 ||
        search.bins_max < search.bins_min) {
      throw std::invalid_argument("invalid search range");
    }
    validate_motion_range(search.motion.accel, "accel");
    validate_motion_range(search.motion.jerk, "jerk");
    if (search.reduction.top_k_per_group != 0 &&
        search.reduction.max_groups_per_series == 0) {
      throw std::invalid_argument(
          "reduction max_groups must be > 0 when top_k is enabled");
    }
    if (!std::isfinite(search.reduction.frequency_tolerance_hz) ||
        search.reduction.frequency_tolerance_hz < 0.0) {
      throw std::invalid_argument(
          "reduction frequency tolerance must be finite and non-negative");
    }
    if (search.motion.jerk && !search.motion.accel) {
      throw std::invalid_argument("jerk search requires accel search");
    }
    if (has_motion(search.motion) && search.backend != Backend::LokiCuda) {
      throw std::invalid_argument(
          "accel and jerk search are currently supported only by loki-cuda");
    }
    if (search.backend == Backend::NativeCuda &&
        config.native_cuda_devices.empty()) {
      throw std::invalid_argument(
          "native-cuda requires --native-devices");
    }
    if (search.backend == Backend::LokiCuda &&
        config.loki_cuda_devices.empty()) {
      throw std::invalid_argument("loki-cuda requires --loki-devices");
    }
  }
}

}  // namespace

void validate_config(const Config& config) {
  validate_config_impl(config);
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n  " << program << " --config FILE\n  " << program
      << " --input FILE_OR_DIR --dm-range LOW:STEP:COUNT"
         " --search BACKEND:PMIN:PMAX:BMIN:BMAX[:WINDOW] [options]\n\n"
      << "Configuration modes are mutually exclusive.\n"
      << "YAML mode reads one complete configuration from FILE.\n\n"
      << "BACKEND: native (Native CUDA), native-cpu, or loki-cuda\n"
      << "WINDOW: truncate or zero-pad; relevant to Loki and defaults to truncate\n"
      << "FILE_OR_DIR: one filterbank file, or a directory whose regular .fil files are scanned\n"
      << "Options:\n"
      << "  --dm-range VALUE          Repeatable DM range\n"
      << "  --search VALUE            Repeatable search range\n"
      << "  --search-accel ID:MIN:MAX Loki acceleration range in m/s^2\n"
      << "  --search-jerk ID:MIN:MAX  Loki jerk range in m/s^3\n"
      << "  --search-top-k ID:N       GPU peak top-K per coordinate group\n"
      << "  --search-max-groups ID:N Maximum coordinate groups per DM\n"
      << "  --dedisp-backend VALUE    cpu-subband or cuda-subband\n"
      << "  --dedisp-device ID        CUDA device for dedispersion\n"
      << "  --native-devices IDS      Comma-separated Native CUDA devices\n"
      << "  --loki-devices IDS        Comma-separated Loki CUDA devices\n"
      << "  --dm-tile-size N          DM rows per GPU work item\n"
      << "  --preprocess VALUE        none, normalise, or riptide\n"
      << "  --median-seconds VALUE    Running median width\n"
      << "  --subband-channels N      Subband channel count\n"
      << "  --ndm-per-nominal N       DM trials per nominal subband\n"
      << "  --snr-threshold VALUE     Raw peak threshold\n"
      << "  --max-peaks N             Per-series raw peak limit; 0 is unlimited\n"
      << "  --max-total-raw-peaks N   Whole-run raw peak limit; 0 is unlimited\n"
      << "  --max-candidates N        Final candidate limit; 0 is unlimited\n"
      << "  --print-candidates N      Number shown on stdout; 0 prints all\n"
      << "  --cand PATH               Output prefix/directory for .cand and .out\n"
      << "  --overwrite               Allow replacing existing output files\n"
      << "  --candidate-dm-radius N   Cross-DM radius in pc cm^-3\n"
      << "  --help                    Show this message\n";
}

Config parse_arguments(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    print_usage(argv[0]);
    std::exit(0);
  }
  bool has_config = false;
  std::string_view config_path;
  for (int index = 1; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--config") {
      if (has_config || index + 1 >= argc || index != 1 || argc != 3) {
        throw std::invalid_argument(
            "--config FILE cannot be combined with other command-line options");
      }
      has_config = true;
      config_path = argv[index + 1];
    }
  }
  if (has_config) {
    return load_yaml_config(config_path);
  }

  Config config;
  std::vector<MotionOverride> motion_overrides;
  std::vector<ReductionOverride> reduction_overrides;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    auto require_value = [&](const char* option) -> std::string_view {
      if (index + 1 >= argc) {
        throw std::invalid_argument(std::string(option) + " requires a value");
      }
      return argv[++index];
    };
    if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (argument == "--input") {
      config.input = require_value("--input");
    } else if (argument == "--dm-range") {
      config.dm_ranges.push_back(parse_dm_range(require_value("--dm-range")));
    } else if (argument == "--search") {
      config.search_ranges.push_back(
          parse_search_range(require_value("--search"),
                             config.search_ranges.size()));
    } else if (argument == "--search-accel" ||
               argument == "--search-jerk") {
      const auto value = require_value(argument == "--search-accel"
                                            ? "--search-accel"
                                            : "--search-jerk");
      const auto parts = split(value, ':');
      if (parts.size() != 3) {
        throw std::invalid_argument(std::string(argument) +
                                    " expects search_id:min:max");
      }
      motion_overrides.push_back(MotionOverride{
          .search_id = parse_number<std::size_t>(parts[0], "search id"),
          .term = argument == "--search-accel" ? MotionTerm::Accel
                                                 : MotionTerm::Jerk,
          .range = gaffa::ValueRange{
              .minimum = parse_number<double>(parts[1], "motion minimum"),
              .maximum = parse_number<double>(parts[2], "motion maximum"),
          },
      });
    } else if (argument == "--search-top-k" ||
               argument == "--search-max-groups") {
      const char* option = argument == "--search-top-k"
                               ? "--search-top-k"
                               : "--search-max-groups";
      const auto parts = split(require_value(option), ':');
      if (parts.size() != 2) {
        throw std::invalid_argument(std::string(option) +
                                    " expects search_id:value");
      }
      reduction_overrides.push_back(ReductionOverride{
          .search_id = parse_number<std::size_t>(parts[0], "search id"),
          .top_k = argument == "--search-top-k",
          .value = parse_number<std::size_t>(parts[1], "reduction value"),
      });
    } else if (argument == "--dedisp-backend") {
      const auto value = require_value("--dedisp-backend");
      if (value == "cpu-subband") {
        config.dedispersion_backend = DedispersionBackend::CpuSubband;
      } else if (value == "cuda-subband") {
        config.dedispersion_backend = DedispersionBackend::CudaSubband;
      } else {
        throw std::invalid_argument(
            "dedisp backend must be cpu-subband or cuda-subband");
      }
    } else if (argument == "--dedisp-device") {
      const std::size_t device = parse_number<std::size_t>(
          require_value("--dedisp-device"), "dedisp device");
      if (device > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("dedisp device id is too large");
      }
      config.dedispersion_device = static_cast<int>(device);
    } else if (argument == "--native-devices") {
      config.native_cuda_devices =
          parse_devices(require_value("--native-devices"));
    } else if (argument == "--loki-devices") {
      config.loki_cuda_devices =
          parse_devices(require_value("--loki-devices"));
    } else if (argument == "--dm-tile-size") {
      config.dm_tile_size =
          parse_number<std::size_t>(require_value("--dm-tile-size"),
                                    "DM tile size");
    } else if (argument == "--preprocess") {
      config.preprocess = require_value("--preprocess");
    } else if (argument == "--median-seconds") {
      config.running_median_seconds =
          parse_number<double>(require_value("--median-seconds"),
                               "median seconds");
    } else if (argument == "--subband-channels") {
      config.subband_channels =
          parse_number<std::size_t>(require_value("--subband-channels"),
                                    "subband channels");
    } else if (argument == "--ndm-per-nominal") {
      config.ndm_per_nominal =
          parse_number<std::size_t>(require_value("--ndm-per-nominal"),
                                    "ndm per nominal");
    } else if (argument == "--snr-threshold") {
      config.snr_threshold =
          parse_number<float>(require_value("--snr-threshold"),
                              "SNR threshold");
    } else if (argument == "--max-peaks") {
      config.max_peaks =
          parse_number<std::size_t>(require_value("--max-peaks"),
                                    "max peaks");
    } else if (argument == "--max-total-raw-peaks") {
      config.max_total_raw_peaks =
          parse_number<std::size_t>(require_value("--max-total-raw-peaks"),
                                    "max total raw peaks");
    } else if (argument == "--max-candidates") {
      config.max_candidates =
          parse_number<std::size_t>(require_value("--max-candidates"),
                                    "max candidates");
    } else if (argument == "--print-candidates") {
      config.print_candidates =
          parse_number<std::size_t>(require_value("--print-candidates"),
                                    "print candidates");
    } else if (argument == "--cand") {
      config.candidate_output =
          std::filesystem::path(require_value("--cand"));
    } else if (argument == "--overwrite") {
      config.overwrite_output = true;
    } else if (argument == "--candidate-dm-radius") {
      config.candidate_dm_radius =
          parse_number<double>(require_value("--candidate-dm-radius"),
                               "candidate DM radius");
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  apply_motion_overrides(config, motion_overrides);
  apply_reduction_overrides(config, reduction_overrides);
  validate_config(config);
  return config;
}

const char* backend_name(Backend backend) noexcept {
  switch (backend) {
    case Backend::NativeCpu:
      return "native-cpu";
    case Backend::NativeCuda:
      return "native-cuda";
    case Backend::LokiCuda:
      return "loki-cuda";
  }
  return "unknown";
}

const char* window_mode_name(WindowMode mode) noexcept {
  return mode == WindowMode::Truncate ? "truncate" : "zero-pad";
}

}  // namespace gaffa_search
