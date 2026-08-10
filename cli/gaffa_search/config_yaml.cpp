#include "config_yaml.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace gaffa_search {
namespace {

void require_map(const YAML::Node& node, std::string_view path) {
  if (!node || !node.IsMap()) {
    throw std::invalid_argument(std::string(path) + " must be a mapping");
  }
}

void require_sequence(const YAML::Node& node, std::string_view path) {
  if (!node || !node.IsSequence()) {
    throw std::invalid_argument(std::string(path) + " must be a sequence");
  }
}

YAML::Node required_node(const YAML::Node& parent, const char* key,
                         std::string_view path) {
  const YAML::Node node = parent[key];
  if (!node) {
    throw std::invalid_argument(std::string(path) + " is required");
  }
  return node;
}

void reject_unknown_keys(const YAML::Node& node,
                         std::initializer_list<const char*> allowed,
                         std::string_view path) {
  for (const auto& entry : node) {
    if (!entry.first.IsScalar()) {
      throw std::invalid_argument(std::string(path) + " contains a non-scalar key");
    }
    const std::string key = entry.first.as<std::string>();
    const bool known = std::any_of(
        allowed.begin(), allowed.end(),
        [&key](const char* candidate) { return key == candidate; });
    if (!known) {
      throw std::invalid_argument(std::string(path) + " contains unknown key '" +
                                  key + "'");
    }
  }
}

template <typename T>
T scalar(const YAML::Node& node, std::string_view path) {
  if (!node || !node.IsScalar()) {
    throw std::invalid_argument(std::string(path) + " must be a scalar");
  }
  try {
    return node.as<T>();
  } catch (const YAML::Exception& error) {
    throw std::invalid_argument(std::string(path) + " has invalid value: " +
                                error.what());
  }
}

template <typename T>
T optional_scalar(const YAML::Node& parent, const char* key, T fallback,
                 std::string_view path) {
  const YAML::Node node = parent[key];
  return node ? scalar<T>(node, path) : fallback;
}

template <typename T>
std::optional<T> optional_nullable_scalar(const YAML::Node& parent,
                                          const char* key,
                                          std::string_view path) {
  const YAML::Node node = parent[key];
  if (!node || node.IsNull()) {
    return std::nullopt;
  }
  return scalar<T>(node, path);
}

std::size_t nonnegative_size(const YAML::Node& node, std::string_view path) {
  const long long value = scalar<long long>(node, path);
  if (value < 0) {
    throw std::invalid_argument(std::string(path) +
                                " must be a non-negative integer");
  }
  return static_cast<std::size_t>(value);
}

std::size_t byte_size(const YAML::Node& node, std::string_view path) {
  const std::string text = scalar<std::string>(node, path);
  struct Unit {
    std::string_view suffix;
    std::size_t multiplier;
  };
  constexpr std::array<Unit, 7> units = {{
      {"KiB", 1024ULL},
      {"MiB", 1024ULL * 1024ULL},
      {"GiB", 1024ULL * 1024ULL * 1024ULL},
      {"KB", 1000ULL},
      {"MB", 1000ULL * 1000ULL},
      {"GB", 1000ULL * 1000ULL * 1000ULL},
      {"B", 1ULL},
  }};

  const Unit* selected = nullptr;
  for (const Unit& unit : units) {
    if (text.size() >= unit.suffix.size() &&
        std::string_view(text).ends_with(unit.suffix)) {
      selected = &unit;
      break;
    }
  }
  if (selected == nullptr) {
    throw std::invalid_argument(std::string(path) +
                                " must use a size suffix (B, KB, MB, GB, "
                                "KiB, MiB, or GiB)");
  }

  const std::string_view number(
      text.data(), text.size() - selected->suffix.size());
  if (number.empty() ||
      !std::all_of(number.begin(), number.end(), [](unsigned char value) {
        return std::isdigit(value) != 0;
      })) {
    throw std::invalid_argument(std::string(path) + " has invalid size '" +
                                text + "'");
  }

  unsigned long long value = 0;
  try {
    std::size_t position = 0;
    value = std::stoull(std::string(number), &position);
    if (position != number.size()) {
      throw std::invalid_argument("trailing characters");
    }
  } catch (const std::exception& error) {
    throw std::invalid_argument(std::string(path) + " has invalid size '" +
                                text + "': " + error.what());
  }

  const unsigned long long multiplier = selected->multiplier;
  if (value > std::numeric_limits<std::size_t>::max() / multiplier) {
    throw std::invalid_argument(std::string(path) + " is too large");
  }
  const unsigned long long bytes = value * multiplier;
  if (bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string(path) + " is too large");
  }
  return static_cast<std::size_t>(bytes);
}

std::filesystem::path resolve_path(const std::filesystem::path& config_path,
                                   std::string_view value) {
  const std::filesystem::path path(value);
  if (path.is_absolute()) {
    return path;
  }
  return config_path.parent_path() / path;
}

Backend parse_backend_value(std::string_view value, std::string_view path) {
  if (value == "native-cpu") {
    return Backend::NativeCpu;
  }
  if (value == "native-cuda") {
    return Backend::NativeCuda;
  }
  if (value == "loki-cuda") {
    return Backend::LokiCuda;
  }
  throw std::invalid_argument(std::string(path) + " has unknown backend '" +
                              std::string(value) + "'");
}

DedispersionBackend parse_dedispersion_backend_value(
    std::string_view value, std::string_view path) {
  if (value == "cpu-subband") {
    return DedispersionBackend::CpuSubband;
  }
  if (value == "cuda-subband") {
    return DedispersionBackend::CudaSubband;
  }
  throw std::invalid_argument(std::string(path) +
                              " has unknown dedispersion backend '" +
                              std::string(value) + "'");
}

WindowMode parse_window_value(std::string_view value, std::string_view path) {
  if (value == "truncate") {
    return WindowMode::Truncate;
  }
  if (value == "zero-pad") {
    return WindowMode::ZeroPad;
  }
  throw std::invalid_argument(std::string(path) + " has unknown window '" +
                              std::string(value) + "'");
}

std::vector<int> parse_devices(const YAML::Node& node, std::string_view path) {
  require_sequence(node, path);
  if (node.size() == 0) {
    throw std::invalid_argument(std::string(path) + " must not be empty");
  }
  std::vector<int> devices;
  devices.reserve(node.size());
  for (std::size_t index = 0; index < node.size(); ++index) {
    const std::string item_path =
        std::string(path) + "[" + std::to_string(index) + "]";
    const long long value = scalar<long long>(node[index], item_path);
    if (value < 0 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(item_path + " must be a non-negative int");
    }
    devices.push_back(static_cast<int>(value));
  }
  std::sort(devices.begin(), devices.end());
  devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
  return devices;
}

DmRangeConfig parse_dm_range(const YAML::Node& node, std::size_t index) {
  const std::string path = "dm_ranges[" + std::to_string(index) + "]";
  require_map(node, path);
  reject_unknown_keys(node, {"low", "step", "count"}, path);
  return DmRangeConfig{
      .dm_low = scalar<double>(required_node(node, "low", path + ".low"),
                               path + ".low"),
      .dm_step = scalar<double>(required_node(node, "step", path + ".step"),
                                path + ".step"),
      .ndm = nonnegative_size(
          required_node(node, "count", path + ".count"), path + ".count"),
  };
}

std::optional<gaffa::ValueRange> parse_optional_range(
    const YAML::Node& parent, const char* key, std::string_view path) {
  const YAML::Node node = parent[key];
  if (!node) {
    return std::nullopt;
  }
  require_map(node, path);
  reject_unknown_keys(node, {"min", "max"}, path);
  return gaffa::ValueRange{
      .minimum = scalar<double>(required_node(node, "min",
                                              std::string(path) + ".min"),
                                std::string(path) + ".min"),
      .maximum = scalar<double>(required_node(node, "max",
                                              std::string(path) + ".max"),
                                std::string(path) + ".max"),
  };
}

SearchRangeConfig parse_search_range(const YAML::Node& node,
                                     std::size_t index) {
  const std::string path = "search.ranges[" + std::to_string(index) + "]";
  require_map(node, path);
  reject_unknown_keys(node,
                      {"backend", "period", "bins", "accel", "jerk",
                       "window", "duty_cycle_max", "width_trial_spacing",
                       "reduction"},
                      path);

  const std::string backend_path = path + ".backend";
  const Backend backend = parse_backend_value(
      scalar<std::string>(required_node(node, "backend", backend_path),
                          backend_path),
      backend_path);

  const YAML::Node period = required_node(node, "period", path + ".period");
  require_map(period, path + ".period");
  reject_unknown_keys(period, {"min", "max"}, path + ".period");
  const YAML::Node bins = required_node(node, "bins", path + ".bins");
  require_map(bins, path + ".bins");
  reject_unknown_keys(bins, {"min", "max"}, path + ".bins");

  const YAML::Node window = node["window"];
  if (window && backend != Backend::LokiCuda) {
    throw std::invalid_argument(path +
                                ".window is only valid for loki-cuda");
  }

  gaffa::PeakReductionOptions reduction{};
  const YAML::Node reduction_node = node["reduction"];
  if (reduction_node) {
    require_map(reduction_node, path + ".reduction");
    reject_unknown_keys(reduction_node,
                        {"top_k", "max_groups", "frequency_tolerance_hz",
                         "phase_tolerance_cycles"},
                        path + ".reduction");
    if (reduction_node["top_k"]) {
      reduction.top_k_per_group = nonnegative_size(
          reduction_node["top_k"], path + ".reduction.top_k");
    }
    if (reduction_node["max_groups"]) {
      reduction.max_groups_per_series = nonnegative_size(
          reduction_node["max_groups"], path + ".reduction.max_groups");
    }
    if (reduction_node["frequency_tolerance_hz"]) {
      reduction.frequency_tolerance_hz = scalar<double>(
          reduction_node["frequency_tolerance_hz"],
          path + ".reduction.frequency_tolerance_hz");
    }
    if (reduction_node["phase_tolerance_cycles"]) {
      reduction.phase_tolerance_cycles = scalar<double>(
          reduction_node["phase_tolerance_cycles"],
          path + ".reduction.phase_tolerance_cycles");
    }
  }

  return SearchRangeConfig{
      .id = index,
      .backend = backend,
      .period_min = scalar<double>(required_node(
          period, "min", path + ".period.min"), path + ".period.min"),
      .period_max = scalar<double>(required_node(
          period, "max", path + ".period.max"), path + ".period.max"),
      .bins_min = nonnegative_size(
          required_node(bins, "min", path + ".bins.min"),
          path + ".bins.min"),
      .bins_max = nonnegative_size(
          required_node(bins, "max", path + ".bins.max"),
          path + ".bins.max"),
      .duty_cycle_max = optional_scalar<double>(
          node, "duty_cycle_max", 0.20, path + ".duty_cycle_max"),
      .width_trial_spacing = optional_scalar<double>(
          node, "width_trial_spacing", 1.5,
          path + ".width_trial_spacing"),
      .window_mode = window
                          ? parse_window_value(
                                scalar<std::string>(window, path + ".window"),
                                path + ".window")
                          : WindowMode::Truncate,
      .motion = MotionRangeConfig{
          .accel = parse_optional_range(node, "accel", path + ".accel"),
          .jerk = parse_optional_range(node, "jerk", path + ".jerk"),
      },
      .reduction = reduction,
  };
}

void parse_input(const YAML::Node& root, const std::filesystem::path& path,
                 Config& config) {
  const YAML::Node input = required_node(root, "input", "input");
  require_map(input, "input");
  reject_unknown_keys(input, {"path"}, "input");
  const std::string input_path = scalar<std::string>(
      required_node(input, "path", "input.path"), "input.path");
  config.input = resolve_path(path, input_path);
}

void parse_dm_ranges(const YAML::Node& root, Config& config) {
  const YAML::Node ranges =
      required_node(root, "dm_ranges", "dm_ranges");
  require_sequence(ranges, "dm_ranges");
  config.dm_ranges.clear();
  config.dm_ranges.reserve(ranges.size());
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    config.dm_ranges.push_back(parse_dm_range(ranges[index], index));
  }
}

void parse_dedispersion(const YAML::Node& root, Config& config) {
  const YAML::Node node = root["dedispersion"];
  if (!node) {
    return;
  }
  require_map(node, "dedispersion");
  reject_unknown_keys(node,
                      {"backend", "device", "subband_channels",
                       "ndm_per_nominal"},
                      "dedispersion");
  if (node["backend"]) {
    const std::string path = "dedispersion.backend";
    config.dedispersion_backend = parse_dedispersion_backend_value(
        scalar<std::string>(node["backend"], path), path);
  }
  if (node["device"]) {
    const long long device =
        scalar<long long>(node["device"], "dedispersion.device");
    if (device < 0 || device > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(
          "dedispersion.device must be a non-negative int");
    }
    config.dedispersion_device = static_cast<int>(device);
  }
  if (node["subband_channels"]) {
    config.subband_channels = nonnegative_size(
        node["subband_channels"], "dedispersion.subband_channels");
  }
  if (node["ndm_per_nominal"]) {
    config.ndm_per_nominal = nonnegative_size(
        node["ndm_per_nominal"], "dedispersion.ndm_per_nominal");
  }
}

void parse_resources(const YAML::Node& root, Config& config) {
  const YAML::Node resources = root["resources"];
  if (!resources) {
    return;
  }
  require_map(resources, "resources");
  reject_unknown_keys(resources, {"native_cuda"}, "resources");

  const YAML::Node native_cuda = resources["native_cuda"];
  if (!native_cuda) {
    return;
  }
  require_map(native_cuda, "resources.native_cuda");
  reject_unknown_keys(native_cuda, {"max_peak_memory"},
                      "resources.native_cuda");
  if (native_cuda["max_peak_memory"]) {
    config.resources.native_cuda.max_peak_memory_bytes = byte_size(
        native_cuda["max_peak_memory"],
        "resources.native_cuda.max_peak_memory");
  }
}

void parse_search(const YAML::Node& root, Config& config) {
  const YAML::Node search = required_node(root, "search", "search");
  require_map(search, "search");
  reject_unknown_keys(search, {"dm_tile_size", "devices", "ranges"},
                      "search");
  if (search["dm_tile_size"]) {
    config.dm_tile_size =
        nonnegative_size(search["dm_tile_size"], "search.dm_tile_size");
  }

  const YAML::Node devices = search["devices"];
  if (devices) {
    require_map(devices, "search.devices");
    reject_unknown_keys(devices, {"native_cuda", "loki_cuda"},
                        "search.devices");
    if (devices["native_cuda"]) {
      config.native_cuda_devices = parse_devices(
          devices["native_cuda"], "search.devices.native_cuda");
    }
    if (devices["loki_cuda"]) {
      config.loki_cuda_devices = parse_devices(
          devices["loki_cuda"], "search.devices.loki_cuda");
    }
  }

  const YAML::Node ranges = required_node(search, "ranges", "search.ranges");
  require_sequence(ranges, "search.ranges");
  config.search_ranges.clear();
  config.search_ranges.reserve(ranges.size());
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    config.search_ranges.push_back(parse_search_range(ranges[index], index));
  }
}

void parse_preprocess(const YAML::Node& root, Config& config) {
  const YAML::Node node = root["preprocess"];
  if (!node) {
    return;
  }
  require_map(node, "preprocess");
  reject_unknown_keys(node, {"kind", "running_median_seconds"},
                      "preprocess");
  config.preprocess = optional_scalar<std::string>(
      node, "kind", config.preprocess, "preprocess.kind");
  config.running_median_seconds = optional_scalar<double>(
      node, "running_median_seconds", config.running_median_seconds,
      "preprocess.running_median_seconds");
}

void parse_candidate(const YAML::Node& root, Config& config) {
  const YAML::Node node = root["candidate"];
  if (!node) {
    return;
  }
  require_map(node, "candidate");
  reject_unknown_keys(node,
                      {"detection", "grouping", "clustering", "harmonic",
                       "selection"},
                      "candidate");

  const YAML::Node detection = node["detection"];
  if (detection) {
    require_map(detection, "candidate.detection");
    reject_unknown_keys(detection,
                        {"snr_threshold", "max_peaks_per_dm",
                         "max_total_raw_peaks"},
                        "candidate.detection");
    config.candidate.detection.snr_threshold = optional_scalar<float>(
        detection, "snr_threshold", config.candidate.detection.snr_threshold,
        "candidate.detection.snr_threshold");
    if (detection["max_peaks_per_dm"]) {
      config.candidate.detection.max_peaks_per_dm = nonnegative_size(
          detection["max_peaks_per_dm"],
          "candidate.detection.max_peaks_per_dm");
    }
    if (detection["max_total_raw_peaks"]) {
      config.candidate.detection.max_total_raw_peaks = nonnegative_size(
          detection["max_total_raw_peaks"],
          "candidate.detection.max_total_raw_peaks");
    }
  }

  const YAML::Node grouping = node["grouping"];
  if (grouping) {
    require_map(grouping, "candidate.grouping");
    reject_unknown_keys(grouping, {"max_phase_distance_cycles", "merge_widths"},
                        "candidate.grouping");
    config.candidate.grouping.max_phase_distance_cycles = optional_scalar<double>(
        grouping, "max_phase_distance_cycles",
        config.candidate.grouping.max_phase_distance_cycles,
        "candidate.grouping.max_phase_distance_cycles");
    config.candidate.grouping.merge_widths = optional_scalar<bool>(
        grouping, "merge_widths", config.candidate.grouping.merge_widths,
        "candidate.grouping.merge_widths");
  }

  const YAML::Node clustering = node["clustering"];
  if (clustering) {
    require_map(clustering, "candidate.clustering");
    reject_unknown_keys(clustering,
                        {"max_phase_distance_cycles", "max_dm_distance",
                         "merge_widths"},
                        "candidate.clustering");
    config.candidate.clustering.max_phase_distance_cycles = optional_scalar<double>(
        clustering, "max_phase_distance_cycles",
        config.candidate.clustering.max_phase_distance_cycles,
        "candidate.clustering.max_phase_distance_cycles");
    config.candidate.clustering.max_dm_distance = optional_scalar<double>(
        clustering, "max_dm_distance",
        config.candidate.clustering.max_dm_distance,
        "candidate.clustering.max_dm_distance");
    config.candidate.clustering.cluster_across_widths = optional_scalar<bool>(
        clustering, "merge_widths",
        config.candidate.clustering.cluster_across_widths,
        "candidate.clustering.merge_widths");
  }

  const YAML::Node harmonic = node["harmonic"];
  if (harmonic) {
    require_map(harmonic, "candidate.harmonic");
    reject_unknown_keys(harmonic,
                        {"max_harmonic", "denominator_max",
                         "frequency_tolerance_bins", "phase_distance_max",
                         "dm_distance_max", "use_snr_consistency",
                         "snr_distance_max"},
                        "candidate.harmonic");
    config.candidate.harmonic.max_harmonic = optional_scalar<std::size_t>(
        harmonic, "max_harmonic", config.candidate.harmonic.max_harmonic,
        "candidate.harmonic.max_harmonic");
    config.candidate.harmonic.denominator_max = optional_scalar<std::size_t>(
        harmonic, "denominator_max",
        config.candidate.harmonic.denominator_max,
        "candidate.harmonic.denominator_max");
    config.candidate.harmonic.frequency_tolerance_bins =
        optional_nullable_scalar<double>(
            harmonic, "frequency_tolerance_bins",
            "candidate.harmonic.frequency_tolerance_bins");
    config.candidate.harmonic.phase_distance_max = optional_scalar<double>(
        harmonic, "phase_distance_max",
        config.candidate.harmonic.phase_distance_max,
        "candidate.harmonic.phase_distance_max");
    config.candidate.harmonic.dm_distance_max = optional_scalar<double>(
        harmonic, "dm_distance_max", config.candidate.harmonic.dm_distance_max,
        "candidate.harmonic.dm_distance_max");
    config.candidate.harmonic.use_snr_consistency = optional_scalar<bool>(
        harmonic, "use_snr_consistency",
        config.candidate.harmonic.use_snr_consistency,
        "candidate.harmonic.use_snr_consistency");
    config.candidate.harmonic.snr_distance_max = optional_scalar<double>(
        harmonic, "snr_distance_max",
        config.candidate.harmonic.snr_distance_max,
        "candidate.harmonic.snr_distance_max");
  }

  const YAML::Node selection = node["selection"];
  if (selection) {
    require_map(selection, "candidate.selection");
    reject_unknown_keys(selection, {"snr_min", "max_candidates"},
                        "candidate.selection");
    config.candidate.selection.snr_min = optional_scalar<float>(
        selection, "snr_min", config.candidate.selection.snr_min,
        "candidate.selection.snr_min");
    config.candidate.selection.max_candidates = optional_scalar<std::size_t>(
        selection, "max_candidates", config.candidate.selection.max_candidates,
        "candidate.selection.max_candidates");
  }
}

void parse_output(const YAML::Node& root, const std::filesystem::path& path,
                  Config& config) {
  const YAML::Node node = root["output"];
  if (!node) {
    return;
  }
  require_map(node, "output");
  reject_unknown_keys(node, {"prefix", "print_candidates", "overwrite"},
                      "output");
  if (node["prefix"]) {
    const std::string prefix =
        scalar<std::string>(node["prefix"], "output.prefix");
    config.candidate_output = resolve_path(path, prefix);
  }
  if (node["print_candidates"]) {
    config.print_candidates =
        nonnegative_size(node["print_candidates"], "output.print_candidates");
  }
  config.overwrite_output = optional_scalar<bool>(
      node, "overwrite", config.overwrite_output, "output.overwrite");
}

Config parse_yaml(const YAML::Node& root, const std::filesystem::path& path) {
  require_map(root, "root");
  reject_unknown_keys(root,
                      {"version", "input", "dm_ranges", "dedispersion",
                       "search", "preprocess", "candidate", "resources",
                       "output"},
                      "root");
  const int version = scalar<int>(required_node(root, "version", "version"),
                                  "version");
  if (version != 1) {
    throw std::invalid_argument("unsupported YAML config version: " +
                                std::to_string(version));
  }

  Config config;
  parse_input(root, path, config);
  parse_dm_ranges(root, config);
  parse_dedispersion(root, config);
  parse_resources(root, config);
  parse_search(root, config);
  parse_preprocess(root, config);
  parse_candidate(root, config);
  parse_output(root, path, config);
  validate_config(config);
  return config;
}

}  // namespace

Config load_yaml_config(const std::filesystem::path& path) {
  try {
    return parse_yaml(YAML::LoadFile(path.string()), path);
  } catch (const YAML::Exception& error) {
    throw std::invalid_argument("invalid YAML config '" + path.string() +
                                "': " + error.what());
  }
}

}  // namespace gaffa_search
