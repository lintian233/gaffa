#include "config_yaml.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <limits>
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

std::size_t nonnegative_size(const YAML::Node& node, std::string_view path) {
  const long long value = scalar<long long>(node, path);
  if (value < 0) {
    throw std::invalid_argument(std::string(path) +
                                " must be a non-negative integer");
  }
  return static_cast<std::size_t>(value);
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
                       "window"},
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
      .window_mode = window
                          ? parse_window_value(
                                scalar<std::string>(window, path + ".window"),
                                path + ".window")
                          : WindowMode::Truncate,
      .motion = MotionRangeConfig{
          .accel = parse_optional_range(node, "accel", path + ".accel"),
          .jerk = parse_optional_range(node, "jerk", path + ".jerk"),
      },
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
                      {"snr_threshold", "max_peaks", "max_total_raw_peaks",
                       "max_candidates", "dm_radius"},
                      "candidate");
  config.snr_threshold = optional_scalar<float>(
      node, "snr_threshold", config.snr_threshold,
      "candidate.snr_threshold");
  if (node["max_peaks"]) {
    config.max_peaks =
        nonnegative_size(node["max_peaks"], "candidate.max_peaks");
  }
  if (node["max_total_raw_peaks"]) {
    config.max_total_raw_peaks = nonnegative_size(
        node["max_total_raw_peaks"], "candidate.max_total_raw_peaks");
  }
  if (node["max_candidates"]) {
    config.max_candidates =
        nonnegative_size(node["max_candidates"], "candidate.max_candidates");
  }
  if (node["dm_radius"]) {
    config.candidate_dm_radius =
        scalar<double>(node["dm_radius"], "candidate.dm_radius");
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
                       "search", "preprocess", "candidate", "output"},
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
