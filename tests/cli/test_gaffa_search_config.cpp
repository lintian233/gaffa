#include "config.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

gaffa_search::Config parse(std::vector<std::string> arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (auto& argument : arguments) {
    argv.push_back(argument.data());
  }
  return gaffa_search::parse_arguments(static_cast<int>(argv.size()),
                                       argv.data());
}

class TemporaryYaml {
 public:
  explicit TemporaryYaml(std::string contents) {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("gaffa-search-config-" + std::to_string(stamp) + ".yaml");
    std::ofstream output(path_);
    if (!output.good()) {
      throw std::runtime_error("failed to create temporary YAML file");
    }
    output << contents;
    if (!output.good()) {
      throw std::runtime_error("failed to write temporary YAML file");
    }
  }

  ~TemporaryYaml() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace

TEST(GaffaSearchConfig, ParsesMultipleDmAndSearchRanges) {
  const auto config = parse({
      "gaffa_search", "--input", "observation.fil", "--dm-range",
      "0:0.5:4", "--dm-range", "2:0.5:4", "--search",
      "native-cpu:0.018:1:180:256", "--search",
      "native-cuda:0.002:0.02:32:128:truncate", "--native-devices",
      "0,1",
  });

  ASSERT_EQ(config.dm_ranges.size(), 2U);
  ASSERT_EQ(config.search_ranges.size(), 2U);
  EXPECT_EQ(config.dm_ranges[1].ndm, 4U);
  EXPECT_EQ(config.search_ranges[1].backend,
            gaffa_search::Backend::NativeCuda);
  ASSERT_EQ(config.native_cuda_devices.size(), 2U);
  EXPECT_EQ(config.native_cuda_devices[0], 0);
  EXPECT_EQ(config.native_cuda_devices[1], 1);
}

TEST(GaffaSearchConfig, RejectsOverlappingDmRanges) {
  EXPECT_THROW(
      parse({"gaffa_search", "--input", "observation.fil", "--dm-range",
             "0:1:4", "--dm-range", "2:1:4", "--search",
             "native-cpu:0.018:1:180:256"}),
      std::invalid_argument);
}

TEST(GaffaSearchConfig, RejectsCudaBackendWithoutDevices) {
  EXPECT_THROW(
      parse({"gaffa_search", "--input", "observation.fil", "--dm-range",
             "0:1:4", "--search", "native-cuda:0.018:1:180:256"}),
      std::invalid_argument);
}

TEST(GaffaSearchConfig, NativeMeansNativeCudaAndRequiresDevices) {
  EXPECT_THROW(
      parse({"gaffa_search", "--input", "observation.fil", "--dm-range",
             "0:1:4", "--search", "native:0.018:1:180:256"}),
      std::invalid_argument);

  const auto config = parse({
      "gaffa_search", "--input", "observation.fil", "--dm-range", "0:1:4",
      "--search", "native:0.018:1:180:256", "--native-devices", "0",
  });
  ASSERT_EQ(config.search_ranges.size(), 1U);
  EXPECT_EQ(config.search_ranges.front().backend,
            gaffa_search::Backend::NativeCuda);
}

TEST(GaffaSearchConfig, ParsesCliMotionRangesAfterSearchDefinitions) {
  const auto config = parse({
      "gaffa_search", "--input", "observation.fil", "--dm-range",
      "0:1:4", "--search-accel", "1:-5:5", "--search-jerk",
      "1:-0.2:0.2", "--search", "native-cuda:0.018:1:180:256",
      "--search", "loki-cuda:0.05:1:128:256:zero-pad", "--native-devices",
      "0", "--loki-devices", "1",
  });

  ASSERT_EQ(config.search_ranges.size(), 2U);
  EXPECT_FALSE(config.search_ranges[0].motion.accel.has_value());
  ASSERT_TRUE(config.search_ranges[1].motion.accel.has_value());
  EXPECT_DOUBLE_EQ(config.search_ranges[1].motion.accel->minimum, -5.0);
  EXPECT_DOUBLE_EQ(config.search_ranges[1].motion.accel->maximum, 5.0);
  ASSERT_TRUE(config.search_ranges[1].motion.jerk.has_value());
  EXPECT_DOUBLE_EQ(config.search_ranges[1].motion.jerk->minimum, -0.2);
  EXPECT_DOUBLE_EQ(config.search_ranges[1].motion.jerk->maximum, 0.2);
}

TEST(GaffaSearchConfig, RejectsInvalidCliMotionBindings) {
  EXPECT_THROW(
      parse({"gaffa_search", "--input", "observation.fil", "--dm-range",
             "0:1:4", "--search", "loki-cuda:0.05:1:128:256",
             "--loki-devices", "0", "--search-accel", "1:-5:5"}),
      std::invalid_argument);

  EXPECT_THROW(
      parse({"gaffa_search", "--input", "observation.fil", "--dm-range",
             "0:1:4", "--search", "loki-cuda:0.05:1:128:256",
             "--loki-devices", "0", "--search-jerk", "0:-1:1"}),
      std::invalid_argument);

  EXPECT_THROW(
      parse({"gaffa_search", "--input", "observation.fil", "--dm-range",
             "0:1:4", "--search", "loki-cuda:0.05:1:128:256",
             "--loki-devices", "0", "--search-accel", "0:0:0"}),
      std::invalid_argument);

  EXPECT_THROW(
      parse({"gaffa_search", "--input", "observation.fil", "--dm-range",
             "0:1:4", "--search", "native-cpu:0.05:1:128:256",
             "--search-accel", "0:-5:5"}),
      std::invalid_argument);
}

TEST(GaffaSearchConfig, RejectsNegativeDm) {
  EXPECT_THROW(
      parse({"gaffa_search", "--input", "observation.fil", "--dm-range",
             "-1:1:4", "--search", "native-cpu:0.018:1:180:256"}),
      std::invalid_argument);
}

TEST(GaffaSearchConfig, RejectsDifferentDmSteps) {
  EXPECT_THROW(
      parse({"gaffa_search", "--input", "observation.fil", "--dm-range",
             "0:1:4", "--dm-range", "4:0.5:4", "--search",
             "native-cpu:0.018:1:180:256"}),
      std::invalid_argument);
}

TEST(GaffaSearchConfig, ParsesOutputControls) {
  const auto config = parse({
      "gaffa_search", "--input", "observation.fil", "--dm-range", "0:1:4",
      "--search", "native-cpu:0.018:1:180:256", "--print-candidates", "12",
      "--cand", "results/candidates", "--overwrite",
  });

  EXPECT_EQ(config.print_candidates, 12U);
  ASSERT_TRUE(config.candidate_output.has_value());
  EXPECT_EQ(*config.candidate_output, "results/candidates");
  EXPECT_TRUE(config.overwrite_output);
}

TEST(GaffaSearchConfig, RejectsOverwriteWithoutCandidateOutput) {
  EXPECT_THROW(
      parse({"gaffa_search", "--input", "observation.fil", "--dm-range",
             "0:1:4", "--search", "native-cpu:0.018:1:180:256",
             "--overwrite"}),
      std::invalid_argument);
}

TEST(GaffaSearchConfig, ParsesYamlAndResolvesRelativePaths) {
  TemporaryYaml yaml(R"(
version: 1
input:
  path: observation.fil
dm_ranges:
  - low: 0.0
    step: 0.5
    count: 4
  - low: 2.0
    step: 0.5
    count: 4
dedispersion:
  backend: cuda-subband
  device: 1
  subband_channels: 16
  ndm_per_nominal: 8
search:
  dm_tile_size: 4
  devices:
    native_cuda: [0, 1]
    loki_cuda: [2]
  ranges:
    - backend: native-cpu
      period: {min: 0.018, max: 1.0}
      bins: {min: 180, max: 256}
    - backend: loki-cuda
      period: {min: 0.018, max: 1.0}
      bins: {min: 180, max: 256}
      accel: {min: -5.0, max: 5.0}
      jerk: {min: -0.2, max: 0.2}
      window: zero-pad
preprocess:
  kind: normalise
  running_median_seconds: 3.0
candidate:
  snr_threshold: 8.0
  max_peaks: 12
  max_total_raw_peaks: 100
  max_candidates: 4
  dm_index_radius: 10
output:
  prefix: results/candidates
  print_candidates: 7
  overwrite: true
)");

  const auto config = parse(
      {"gaffa_search", "--config", yaml.path().string()});

  EXPECT_EQ(config.input, yaml.path().parent_path() / "observation.fil");
  ASSERT_EQ(config.dm_ranges.size(), 2U);
  EXPECT_EQ(config.dm_ranges[1].dm_low, 2.0);
  EXPECT_EQ(config.dedispersion_device, 1);
  EXPECT_EQ(config.subband_channels, 16U);
  EXPECT_EQ(config.ndm_per_nominal, 8U);
  EXPECT_EQ(config.dm_tile_size, 4U);
  ASSERT_EQ(config.native_cuda_devices, (std::vector<int>{0, 1}));
  ASSERT_EQ(config.loki_cuda_devices, (std::vector<int>{2}));
  ASSERT_EQ(config.search_ranges.size(), 2U);
  EXPECT_EQ(config.search_ranges[0].backend,
            gaffa_search::Backend::NativeCpu);
  EXPECT_EQ(config.search_ranges[1].backend,
            gaffa_search::Backend::LokiCuda);
  EXPECT_EQ(config.search_ranges[1].window_mode,
            gaffa_search::WindowMode::ZeroPad);
  ASSERT_TRUE(config.search_ranges[1].motion.accel.has_value());
  EXPECT_DOUBLE_EQ(config.search_ranges[1].motion.accel->minimum, -5.0);
  EXPECT_DOUBLE_EQ(config.search_ranges[1].motion.accel->maximum, 5.0);
  ASSERT_TRUE(config.search_ranges[1].motion.jerk.has_value());
  EXPECT_DOUBLE_EQ(config.search_ranges[1].motion.jerk->minimum, -0.2);
  EXPECT_DOUBLE_EQ(config.search_ranges[1].motion.jerk->maximum, 0.2);
  EXPECT_FLOAT_EQ(config.snr_threshold, 8.0F);
  EXPECT_EQ(config.max_peaks, 12U);
  EXPECT_EQ(config.max_total_raw_peaks, 100U);
  EXPECT_EQ(config.max_candidates, 4U);
  EXPECT_EQ(config.candidate_dm_index_radius, 10U);
  EXPECT_EQ(config.print_candidates, 7U);
  ASSERT_TRUE(config.candidate_output.has_value());
  EXPECT_EQ(*config.candidate_output,
            yaml.path().parent_path() / "results/candidates");
  EXPECT_TRUE(config.overwrite_output);
}

TEST(GaffaSearchConfig, RejectsConfigCombinedWithLegacyOptions) {
  EXPECT_THROW(
      parse({"gaffa_search", "--config", "missing.yaml", "--help"}),
      std::invalid_argument);
  EXPECT_THROW(parse({"gaffa_search", "--config"}), std::invalid_argument);
}

TEST(GaffaSearchConfig, RejectsUnknownYamlKeysAndInvalidWindow) {
  TemporaryYaml unknown(R"(
version: 1
input: {path: observation.fil}
dm_ranges: [{low: 0.0, step: 1.0, count: 2}]
search:
  ranges:
    - backend: native-cpu
      period: {min: 0.1, max: 1.0}
      bins: {min: 16, max: 32}
      unexpected: true
)");
  EXPECT_THROW(parse({"gaffa_search", "--config", unknown.path().string()}),
               std::invalid_argument);

  TemporaryYaml native_window(R"(
version: 1
input: {path: observation.fil}
dm_ranges: [{low: 0.0, step: 1.0, count: 2}]
search:
  ranges:
    - backend: native-cpu
      period: {min: 0.1, max: 1.0}
      bins: {min: 16, max: 32}
      window: truncate
)");
  EXPECT_THROW(
      parse({"gaffa_search", "--config", native_window.path().string()}),
      std::invalid_argument);
}

TEST(GaffaSearchConfig, RejectsNegativeYamlSizesAndDevices) {
  TemporaryYaml negative_count(R"(
version: 1
input: {path: observation.fil}
dm_ranges: [{low: 0.0, step: 1.0, count: -1}]
search:
  ranges:
    - backend: native-cpu
      period: {min: 0.1, max: 1.0}
      bins: {min: 16, max: 32}
)");
  EXPECT_THROW(
      parse({"gaffa_search", "--config", negative_count.path().string()}),
      std::invalid_argument);

  TemporaryYaml negative_device(R"(
version: 1
input: {path: observation.fil}
dm_ranges: [{low: 0.0, step: 1.0, count: 2}]
dedispersion: {device: -1}
search:
  ranges:
    - backend: native-cpu
      period: {min: 0.1, max: 1.0}
      bins: {min: 16, max: 32}
)");
  EXPECT_THROW(
      parse({"gaffa_search", "--config", negative_device.path().string()}),
      std::invalid_argument);
}

TEST(GaffaSearchConfig, RejectsInvalidYamlMotionRanges) {
  TemporaryYaml jerk_without_accel(R"(
version: 1
input: {path: observation.fil}
dm_ranges: [{low: 0.0, step: 1.0, count: 2}]
search:
  devices: {loki_cuda: [0]}
  ranges:
    - backend: loki-cuda
      period: {min: 0.1, max: 1.0}
      bins: {min: 16, max: 32}
      jerk: {min: -1.0, max: 1.0}
)");
  EXPECT_THROW(
      parse({"gaffa_search", "--config", jerk_without_accel.path().string()}),
      std::invalid_argument);

  TemporaryYaml native_motion(R"(
version: 1
input: {path: observation.fil}
dm_ranges: [{low: 0.0, step: 1.0, count: 2}]
search:
  ranges:
    - backend: native-cpu
      period: {min: 0.1, max: 1.0}
      bins: {min: 16, max: 32}
      accel: {min: -1.0, max: 1.0}
)");
  EXPECT_THROW(
      parse({"gaffa_search", "--config", native_motion.path().string()}),
      std::invalid_argument);

  TemporaryYaml snap(R"(
version: 1
input: {path: observation.fil}
dm_ranges: [{low: 0.0, step: 1.0, count: 2}]
search:
  ranges:
    - backend: loki-cuda
      period: {min: 0.1, max: 1.0}
      bins: {min: 16, max: 32}
      snap: {min: -1.0, max: 1.0}
)");
  EXPECT_THROW(parse({"gaffa_search", "--config", snap.path().string()}),
               std::invalid_argument);
}
