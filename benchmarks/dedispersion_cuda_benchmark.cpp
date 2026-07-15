#include "gaffa/dedispersion.h"
#include "gaffa/dedispersion_cuda.h"
#include "gaffa/filterbank.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/vector_add.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

enum class Mode {
  Single,
  Multi,
  Subband,
};

struct Args {
  std::string mode;
  std::filesystem::path path;
  int iterations = 5;
  std::size_t ndm = 256;
  double dm_step = 1.0;
  std::size_t subband_channels = 32;
  std::size_t ndm_per_nominal = 32;
  std::size_t time_tile_samples = 81920;
  bool checksum_enabled = false;
};

struct IterationResult {
  int iteration = 0;
  double seconds = 0.0;
  std::size_t output_bytes = 0;
  std::optional<double> checksum;
};

struct BenchmarkResult {
  Mode mode = Mode::Single;
  std::vector<IterationResult> runs;
};

struct SummaryStats {
  double mean_seconds = 0.0;
  double median_seconds = 0.0;
  double min_seconds = 0.0;
  double max_seconds = 0.0;
  double stddev_seconds = 0.0;
};

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <single|multi|subband|all> <filterbank.fil> [iterations]"
            << " [ndm] [dm_step] [subband_channels] [ndm_per_nominal]"
            << " [time_tile_samples] [checksum]\n";
}

Args parse_args(int argc, char** argv) {
  if (argc < 3 || argc > 10) {
    usage(argv[0]);
    throw std::invalid_argument("invalid argument count");
  }

  Args args;
  args.mode = argv[1];
  args.path = argv[2];
  if (argc >= 4) {
    args.iterations = std::stoi(argv[3]);
  }
  if (argc >= 5) {
    args.ndm = static_cast<std::size_t>(std::stoull(argv[4]));
  }
  if (argc >= 6) {
    args.dm_step = std::stod(argv[5]);
  }
  if (argc >= 7) {
    args.subband_channels = static_cast<std::size_t>(std::stoull(argv[6]));
  }
  if (argc >= 8) {
    args.ndm_per_nominal = static_cast<std::size_t>(std::stoull(argv[7]));
  }
  if (argc >= 9) {
    args.time_tile_samples = static_cast<std::size_t>(std::stoull(argv[8]));
  }
  if (argc >= 10) {
    const std::string checksum_arg = argv[9];
    if (checksum_arg != "checksum" && checksum_arg != "no-checksum") {
      usage(argv[0]);
      throw std::invalid_argument("checksum option must be checksum or no-checksum");
    }
    args.checksum_enabled = checksum_arg == "checksum";
  }

  if (args.mode != "single" && args.mode != "multi" &&
      args.mode != "subband" && args.mode != "all") {
    usage(argv[0]);
    throw std::invalid_argument("unknown benchmark mode");
  }
  if (args.iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  if (args.ndm == 0) {
    throw std::invalid_argument("ndm must be positive");
  }
  if (!(args.dm_step > 0.0)) {
    throw std::invalid_argument("dm_step must be positive");
  }
  if (args.subband_channels == 0) {
    throw std::invalid_argument("subband_channels must be positive");
  }
  if (args.ndm_per_nominal == 0) {
    throw std::invalid_argument("ndm_per_nominal must be positive");
  }
  if (args.time_tile_samples == 0) {
    throw std::invalid_argument("time_tile_samples must be positive");
  }
  return args;
}

std::string_view mode_name(Mode mode) {
  switch (mode) {
    case Mode::Single:
      return "single";
    case Mode::Multi:
      return "multi";
    case Mode::Subband:
      return "subband";
  }
  return "unknown";
}

std::vector<Mode> selected_modes(const std::string& mode) {
  if (mode == "single") {
    return {Mode::Single};
  }
  if (mode == "multi") {
    return {Mode::Multi};
  }
  if (mode == "subband") {
    return {Mode::Subband};
  }
  return {Mode::Single, Mode::Multi, Mode::Subband};
}

template <typename Func>
double time_once(Func&& func) {
  const auto start = std::chrono::steady_clock::now();
  func();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

std::string_view dtype_name(const gaffa::FilterbankSamples& samples) {
  return std::visit(
      [](const auto& values) -> std::string_view {
        using T = typename std::decay_t<decltype(values)>::value_type;
        if constexpr (std::is_same_v<T, std::uint8_t>) {
          return "uint8";
        } else if constexpr (std::is_same_v<T, std::uint16_t>) {
          return "uint16";
        } else {
          return "float32";
        }
      },
      samples);
}

template <typename T>
double checksum(const std::vector<T>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0,
                         [](double sum, T value) {
                           return sum + static_cast<double>(value);
                         });
}

template <typename T>
IterationResult make_iteration_result(int iteration, double seconds,
                                      const gaffa::DedispersedResult<T>& result,
                                      bool checksum_enabled) {
  IterationResult iteration_result{
      .iteration = iteration,
      .seconds = seconds,
      .output_bytes = result.data.size() * sizeof(T),
  };
  if (checksum_enabled) {
    iteration_result.checksum = checksum(result.data);
  }
  return iteration_result;
}

gaffa::SingleDmDedispersionPlan single_plan(
    const gaffa::FilterbankHeader& header) {
  return gaffa::SingleDmDedispersionPlan{
      .dm = 0.0,
      .ref_frequency_mhz = header.frequency_table.back(),
      .tsamp = header.tsamp,
      .chan_begin = 0,
      .chan_end = static_cast<std::size_t>(header.nchans),
  };
}

gaffa::MultiDmDedispersionPlan multi_plan(
    const gaffa::FilterbankHeader& header, const Args& args) {
  return gaffa::MultiDmDedispersionPlan{
      .dm_low = 0.0,
      .dm_step = args.dm_step,
      .ndm = args.ndm,
      .ref_frequency_mhz = header.frequency_table.back(),
      .tsamp = header.tsamp,
      .chan_begin = 0,
      .chan_end = static_cast<std::size_t>(header.nchans),
  };
}

gaffa::SubbandDedispersionOptions subband_options(const Args& args) {
  return gaffa::SubbandDedispersionOptions{
      .subband_channels = args.subband_channels,
      .ndm_per_nominal = args.ndm_per_nominal,
  };
}

gaffa::CudaDedispersionOptions cuda_options(const Args& args) {
  return gaffa::CudaDedispersionOptions{
      .device_id = 0,
      .threads_per_block = 256,
      .time_tile_samples = args.time_tile_samples,
  };
}

template <typename T>
IterationResult run_typed_mode(const gaffa::FilterbankData& filterbank,
                               const Args& args, Mode mode, int iteration) {
  using OutT = std::conditional_t<std::is_same_v<T, float>, float, std::uint32_t>;
  const auto view = gaffa::sample_view<T>(filterbank);
  IterationResult result;
  if (mode == Mode::Single) {
    gaffa::DedispersedResult<OutT> dedispersed;
    const double seconds = time_once([&] {
      dedispersed = gaffa::dedisperse_single_dm_cuda(
          view, filterbank.header.frequency_table, single_plan(filterbank.header),
          cuda_options(args));
    });
    result = make_iteration_result(iteration, seconds, dedispersed,
                                   args.checksum_enabled);
  } else if (mode == Mode::Multi) {
    gaffa::DedispersedResult<OutT> dedispersed;
    const double seconds = time_once([&] {
      dedispersed = gaffa::dedisperse_multi_dm_cuda(
          view, filterbank.header.frequency_table, multi_plan(filterbank.header, args),
          cuda_options(args));
    });
    result = make_iteration_result(iteration, seconds, dedispersed,
                                   args.checksum_enabled);
  } else {
    gaffa::DedispersedResult<OutT> dedispersed;
    const double seconds = time_once([&] {
      dedispersed = gaffa::dedisperse_subband_cuda(
          view, filterbank.header.frequency_table, multi_plan(filterbank.header, args),
          subband_options(args), cuda_options(args));
    });
    result = make_iteration_result(iteration, seconds, dedispersed,
                                   args.checksum_enabled);
  }
  return result;
}

IterationResult run_mode_once(const gaffa::FilterbankData& filterbank,
                              const Args& args, Mode mode, int iteration) {
  return std::visit(
      [&](const auto& samples) {
        using T = typename std::decay_t<decltype(samples)>::value_type;
        return run_typed_mode<T>(filterbank, args, mode, iteration);
      },
      filterbank.samples);
}

BenchmarkResult run_mode(const gaffa::FilterbankData& filterbank,
                         const Args& args, Mode mode) {
  BenchmarkResult result;
  result.mode = mode;
  result.runs.reserve(static_cast<std::size_t>(args.iterations));
  for (int iteration = 0; iteration < args.iterations; ++iteration) {
    IterationResult run = run_mode_once(filterbank, args, mode, iteration);
    std::cout << "mode=" << mode_name(mode)
              << " iteration=" << iteration
              << " seconds=" << run.seconds
              << " output_bytes=" << run.output_bytes;
    if (run.checksum.has_value()) {
      std::cout << " checksum=" << *run.checksum;
    } else {
      std::cout << " checksum=disabled";
    }
    std::cout << '\n';
    result.runs.push_back(run);
  }
  return result;
}

SummaryStats summarize(const BenchmarkResult& result) {
  SummaryStats stats;
  stats.min_seconds = result.runs.front().seconds;
  stats.max_seconds = result.runs.front().seconds;
  std::vector<double> seconds;
  seconds.reserve(result.runs.size());
  for (const IterationResult& run : result.runs) {
    seconds.push_back(run.seconds);
    stats.mean_seconds += run.seconds;
    stats.min_seconds = std::min(stats.min_seconds, run.seconds);
    stats.max_seconds = std::max(stats.max_seconds, run.seconds);
  }
  stats.mean_seconds /= static_cast<double>(result.runs.size());
  std::sort(seconds.begin(), seconds.end());
  const std::size_t middle = seconds.size() / 2;
  stats.median_seconds = seconds.size() % 2 == 0
                             ? (seconds[middle - 1] + seconds[middle]) / 2.0
                             : seconds[middle];
  double variance = 0.0;
  for (const IterationResult& run : result.runs) {
    const double delta = run.seconds - stats.mean_seconds;
    variance += delta * delta;
  }
  stats.stddev_seconds =
      std::sqrt(variance / static_cast<double>(result.runs.size()));
  return stats;
}

std::size_t sample_bytes(const gaffa::FilterbankData& filterbank) {
  return std::visit(
      [](const auto& samples) { return samples.size() * sizeof(samples[0]); },
      filterbank.samples);
}

void print_report(const gaffa::FilterbankData& filterbank, const Args& args,
                  double read_seconds,
                  const std::vector<BenchmarkResult>& results) {
  std::cout << "final_report_begin"
            << " modes=" << results.size()
            << " iterations=" << args.iterations
            << " read_seconds=" << read_seconds
            << " dtype=" << dtype_name(filterbank.samples)
            << " nsamples=" << filterbank.header.nsamples
            << " nifs=" << filterbank.header.nifs
            << " nchans=" << filterbank.header.nchans
            << " nbits=" << filterbank.header.nbits
            << " input_bytes=" << sample_bytes(filterbank)
            << " ndm=" << args.ndm
            << " dm_step=" << args.dm_step
            << " subband_channels=" << args.subband_channels
            << " ndm_per_nominal=" << args.ndm_per_nominal
            << " time_tile_samples=" << args.time_tile_samples
            << " checksum=" << (args.checksum_enabled ? "enabled" : "disabled")
            << '\n';

  for (const BenchmarkResult& result : results) {
    const SummaryStats stats = summarize(result);
    std::cout << "final_report"
              << " mode=" << mode_name(result.mode)
              << " mean_seconds=" << stats.mean_seconds
              << " median_seconds=" << stats.median_seconds
              << " min_seconds=" << stats.min_seconds
              << " max_seconds=" << stats.max_seconds
              << " stddev_seconds=" << stats.stddev_seconds
              << " output_bytes=" << result.runs.back().output_bytes;
    if (result.runs.back().checksum.has_value()) {
      std::cout << " checksum=" << *result.runs.back().checksum;
    } else {
      std::cout << " checksum=disabled";
    }
    std::cout << '\n';
  }
  std::cout << "final_report_end\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    const Args args = parse_args(argc, argv);
    if (gaffa::cuda_device_count() == 0) {
      throw std::runtime_error("CUDA device is not visible");
    }

    gaffa::FilterbankData filterbank;
    const double read_seconds =
        time_once([&] { filterbank = gaffa::read_filterbank(args.path); });
    if (filterbank.header.nifs != 1) {
      throw std::runtime_error("dedispersion benchmark requires nifs == 1");
    }

    std::vector<BenchmarkResult> results;
    const std::vector<Mode> modes = selected_modes(args.mode);
    results.reserve(modes.size());
    for (Mode mode : modes) {
      results.push_back(run_mode(filterbank, args, mode));
    }
    print_report(filterbank, args, read_seconds, results);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
