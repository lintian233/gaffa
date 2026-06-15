#include "gaffa/filterbank.h"
#include "gaffa/filterbank_legacy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

enum class Mode {
  Legacy,
  Eager,
  EagerPreserve,
};

struct Args {
  std::string mode;
  std::filesystem::path path;
  int iterations = 1;
};

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <legacy|eager|eager-preserve|all> <filterbank.fil> [iterations]\n";
}

Args parse_args(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    usage(argv[0]);
    throw std::invalid_argument("invalid argument count");
  }

  Args args;
  args.mode = argv[1];
  args.path = argv[2];
  if (argc == 4) {
    args.iterations = std::stoi(argv[3]);
  }
  if (args.iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  if (args.mode != "legacy" && args.mode != "eager" &&
      args.mode != "eager-preserve" && args.mode != "all") {
    usage(argv[0]);
    throw std::invalid_argument("unknown benchmark mode");
  }
  return args;
}

std::string_view mode_name(Mode mode) {
  switch (mode) {
    case Mode::Legacy:
      return "legacy";
    case Mode::Eager:
      return "eager";
    case Mode::EagerPreserve:
      return "eager-preserve";
  }
  return "unknown";
}

std::vector<Mode> selected_modes(const std::string& mode) {
  if (mode == "legacy") {
    return {Mode::Legacy};
  }
  if (mode == "eager") {
    return {Mode::Eager};
  }
  if (mode == "eager-preserve") {
    return {Mode::EagerPreserve};
  }
  return {Mode::Legacy, Mode::Eager, Mode::EagerPreserve};
}

struct Measurement {
  double seconds = 0.0;
  double bytes = 0.0;
  std::int64_t nsamples = 0;
  int nifs = 0;
  int nchans = 0;
  int nbits = 0;
  double foff = 0.0;
  std::size_t samples = 0;
};

struct IterationResult {
  int iteration = 0;
  double seconds = 0.0;
  double mib_per_sec = 0.0;
};

struct BenchmarkResult {
  Mode mode = Mode::Legacy;
  int iterations = 0;
  std::uintmax_t file_size_bytes = 0;
  double bytes = 0.0;
  std::int64_t nsamples = 0;
  int nifs = 0;
  int nchans = 0;
  int nbits = 0;
  double foff = 0.0;
  std::size_t samples = 0;
  std::vector<IterationResult> runs;
};

struct SummaryStats {
  double mean_seconds = 0.0;
  double median_seconds = 0.0;
  double min_seconds = 0.0;
  double max_seconds = 0.0;
  double stddev_seconds = 0.0;
  double mean_mib_per_sec = 0.0;
  double median_mib_per_sec = 0.0;
  double best_mib_per_sec = 0.0;
};

gaffa::FilterbankReadOptions options_for_mode(Mode mode) {
  gaffa::FilterbankReadOptions options;
  options.channel_order =
      mode == Mode::EagerPreserve ? gaffa::ChannelOrderPolicy::PreserveFileOrder
                                  : gaffa::ChannelOrderPolicy::FrequencyAscending;
  return options;
}

std::string_view channel_order_name(gaffa::ChannelOrderPolicy policy) {
  switch (policy) {
    case gaffa::ChannelOrderPolicy::FrequencyAscending:
      return "frequency-ascending";
    case gaffa::ChannelOrderPolicy::PreserveFileOrder:
      return "preserve-file-order";
  }
  return "unknown";
}

std::string_view reverse_backend_name(gaffa::ReverseBackend backend) {
  switch (backend) {
    case gaffa::ReverseBackend::Auto:
      return "auto";
    case gaffa::ReverseBackend::CpuScalar:
      return "cpu-scalar";
    case gaffa::ReverseBackend::CpuOpenmp:
      return "cpu-openmp";
  }
  return "unknown";
}

std::string_view reported_reverse_backend(Mode mode,
                                          const gaffa::FilterbankReadOptions& options) {
  if (mode == Mode::Legacy) {
    return "legacy";
  }
  return reverse_backend_name(options.reverse_backend);
}

int omp_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

std::size_t sample_count(const gaffa::FilterbankHeader& header) {
  return static_cast<std::size_t>(header.nsamples) *
         static_cast<std::size_t>(header.nifs) *
         static_cast<std::size_t>(header.nchans);
}

std::size_t sample_bytes(const gaffa::FilterbankData& data) {
  return std::visit(
      [](const auto& samples) { return samples.size() * sizeof(samples[0]); },
      data.samples);
}

double legacy_sample_bytes(const Filterbank& filterbank) {
  return static_cast<double>(filterbank.nsamples) *
         static_cast<double>(filterbank.nifs) *
         static_cast<double>(filterbank.nchans) *
         (static_cast<double>(filterbank.nbits) / 8.0);
}

template <typename Func>
double time_once(Func&& func) {
  const auto start = std::chrono::steady_clock::now();
  func();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

double mib_per_second(double bytes, double seconds) {
  return bytes / 1024.0 / 1024.0 / seconds;
}

Measurement measure_legacy_once(const std::filesystem::path& path) {
  std::unique_ptr<Filterbank> filterbank;
  const double seconds =
      time_once([&] { filterbank = std::make_unique<Filterbank>(path.string()); });

  Measurement measurement;
  measurement.seconds = seconds;
  measurement.bytes = legacy_sample_bytes(*filterbank);
  measurement.nsamples = filterbank->nsamples;
  measurement.nifs = filterbank->nifs;
  measurement.nchans = filterbank->nchans;
  measurement.nbits = filterbank->nbits;
  measurement.foff = filterbank->foff;
  measurement.samples = static_cast<std::size_t>(filterbank->nsamples) *
                        static_cast<std::size_t>(filterbank->nifs) *
                        static_cast<std::size_t>(filterbank->nchans);
  return measurement;
}

Measurement measure_eager_once(const std::filesystem::path& path, Mode mode) {
  const gaffa::FilterbankReadOptions options = options_for_mode(mode);
  gaffa::FilterbankData data;
  const double seconds = time_once([&] { data = gaffa::read_filterbank(path, options); });

  Measurement measurement;
  measurement.seconds = seconds;
  measurement.bytes = static_cast<double>(sample_bytes(data));
  measurement.nsamples = data.header.nsamples;
  measurement.nifs = data.header.nifs;
  measurement.nchans = data.header.nchans;
  measurement.nbits = data.header.nbits;
  measurement.foff = data.header.foff;
  measurement.samples = sample_count(data.header);
  return measurement;
}

void copy_metadata(const Measurement& measurement, BenchmarkResult& result) {
  result.bytes = measurement.bytes;
  result.nsamples = measurement.nsamples;
  result.nifs = measurement.nifs;
  result.nchans = measurement.nchans;
  result.nbits = measurement.nbits;
  result.foff = measurement.foff;
  result.samples = measurement.samples;
}

BenchmarkResult run_mode(const std::filesystem::path& path, Mode mode, int iterations) {
  BenchmarkResult result;
  result.mode = mode;
  result.iterations = iterations;
  result.file_size_bytes = std::filesystem::file_size(path);
  result.runs.reserve(static_cast<std::size_t>(iterations));

  for (int index = 0; index < iterations; ++index) {
    const Measurement measurement =
        mode == Mode::Legacy ? measure_legacy_once(path) : measure_eager_once(path, mode);
    copy_metadata(measurement, result);

    const double throughput = mib_per_second(measurement.bytes, measurement.seconds);
    result.runs.push_back({index, measurement.seconds, throughput});

    std::cout << "mode=" << mode_name(mode)
              << " iteration=" << index
              << " seconds=" << measurement.seconds
              << " mib_per_sec=" << throughput
              << " nsamples=" << measurement.nsamples
              << " nifs=" << measurement.nifs
              << " nchans=" << measurement.nchans
              << " nbits=" << measurement.nbits
              << " sample_count=" << measurement.samples
              << " foff=" << measurement.foff
              << '\n';
  }

  return result;
}

SummaryStats summarize(const BenchmarkResult& result) {
  SummaryStats stats;
  stats.min_seconds = result.runs.front().seconds;
  stats.max_seconds = result.runs.front().seconds;
  std::vector<double> seconds;
  std::vector<double> mib_per_sec;
  seconds.reserve(result.runs.size());
  mib_per_sec.reserve(result.runs.size());

  for (const IterationResult& run : result.runs) {
    stats.mean_seconds += run.seconds;
    stats.mean_mib_per_sec += run.mib_per_sec;
    seconds.push_back(run.seconds);
    mib_per_sec.push_back(run.mib_per_sec);
    stats.min_seconds = std::min(stats.min_seconds, run.seconds);
    stats.max_seconds = std::max(stats.max_seconds, run.seconds);
    stats.best_mib_per_sec = std::max(stats.best_mib_per_sec, run.mib_per_sec);
  }
  stats.mean_seconds /= static_cast<double>(result.runs.size());
  stats.mean_mib_per_sec /= static_cast<double>(result.runs.size());
  std::sort(seconds.begin(), seconds.end());
  std::sort(mib_per_sec.begin(), mib_per_sec.end());
  const std::size_t middle = seconds.size() / 2;
  if (seconds.size() % 2 == 0) {
    stats.median_seconds = (seconds[middle - 1] + seconds[middle]) / 2.0;
    stats.median_mib_per_sec = (mib_per_sec[middle - 1] + mib_per_sec[middle]) / 2.0;
  } else {
    stats.median_seconds = seconds[middle];
    stats.median_mib_per_sec = mib_per_sec[middle];
  }

  double variance = 0.0;
  for (const IterationResult& run : result.runs) {
    const double delta = run.seconds - stats.mean_seconds;
    variance += delta * delta;
  }
  stats.stddev_seconds = std::sqrt(variance / static_cast<double>(result.runs.size()));
  return stats;
}

void print_seconds_list(const BenchmarkResult& result) {
  std::cout << "final_samples"
            << " mode=" << mode_name(result.mode)
            << " seconds=[";
  for (std::size_t index = 0; index < result.runs.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << result.runs[index].seconds;
  }
  std::cout << "]\n";
}

void print_mode_report(const BenchmarkResult& result) {
  const SummaryStats stats = summarize(result);
  const gaffa::FilterbankReadOptions options = options_for_mode(result.mode);
  const auto payload_bytes = static_cast<std::uint64_t>(result.bytes);
  const std::uint64_t sample_element_bytes =
      result.nbits > 0 ? static_cast<std::uint64_t>(result.nbits / 8) : 0;
  std::cout << "final_report"
            << " mode=" << mode_name(result.mode)
            << " iterations=" << result.iterations
            << " mean_seconds=" << stats.mean_seconds
            << " median_seconds=" << stats.median_seconds
            << " min_seconds=" << stats.min_seconds
            << " max_seconds=" << stats.max_seconds
            << " stddev_seconds=" << stats.stddev_seconds
            << " mean_mib_per_sec=" << stats.mean_mib_per_sec
            << " median_mib_per_sec=" << stats.median_mib_per_sec
            << " best_mib_per_sec=" << stats.best_mib_per_sec
            << " file_size_bytes=" << result.file_size_bytes
            << " payload_bytes=" << payload_bytes
            << " sample_memory_bytes=" << payload_bytes
            << " sample_element_bytes=" << sample_element_bytes
            << " nsamples=" << result.nsamples
            << " nifs=" << result.nifs
            << " nchans=" << result.nchans
            << " nbits=" << result.nbits
            << " sample_count=" << result.samples
            << " io_buffer_bytes=" << options.io_buffer_bytes
            << " omp_max_threads=" << omp_max_threads()
            << " channel_order=" << channel_order_name(options.channel_order)
            << " reverse_backend=" << reported_reverse_backend(result.mode, options)
            << " foff=" << result.foff
            << '\n';
  print_seconds_list(result);
}

double legacy_baseline_seconds(const std::vector<BenchmarkResult>& results) {
  for (const BenchmarkResult& result : results) {
    if (result.mode == Mode::Legacy) {
      return summarize(result).mean_seconds;
    }
  }
  return std::numeric_limits<double>::quiet_NaN();
}

void print_final_report(const std::vector<BenchmarkResult>& results) {
  std::cout << "final_report_begin"
            << " modes=" << results.size()
            << " iterations=" << results.front().iterations
            << " file_size_bytes=" << results.front().file_size_bytes
            << " payload_bytes=" << static_cast<std::uint64_t>(results.front().bytes)
            << " omp_max_threads=" << omp_max_threads()
            << '\n';

  for (const BenchmarkResult& result : results) {
    print_mode_report(result);
  }

  const double baseline = legacy_baseline_seconds(results);
  std::cout << "overview_table"
            << " mode mean_seconds median_seconds min_seconds max_seconds stddev_seconds"
            << " mean_mib_per_sec median_mib_per_sec best_mib_per_sec speedup_vs_legacy\n";
  for (const BenchmarkResult& result : results) {
    const SummaryStats stats = summarize(result);
    std::cout << "overview_row"
              << " " << mode_name(result.mode)
              << " " << stats.mean_seconds
              << " " << stats.median_seconds
              << " " << stats.min_seconds
              << " " << stats.max_seconds
              << " " << stats.stddev_seconds
              << " " << stats.mean_mib_per_sec
              << " " << stats.median_mib_per_sec
              << " " << stats.best_mib_per_sec
              << " " << baseline / stats.mean_seconds
              << '\n';
  }
  std::cout << "final_report_end\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    std::vector<BenchmarkResult> results;
    const std::vector<Mode> modes = selected_modes(args.mode);
    results.reserve(modes.size());
    for (Mode mode : modes) {
      results.push_back(run_mode(args.path, mode, args.iterations));
    }
    print_final_report(results);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
