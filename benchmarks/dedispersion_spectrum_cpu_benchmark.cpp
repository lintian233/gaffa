#include "gaffa/dedispersion.h"
#include "gaffa/filterbank.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/internal/dedispersion_delay.h"

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
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

struct Args {
  std::filesystem::path path;
  double dm = 1050.2;
  int iterations = 3;
  std::size_t chan_begin = 0;
  std::size_t chan_end = 0;
};

struct IterationResult {
  int iteration = 0;
  double seconds = 0.0;
  std::size_t output_nsamples = 0;
  std::size_t output_nchans = 0;
  std::size_t output_bytes = 0;
};

struct SummaryStats {
  double mean_seconds = 0.0;
  double median_seconds = 0.0;
  double min_seconds = 0.0;
  double max_seconds = 0.0;
  double stddev_seconds = 0.0;
};

struct DelayRunStats {
  std::size_t channel_count = 0;
  std::size_t run_count = 0;
  std::size_t unique_delay_count = 0;
  std::size_t max_run_length = 0;
  std::size_t singleton_run_count = 0;
  double average_run_length = 0.0;
};

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <filterbank.fil> [dm] [iterations] [chan_begin] [chan_end]\n";
}

Args parse_args(int argc, char** argv) {
  if (argc < 2 || argc > 6) {
    usage(argv[0]);
    throw std::invalid_argument("invalid argument count");
  }

  Args args;
  args.path = argv[1];
  if (argc >= 3) {
    args.dm = std::stod(argv[2]);
  }
  if (argc >= 4) {
    args.iterations = std::stoi(argv[3]);
  }
  if (argc >= 5) {
    args.chan_begin = static_cast<std::size_t>(std::stoull(argv[4]));
  }
  if (argc >= 6) {
    args.chan_end = static_cast<std::size_t>(std::stoull(argv[5]));
  }

  if (!std::filesystem::exists(args.path)) {
    throw std::invalid_argument("filterbank file does not exist");
  }
  if (!std::isfinite(args.dm) || args.dm < 0.0) {
    throw std::invalid_argument("dm must be finite and non-negative");
  }
  if (args.iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  if (args.chan_end != 0 && args.chan_begin >= args.chan_end) {
    throw std::invalid_argument("channel range must be non-empty");
  }
  return args;
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

std::size_t sample_bytes(const gaffa::FilterbankData& filterbank) {
  return std::visit(
      [](const auto& samples) { return samples.size() * sizeof(samples[0]); },
      filterbank.samples);
}

gaffa::SingleDmDedispersionPlan make_plan(
    const gaffa::FilterbankHeader& header, const Args& args) {
  const std::size_t nchans = static_cast<std::size_t>(header.nchans);
  const std::size_t chan_end = args.chan_end == 0 ? nchans : args.chan_end;
  if (chan_end > nchans) {
    throw std::invalid_argument("channel range exceeds sample channel count");
  }
  return gaffa::SingleDmDedispersionPlan{
      .dm = args.dm,
      .ref_frequency_mhz = header.frequency_table.back(),
      .tsamp = header.tsamp,
      .chan_begin = args.chan_begin,
      .chan_end = chan_end,
  };
}

template <typename T>
IterationResult make_iteration_result(
    int iteration, double seconds,
    const gaffa::DedispersedSpectrum<T>& spectrum) {
  return IterationResult{
      .iteration = iteration,
      .seconds = seconds,
      .output_nsamples = spectrum.nsamples(),
      .output_nchans = spectrum.nchans(),
      .output_bytes = spectrum.data.size() * sizeof(T),
  };
}

template <typename T>
IterationResult run_typed_once(const gaffa::FilterbankData& filterbank,
                               const gaffa::SingleDmDedispersionPlan& plan,
                               int iteration) {
  gaffa::DedispersedSpectrum<T> spectrum;
  const auto view = gaffa::sample_view<T>(filterbank);
  const double seconds = time_once([&] {
    spectrum = gaffa::dedisperse_spectrum_cpu(
        view, filterbank.header.frequency_table, plan);
  });
  return make_iteration_result(iteration, seconds, spectrum);
}

IterationResult run_once(const gaffa::FilterbankData& filterbank,
                         const gaffa::SingleDmDedispersionPlan& plan,
                         int iteration) {
  return std::visit(
      [&](const auto& samples) {
        using T = typename std::decay_t<decltype(samples)>::value_type;
        return run_typed_once<T>(filterbank, plan, iteration);
      },
      filterbank.samples);
}

SummaryStats summarize(const std::vector<IterationResult>& runs) {
  SummaryStats stats;
  stats.min_seconds = runs.front().seconds;
  stats.max_seconds = runs.front().seconds;
  std::vector<double> seconds;
  seconds.reserve(runs.size());
  for (const IterationResult& run : runs) {
    seconds.push_back(run.seconds);
    stats.mean_seconds += run.seconds;
    stats.min_seconds = std::min(stats.min_seconds, run.seconds);
    stats.max_seconds = std::max(stats.max_seconds, run.seconds);
  }
  stats.mean_seconds /= static_cast<double>(runs.size());
  std::sort(seconds.begin(), seconds.end());
  const std::size_t middle = seconds.size() / 2;
  stats.median_seconds = seconds.size() % 2 == 0
                             ? (seconds[middle - 1] + seconds[middle]) / 2.0
                             : seconds[middle];
  double variance = 0.0;
  for (const IterationResult& run : runs) {
    const double delta = run.seconds - stats.mean_seconds;
    variance += delta * delta;
  }
  stats.stddev_seconds =
      std::sqrt(variance / static_cast<double>(runs.size()));
  return stats;
}

double gib_per_second(std::size_t bytes, double seconds) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / seconds;
}

DelayRunStats summarize_delay_runs(
    std::span<const std::int32_t> delays) {
  if (delays.empty()) {
    throw std::invalid_argument("delay table must not be empty");
  }

  DelayRunStats stats;
  stats.channel_count = delays.size();

  std::vector<std::int32_t> unique_delays(delays.begin(), delays.end());
  std::sort(unique_delays.begin(), unique_delays.end());
  unique_delays.erase(std::unique(unique_delays.begin(), unique_delays.end()),
                      unique_delays.end());
  stats.unique_delay_count = unique_delays.size();

  std::size_t run_length = 1;
  for (std::size_t index = 1; index < delays.size(); ++index) {
    if (delays[index] == delays[index - 1]) {
      ++run_length;
      continue;
    }
    ++stats.run_count;
    stats.max_run_length = std::max(stats.max_run_length, run_length);
    stats.singleton_run_count += static_cast<std::size_t>(run_length == 1);
    run_length = 1;
  }
  ++stats.run_count;
  stats.max_run_length = std::max(stats.max_run_length, run_length);
  stats.singleton_run_count += static_cast<std::size_t>(run_length == 1);
  stats.average_run_length =
      static_cast<double>(stats.channel_count) /
      static_cast<double>(stats.run_count);
  return stats;
}

void print_report(const gaffa::FilterbankData& filterbank, const Args& args,
                  const gaffa::SingleDmDedispersionPlan& plan,
                  double read_seconds,
                  const std::vector<IterationResult>& runs) {
  const SummaryStats stats = summarize(runs);
  const std::vector<std::int32_t> delays =
      gaffa::internal::make_single_dm_delay_table(
          filterbank.header.frequency_table, plan.dm, plan.ref_frequency_mhz,
          plan.tsamp, plan.chan_begin, plan.chan_end);
  const DelayRunStats delay_stats = summarize_delay_runs(delays);
  const std::int32_t max_delay = gaffa::internal::single_dm_max_delay(
      filterbank.header.frequency_table, plan);
  const IterationResult& last = runs.back();

  std::cout << "final_report_begin"
            << " iterations=" << args.iterations
            << " read_seconds=" << read_seconds
            << " file=\"" << args.path.string() << '"'
            << " dtype=" << dtype_name(filterbank.samples)
            << " input_nsamples=" << filterbank.header.nsamples
            << " input_nifs=" << filterbank.header.nifs
            << " input_nchans=" << filterbank.header.nchans
            << " input_nbits=" << filterbank.header.nbits
            << " input_bytes=" << sample_bytes(filterbank)
            << " dm=" << plan.dm
            << " ref_frequency_mhz=" << plan.ref_frequency_mhz
            << " tsamp=" << plan.tsamp
            << " chan_begin=" << plan.chan_begin
            << " chan_end=" << plan.chan_end
            << " max_delay=" << max_delay
            << " delay_runs=" << delay_stats.run_count
            << " unique_delays=" << delay_stats.unique_delay_count
            << " avg_run_length=" << delay_stats.average_run_length
            << " max_run_length=" << delay_stats.max_run_length
            << " singleton_runs=" << delay_stats.singleton_run_count
            << " output_nsamples=" << last.output_nsamples
            << " output_nchans=" << last.output_nchans
            << " output_bytes=" << last.output_bytes
            << '\n';

  std::cout << "final_report"
            << " mode=spectrum_cpu"
            << " mean_seconds=" << stats.mean_seconds
            << " median_seconds=" << stats.median_seconds
            << " min_seconds=" << stats.min_seconds
            << " max_seconds=" << stats.max_seconds
            << " stddev_seconds=" << stats.stddev_seconds
            << " throughput_gib_per_second="
            << gib_per_second(last.output_bytes, stats.mean_seconds)
            << '\n';
  std::cout << "final_report_end\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    const Args args = parse_args(argc, argv);

    gaffa::FilterbankData filterbank;
    const double read_seconds =
        time_once([&] { filterbank = gaffa::read_filterbank(args.path); });
    if (filterbank.header.nifs != 1) {
      throw std::runtime_error("spectrum benchmark requires nifs == 1");
    }

    const gaffa::SingleDmDedispersionPlan plan =
        make_plan(filterbank.header, args);
    std::vector<IterationResult> runs;
    runs.reserve(static_cast<std::size_t>(args.iterations));
    for (int iteration = 0; iteration < args.iterations; ++iteration) {
      IterationResult run = run_once(filterbank, plan, iteration);
      std::cout << "mode=spectrum_cpu"
                << " iteration=" << iteration
                << " seconds=" << run.seconds
                << " output_nsamples=" << run.output_nsamples
                << " output_nchans=" << run.output_nchans
                << " output_bytes=" << run.output_bytes
                << " throughput_gib_per_second="
                << gib_per_second(run.output_bytes, run.seconds)
                << '\n';
      runs.push_back(run);
    }
    print_report(filterbank, args, plan, read_seconds, runs);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
