#include "gaffa/dedispersion.h"
#include "gaffa/dedispersion_cuda.h"
#include "gaffa/dm_search.h"
#include "gaffa/filterbank.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/preprocessing.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

struct Args {
  std::filesystem::path path;
  std::string dedispersion_backend = "cuda-subband";
  std::size_t ndm = 32;
  double dm_low = 1100.20;
  double dm_step = 1.0;
  double period_min = 0.1;
  double period_max = 1;
  std::size_t max_candidates = 1024;
  float snr_threshold = 6.0F;
  std::size_t bins_min = 200;
  std::size_t bins_max = 256;
  std::string preprocess = "riptide";
  double running_median_seconds = 5.0;
  std::size_t subband_channels = 32;
  std::size_t ndm_per_nominal = 32;
};

struct Timings {
  double read_seconds = 0.0;
  double dedisperse_seconds = 0.0;
  double search_seconds = 0.0;
  double total_seconds = 0.0;
};

void usage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " <filterbank.fil> [cpu-subband|cuda-subband]"
      << " [ndm] [dm_low] [dm_step] [period_min] [period_max]"
      << " [max_candidates] [snr_threshold] [bins_min] [bins_max]"
      << " [none|normalise|riptide] [running_median_seconds]"
      << " [subband_channels] [ndm_per_nominal]\n";
}

template <typename T>
T parse_number(const char* value);

template <>
std::size_t parse_number<std::size_t>(const char* value) {
  return static_cast<std::size_t>(std::stoull(value));
}

template <>
double parse_number<double>(const char* value) {
  return std::stod(value);
}

template <>
float parse_number<float>(const char* value) {
  return std::stof(value);
}

Args parse_args(int argc, char** argv) {
  if (argc < 2 || argc > 16) {
    usage(argv[0]);
    throw std::invalid_argument("invalid argument count");
  }

  Args args;
  args.path = argv[1];
  if (argc >= 3) {
    args.dedispersion_backend = argv[2];
  }
  if (argc >= 4) {
    args.ndm = parse_number<std::size_t>(argv[3]);
  }
  if (argc >= 5) {
    args.dm_low = parse_number<double>(argv[4]);
  }
  if (argc >= 6) {
    args.dm_step = parse_number<double>(argv[5]);
  }
  if (argc >= 7) {
    args.period_min = parse_number<double>(argv[6]);
  }
  if (argc >= 8) {
    args.period_max = parse_number<double>(argv[7]);
  }
  if (argc >= 9) {
    args.max_candidates = parse_number<std::size_t>(argv[8]);
  }
  if (argc >= 10) {
    args.snr_threshold = parse_number<float>(argv[9]);
  }
  if (argc >= 11) {
    args.bins_min = parse_number<std::size_t>(argv[10]);
  }
  if (argc >= 12) {
    args.bins_max = parse_number<std::size_t>(argv[11]);
  }
  if (argc >= 13) {
    args.preprocess = argv[12];
  }
  if (argc >= 14) {
    args.running_median_seconds = parse_number<double>(argv[13]);
  }
  if (argc >= 15) {
    args.subband_channels = parse_number<std::size_t>(argv[14]);
  }
  if (argc >= 16) {
    args.ndm_per_nominal = parse_number<std::size_t>(argv[15]);
  }

  if (args.dedispersion_backend != "cpu-subband" &&
      args.dedispersion_backend != "cuda-subband") {
    throw std::invalid_argument(
        "dedispersion backend must be cpu-subband or cuda-subband");
  }
  if (args.ndm == 0) {
    throw std::invalid_argument("ndm must be > 0");
  }
  if (!std::isfinite(args.dm_low) || !(args.dm_step > 0.0) ||
      !std::isfinite(args.dm_step)) {
    throw std::invalid_argument("dm_low must be finite and dm_step must be finite and > 0");
  }
  if (!(args.period_min > 0.0) || !(args.period_max > args.period_min) ||
      !std::isfinite(args.period_min) || !std::isfinite(args.period_max)) {
    throw std::invalid_argument("period range must be finite and satisfy 0 < min < max");
  }
  if (args.max_candidates == 0) {
    throw std::invalid_argument("max_candidates must be > 0");
  }
  if (!std::isfinite(args.snr_threshold)) {
    throw std::invalid_argument("snr_threshold must be finite");
  }
  if (args.bins_min <= 1 || args.bins_max < args.bins_min) {
    throw std::invalid_argument("bins must satisfy 1 < bins_min <= bins_max");
  }
  if (args.preprocess != "none" && args.preprocess != "normalise" &&
      args.preprocess != "riptide") {
    throw std::invalid_argument("preprocess must be none, normalise, or riptide");
  }
  if (!(args.running_median_seconds > 0.0) ||
      !std::isfinite(args.running_median_seconds)) {
    throw std::invalid_argument("running_median_seconds must be finite and > 0");
  }
  if (args.subband_channels == 0 || args.ndm_per_nominal == 0) {
    throw std::invalid_argument("subband options must be > 0");
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

std::size_t sample_bytes(const gaffa::FilterbankData& data) {
  return std::visit(
      [](const auto& samples) { return samples.size() * sizeof(samples[0]); },
      data.samples);
}

std::vector<double> dm_values(const Args& args) {
  std::vector<double> values(args.ndm);
  for (std::size_t index = 0; index < args.ndm; ++index) {
    values[index] = args.dm_low + static_cast<double>(index) * args.dm_step;
  }
  return values;
}

gaffa::MultiDmDedispersionPlan dedispersion_plan(
    const gaffa::FilterbankHeader& header,
    const Args& args) {
  return gaffa::MultiDmDedispersionPlan{
      .dm_low = args.dm_low,
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

gaffa::CudaDedispersionOptions cuda_options() {
  return gaffa::CudaDedispersionOptions{
      .device_id = 0,
      .threads_per_block = 256,
  };
}

gaffa::PreprocessPlan preprocess_plan(const Args& args, double tsamp) {
  if (args.preprocess == "none") {
    return {};
  }
  if (args.preprocess == "normalise") {
    return gaffa::PreprocessPlan{
        .steps = {
            gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise},
        },
    };
  }
  return gaffa::make_riptide_preprocess_plan(
      tsamp, gaffa::RiptidePreprocessOptions{
                 .running_median_width_seconds = args.running_median_seconds,
                 .normalise = true,
             });
}

gaffa::DmSearchOptions search_options(const Args& args, double tsamp) {
  const double min_period = tsamp * static_cast<double>(args.bins_min);
  if (args.period_min < min_period) {
    throw std::invalid_argument(
        "period_min must be >= tsamp * bins_min for the FFA plan");
  }
  return gaffa::DmSearchOptions{
      .plan = gaffa::RiptideFfaPlanOptions{
          .period_min = args.period_min,
          .period_max = args.period_max,
          .bins_min = args.bins_min,
          .bins_max = args.bins_max,
      },
      .preprocess = preprocess_plan(args, tsamp),
      .snr_threshold = args.snr_threshold,
      .max_candidates = args.max_candidates,
  };
}

template <typename T>
gaffa::DmSearchResult run_typed_search(const gaffa::FilterbankData& filterbank,
                                       const Args& args,
                                       Timings& timings) {
  const auto view = gaffa::sample_view<T>(filterbank);
  using OutT = std::conditional_t<std::is_same_v<T, float>, float, std::uint32_t>;
  gaffa::DedispersedResult<OutT> dedispersed;
  timings.dedisperse_seconds = time_once([&] {
    if (args.dedispersion_backend == "cpu-subband") {
      dedispersed = gaffa::dedisperse_subband_cpu(
          view, filterbank.header.frequency_table,
          dedispersion_plan(filterbank.header, args), subband_options(args));
    } else {
      dedispersed = gaffa::dedisperse_subband_cuda(
          view, filterbank.header.frequency_table,
          dedispersion_plan(filterbank.header, args), subband_options(args),
          cuda_options());
    }
  });

  gaffa::DmSearchResult search_result;
  const std::vector<double> dms = dm_values(args);
  const gaffa::DmSearchOptions options =
      search_options(args, filterbank.header.tsamp);
  timings.search_seconds = time_once([&] {
    search_result =
        gaffa::search_dms_cpu(dedispersed, dms, filterbank.header.tsamp, options);
  });
  return search_result;
}

gaffa::DmSearchResult run_search(const gaffa::FilterbankData& filterbank,
                                 const Args& args,
                                 Timings& timings) {
  return std::visit(
      [&](const auto& values) -> gaffa::DmSearchResult {
        using T = typename std::decay_t<decltype(values)>::value_type;
        return run_typed_search<T>(filterbank, args, timings);
      },
      filterbank.samples);
}

void print_candidate(std::size_t rank, const gaffa::DmCandidate& candidate) {
  const gaffa::FfaCandidate& ffa = candidate.ffa;
  std::cout << "candidate"
            << " rank=" << rank
            << " dm=" << candidate.dm
            << " dm_index=" << candidate.dm_index
            << " snr=" << ffa.snr
            << " period=" << ffa.period
            << " width=" << ffa.width
            << " phase=" << ffa.phase
            << " shift=" << ffa.shift
            << " bins=" << ffa.bins << '\n';
}

void print_report(const Args& args,
                  const gaffa::FilterbankData& filterbank,
                  const gaffa::DmSearchResult& result,
                  const Timings& timings) {
  const gaffa::FilterbankHeader& header = filterbank.header;
  std::cout << "dm_search_begin"
            << " file=" << args.path
            << " dedispersion_backend=" << args.dedispersion_backend
            << " dtype=" << dtype_name(filterbank.samples)
            << " nsamples=" << header.nsamples
            << " nifs=" << header.nifs
            << " nchans=" << header.nchans
            << " nbits=" << header.nbits
            << " tsamp=" << header.tsamp
            << " input_bytes=" << sample_bytes(filterbank)
            << " ndm=" << args.ndm
            << " dm_low=" << args.dm_low
            << " dm_step=" << args.dm_step
            << " period_min=" << args.period_min
            << " period_max=" << args.period_max
            << " bins_min=" << args.bins_min
            << " bins_max=" << args.bins_max
            << " preprocess=" << args.preprocess
            << " snr_threshold=" << args.snr_threshold
            << " max_candidates=" << args.max_candidates << '\n';
  std::cout << "timing"
            << " read_seconds=" << timings.read_seconds
            << " dedisperse_seconds=" << timings.dedisperse_seconds
            << " search_seconds=" << timings.search_seconds
            << " total_seconds=" << timings.total_seconds << '\n';
  std::cout << "result candidates=" << result.candidates.size() << '\n';
  for (std::size_t rank = 0; rank < result.candidates.size(); ++rank) {
    print_candidate(rank, result.candidates[rank]);
  }
  std::cout << "dm_search_end\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto total_start = std::chrono::steady_clock::now();
    const Args args = parse_args(argc, argv);

    gaffa::FilterbankData filterbank;
    Timings timings;
    timings.read_seconds =
        time_once([&] { filterbank = gaffa::read_filterbank(args.path); });
    const gaffa::DmSearchResult result = run_search(filterbank, args, timings);
    const auto total_end = std::chrono::steady_clock::now();
    timings.total_seconds =
        std::chrono::duration<double>(total_end - total_start).count();

    print_report(args, filterbank, result, timings);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
