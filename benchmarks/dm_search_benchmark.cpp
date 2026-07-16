#include "gaffa/candidate.h"
#include "gaffa/dedispersion.h"
#include "gaffa/dedispersion_cuda.h"
#include "gaffa/dm_search.h"
#include "gaffa/filterbank.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/harmonic.h"
#include "gaffa/preprocessing.h"

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
  std::size_t max_peaks = 0;
  std::size_t print_peaks = 64;
  float snr_threshold = 6.0F;
  std::size_t bins_min = 200;
  std::size_t bins_max = 256;
  std::string preprocess = "riptide";
  double running_median_seconds = 5.0;
  std::size_t subband_channels = 32;
  std::size_t ndm_per_nominal = 32;
  std::size_t max_candidates = 0;
};

struct Timings {
  double read_seconds = 0.0;
  double dedisperse_seconds = 0.0;
  double search_seconds = 0.0;
  double candidate_seconds = 0.0;
  double harmonic_seconds = 0.0;
  double total_seconds = 0.0;
};

void usage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " <filterbank.fil> [cpu-subband|cuda-subband]"
      << " [ndm] [dm_low] [dm_step] [period_min] [period_max]"
      << " [max_peaks] [print_peaks] [snr_threshold] [bins_min] [bins_max]"
      << " [none|normalise|riptide] [running_median_seconds]"
      << " [subband_channels] [ndm_per_nominal] [max_candidates]\n";
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
  if (argc < 2 || argc > 18) {
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
    args.max_peaks = parse_number<std::size_t>(argv[8]);
  }
  if (argc >= 10) {
    args.print_peaks = parse_number<std::size_t>(argv[9]);
  }
  if (argc >= 11) {
    args.snr_threshold = parse_number<float>(argv[10]);
  }
  if (argc >= 12) {
    args.bins_min = parse_number<std::size_t>(argv[11]);
  }
  if (argc >= 13) {
    args.bins_max = parse_number<std::size_t>(argv[12]);
  }
  if (argc >= 14) {
    args.preprocess = argv[13];
  }
  if (argc >= 15) {
    args.running_median_seconds = parse_number<double>(argv[14]);
  }
  if (argc >= 16) {
    args.subband_channels = parse_number<std::size_t>(argv[15]);
  }
  if (argc >= 17) {
    args.ndm_per_nominal = parse_number<std::size_t>(argv[16]);
  }
  if (argc >= 18) {
    args.max_candidates = parse_number<std::size_t>(argv[17]);
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
      .max_peaks = args.max_peaks,
  };
}

gaffa::CandidateSelectionOptions candidate_options() {
  return gaffa::CandidateSelectionOptions{
      .frequency_cluster_radius = 0.1,
      .dm_cluster_radius = 50,
      .cluster_across_widths = true,
      .max_candidates = 0,
  };
}

gaffa::HarmonicOptions harmonic_options() {
  return gaffa::HarmonicOptions{
      .max_harmonic = 32,
      .denominator_max = 100,
      .frequency_tolerance_bins = 1.5,
      .phase_distance_max = 1.0,
      .dm_distance_max = 3.0,
      .use_snr_consistency = false,
      .snr_distance_max = 3.0,
  };
}

gaffa::HarmonicContext harmonic_context(
    const gaffa::FilterbankHeader& header) {
  const auto [low, high] = std::minmax_element(header.frequency_table.begin(),
                                               header.frequency_table.end());
  return gaffa::HarmonicContext{
      .observation_seconds =
          header.tsamp * static_cast<double>(header.nsamples),
      .frequency_low_mhz = *low,
      .frequency_high_mhz = *high,
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
        gaffa::search_dedispersed_ffa_cpu(
            dedispersed, dms, filterbank.header.tsamp, options);
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

void print_peak(std::size_t rank, const gaffa::DmPeak& dm_peak) {
  const gaffa::PeriodicPeak& peak = dm_peak.peak;
  std::cout << "peak"
            << " rank=" << rank
            << " dm=" << dm_peak.dm
            << " dm_index=" << dm_peak.dm_index
            << " snr=" << peak.snr
            << " period=" << peak.period_seconds()
            << " frequency=" << peak.motion.frequency_hz
            << " width=" << peak.boxcar_width_bins
            << " duty_cycle=" << peak.duty_cycle
            << " phase=" << peak.phase_bin.value_or(0)
            << " phase_known=" << peak.phase_bin.has_value()
            << " bins=" << peak.phase_bins << '\n';
}

void print_candidate(std::string_view label,
                     std::size_t rank,
                     const gaffa::Candidate& candidate) {
  const auto& best = candidate.best;
  const auto& peak = best.peak;
  std::cout << label
            << " rank=" << rank
            << " dm=" << best.dm
            << " dm_index=" << best.dm_index
            << " snr=" << peak.snr
            << " period=" << peak.period_seconds()
            << " frequency=" << peak.motion.frequency_hz
            << " width=" << peak.boxcar_width_bins
            << " duty_cycle=" << peak.duty_cycle
            << " peak_count=" << candidate.peak_count
            << " dm_index_min=" << candidate.extent.dm_index_min
            << " dm_index_max=" << candidate.extent.dm_index_max
            << " frequency_min=" << candidate.extent.frequency_hz.minimum
            << " frequency_max=" << candidate.extent.frequency_hz.maximum << '\n';
}

void print_harmonic_candidate(std::size_t rank,
                              const gaffa::HarmonicCandidate& candidate) {
  const gaffa::HarmonicRelation& harmonic = candidate.harmonic;
  const auto& best = candidate.candidate.best;
  const auto& peak = best.peak;
  std::cout << "harmonic_candidate"
            << " rank=" << rank
            << " parent_rank=" << harmonic.parent_index
            << " ratio=" << harmonic.numerator << '/'
            << harmonic.denominator
            << " frequency_error_bins=" << harmonic.frequency_error_bins
            << " phase_distance=" << harmonic.phase_distance
            << " dm_distance=" << harmonic.dm_distance
            << " expected_snr=" << harmonic.expected_snr
            << " snr_distance=" << harmonic.snr_distance
            << " dm=" << best.dm
            << " dm_index=" << best.dm_index
            << " snr=" << peak.snr
            << " period=" << peak.period_seconds()
            << " frequency=" << peak.motion.frequency_hz
            << " width=" << peak.boxcar_width_bins
            << " duty_cycle=" << peak.duty_cycle
            << " peak_count=" << candidate.candidate.peak_count << '\n';
}

void print_report(const Args& args,
                  const gaffa::FilterbankData& filterbank,
                  const gaffa::DmSearchResult& result,
                  const std::vector<gaffa::Candidate>& candidates,
                  const std::vector<gaffa::HarmonicCandidate>& flagged_candidates,
                  const std::vector<gaffa::Candidate>& filtered_candidates,
                  const Timings& timings) {
  const gaffa::FilterbankHeader& header = filterbank.header;
  const gaffa::CandidateSelectionOptions candidate = candidate_options();
  const gaffa::HarmonicOptions harmonic = harmonic_options();
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
            << " max_peaks=" << args.max_peaks
            << " max_candidates=" << args.max_candidates
            << " candidate_frequency_cluster_radius="
            << candidate.frequency_cluster_radius
            << " candidate_dm_cluster_radius="
            << candidate.dm_cluster_radius
            << " candidate_cluster_across_widths="
            << candidate.cluster_across_widths
            << " candidate_max_candidates=" << candidate.max_candidates
            << " harmonic_max_harmonic=" << harmonic.max_harmonic
            << " harmonic_denominator_max=" << harmonic.denominator_max
            << " harmonic_frequency_tolerance_bins="
            << harmonic.frequency_tolerance_bins
            << " harmonic_phase_distance_max=" << harmonic.phase_distance_max
            << " harmonic_dm_distance_max=" << harmonic.dm_distance_max
            << " harmonic_use_snr_consistency="
            << harmonic.use_snr_consistency
            << " print_peaks=" << args.print_peaks << '\n';
  std::cout << "timing"
            << " read_seconds=" << timings.read_seconds
            << " dedisperse_seconds=" << timings.dedisperse_seconds
            << " search_seconds=" << timings.search_seconds
            << " candidate_seconds=" << timings.candidate_seconds
            << " harmonic_seconds=" << timings.harmonic_seconds
            << " total_seconds=" << timings.total_seconds << '\n';
  std::cout << "result"
            << " peaks=" << result.peaks.size()
            << " raw_candidates=" << candidates.size()
            << " harmonic_candidates="
            << std::count_if(flagged_candidates.begin(), flagged_candidates.end(),
                             [](const gaffa::HarmonicCandidate& candidate) {
                               return candidate.harmonic.is_harmonic;
                             })
            << " candidates=" << filtered_candidates.size() << '\n';
  const std::size_t candidate_print_count =
      std::min(args.print_peaks, candidates.size());
  for (std::size_t rank = 0; rank < candidate_print_count; ++rank) {
    print_candidate("raw_candidate", rank, candidates[rank]);
  }
  if (candidate_print_count < candidates.size()) {
    std::cout << "result raw_candidates_omitted="
              << candidates.size() - candidate_print_count << '\n';
  }
  std::size_t harmonic_printed = 0;
  for (std::size_t rank = 0;
       rank < flagged_candidates.size() && harmonic_printed < args.print_peaks;
       ++rank) {
    if (!flagged_candidates[rank].harmonic.is_harmonic) {
      continue;
    }
    print_harmonic_candidate(rank, flagged_candidates[rank]);
    ++harmonic_printed;
  }
  const std::size_t filtered_candidate_print_count =
      std::min(args.print_peaks, filtered_candidates.size());
  for (std::size_t rank = 0; rank < filtered_candidate_print_count; ++rank) {
    print_candidate("candidate", rank, filtered_candidates[rank]);
  }
  if (filtered_candidate_print_count < filtered_candidates.size()) {
    std::cout << "result candidates_omitted="
              << filtered_candidates.size() - filtered_candidate_print_count
              << '\n';
  }
  const std::size_t print_count = std::min(args.print_peaks, result.peaks.size());
  for (std::size_t rank = 0; rank < print_count; ++rank) {
    print_peak(rank, result.peaks[rank]);
  }
  if (print_count < result.peaks.size()) {
    std::cout << "result peaks_omitted=" << result.peaks.size() - print_count
              << '\n';
  }
  std::cout << "dm_search_end\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    const auto total_start = std::chrono::steady_clock::now();
    const Args args = parse_args(argc, argv);

    gaffa::FilterbankData filterbank;
    Timings timings;
    timings.read_seconds =
        time_once([&] { filterbank = gaffa::read_filterbank(args.path); });
    const gaffa::DmSearchResult result = run_search(filterbank, args, timings);
    std::vector<gaffa::Candidate> candidates;
    timings.candidate_seconds = time_once([&] {
      candidates = gaffa::select_candidates_cpu(
          result.peaks,
          filterbank.header.tsamp * static_cast<double>(filterbank.header.nsamples),
          candidate_options());
    });
    std::vector<gaffa::HarmonicCandidate> flagged_candidates;
    std::vector<gaffa::Candidate> filtered_candidates;
    timings.harmonic_seconds = time_once([&] {
      flagged_candidates = gaffa::flag_harmonics_cpu(
          candidates, harmonic_context(filterbank.header), harmonic_options());
      filtered_candidates =
          gaffa::remove_harmonics_cpu(flagged_candidates, args.max_candidates);
    });
    const auto total_end = std::chrono::steady_clock::now();
    timings.total_seconds =
        std::chrono::duration<double>(total_end - total_start).count();

    print_report(args, filterbank, result, candidates, flagged_candidates,
                 filtered_candidates, timings);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
