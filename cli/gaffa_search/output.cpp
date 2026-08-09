#include "output.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <set>
#include <system_error>

namespace gaffa_search {
namespace {

std::string fixed(double value, int precision) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string csv_value(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (const char character : value) {
    if (character == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

std::string device_list(const std::vector<int>& devices) {
  if (devices.empty()) {
    return "-";
  }
  std::ostringstream stream;
  for (std::size_t index = 0; index < devices.size(); ++index) {
    if (index != 0) {
      stream << ',';
    }
    stream << devices[index];
  }
  return stream.str();
}

const char* dedispersion_backend_name(DedispersionBackend backend) noexcept {
  switch (backend) {
    case DedispersionBackend::CpuSubband:
      return "cpu-subband";
    case DedispersionBackend::CudaSubband:
      return "cuda-subband";
  }
  return "unknown";
}

const char* effective_window_mode(const SearchRangeConfig& search) noexcept {
  // Native FFA accepts arbitrary input lengths, so its truncate setting is a
  // no-op. Native CUDA still honors an explicit zero-pad request in the
  // current execution path; Loki requires a power-of-two search length for
  // either non-original policy.
  if ((search.backend == Backend::NativeCpu ||
       search.backend == Backend::NativeCuda) &&
      search.window_mode == WindowMode::Truncate) {
    return "original";
  }
  return window_mode_name(search.window_mode);
}

std::string search_device_list(const Config& config,
                               const SearchRangeConfig& search) {
  if (search.backend == Backend::NativeCpu) {
    return "host";
  }
  if (search.backend == Backend::NativeCuda) {
    return device_list(config.native_cuda_devices);
  }
  return device_list(config.loki_cuda_devices);
}

void write_motion_range(std::ostream& output,
                        const SearchRangeConfig& search) {
  if (!search.motion.accel) {
    output << "motion=frequency-only";
    return;
  }
  output << "motion=accel[" << fixed(search.motion.accel->minimum, 6) << ","
         << fixed(search.motion.accel->maximum, 6) << "]";
  if (search.motion.jerk) {
    output << " jerk[" << fixed(search.motion.jerk->minimum, 6) << ","
           << fixed(search.motion.jerk->maximum, 6) << "]";
  }
}

void write_search_ranges(std::ostream& output, const Config& config) {
  output << "DM ranges (" << config.dm_ranges.size() << ")\n";
  for (std::size_t index = 0; index < config.dm_ranges.size(); ++index) {
    const DmRangeConfig& range = config.dm_ranges[index];
    const double dm_high =
        range.dm_low +
        (range.ndm == 0 ? 0.0
                        : static_cast<double>(range.ndm - 1) * range.dm_step);
    output << "  [" << index << "] dm=[" << fixed(range.dm_low, 3) << ","
           << fixed(dm_high, 3) << "] step=" << fixed(range.dm_step, 3)
           << " trials=" << range.ndm << '\n';
  }

  output << "\nSearch ranges (" << config.search_ranges.size() << ")\n";
  for (const SearchRangeConfig& search : config.search_ranges) {
    output << "  [" << search.id << "] backend="
           << backend_name(search.backend) << " devices=["
           << search_device_list(config, search) << "]\n"
           << "      period=[" << fixed(search.period_min, 6) << ","
           << fixed(search.period_max, 6) << "] s bins=["
           << search.bins_min << "," << search.bins_max
           << "] ";
    write_motion_range(output, search);
    output << " window=" << effective_window_mode(search) << '\n';
  }
}

template <typename Writer>
void write_atomically(const std::filesystem::path& path, bool overwrite,
                      Writer&& writer) {
  if (!overwrite && std::filesystem::exists(path)) {
    throw std::runtime_error("output file already exists: " + path.string());
  }
  if (const auto parent = path.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }

  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path temporary =
      path.string() + ".tmp." + std::to_string(stamp);
  try {
    std::ofstream output(temporary, std::ios::out | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("cannot open output file: " +
                               temporary.string());
    }
    writer(output);
    if (!output) {
      throw std::runtime_error("failed while writing output file: " +
                               temporary.string());
    }
    output.close();
    if (!output) {
      throw std::runtime_error("failed to close output file: " +
                               temporary.string());
    }
    if (overwrite) {
      std::error_code remove_error;
      std::filesystem::remove(path, remove_error);
      if (remove_error) {
        throw std::system_error(remove_error,
                                 "cannot replace output file");
      }
    }
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

void write_table_header(std::ostream& output) {
  output << "rank  DM        SNR     Period(ms)  Frequency(Hz)  "
            "Acceleration  Jerk       Snap       Members  DM range\n";
  output << "----  --------  ------  ----------  -------------  "
            "------------  ---------  ---------  -------  --------\n";
}

void write_table_row(std::ostream& output, const CandidateRow& row) {
  const std::string dm_range = fixed(row.dm_min, 3) + "-" + fixed(row.dm_max, 3);
  output << std::right << std::setw(4) << row.rank << "  "
         << std::setw(8) << fixed(row.dm, 3) << "  "
         << std::setw(6) << fixed(static_cast<double>(row.snr), 2) << "  "
         << std::setw(10) << fixed(row.period_seconds * 1000.0, 3) << "  "
         << std::setw(13) << fixed(row.frequency_hz, 5) << "  "
         << std::setw(12) << fixed(row.acceleration_m_per_s2, 6) << "  "
         << std::setw(9) << fixed(row.jerk_m_per_s3, 6) << "  "
         << std::setw(9) << fixed(row.snap_m_per_s4, 6) << "  "
         << std::setw(7) << row.member_count << "  " << dm_range << '\n';
}

void write_summary(std::ostream& output, const Config& config,
                   const Report& report) {
  output << "GAFFA SEARCH\n"
            "============\n\n"
         << "File: " << report.input.string() << "\n\n"
            "Observation\n"
            "  samples     "
         << report.observation_nsamples << "\n"
            "  tsamp       "
         << std::fixed << std::setprecision(9) << report.tsamp_seconds
         << " s\n"
            "  duration    "
         << std::setprecision(3) << report.observation_seconds << " s\n\n"
            "Search\n"
            "  dedispersion       "
         << dedispersion_backend_name(config.dedispersion_backend) << "\n"
            "  native devices     "
         << device_list(config.native_cuda_devices) << "\n"
            "  loki devices       "
         << device_list(config.loki_cuda_devices) << "\n"
            "  dm tile size       "
         << config.dm_tile_size << "\n"
            "  preprocess         "
         << config.preprocess << "\n\n";

  write_search_ranges(output, config);
  output << "\nTiming\n"
            "  read               "
         << report.timing.read_seconds << " s\n"
            "  dedispersion       "
         << report.timing.dedispersion_seconds << " s\n"
            "  search             "
         << report.timing.search_seconds << " s\n"
            "  candidate          "
         << report.timing.candidate_seconds << " s\n"
            "  total              "
         << report.timing.total_seconds << " s\n\n"
            "Candidates\n"
            "  raw peaks          "
         << report.raw_peak_count << "\n"
            "  candidate groups   "
         << report.candidate_count << "\n"
            "  harmonic relations "
         << report.harmonic_relation_count << "\n"
            "  total candidates   "
         << report.selected_count << "\n\n";
  output << "Status\n"
            "  complete           "
         << (report.complete ? "yes" : "no") << '\n';
  if (!report.warnings.empty()) {
    output << "  warnings           " << report.warnings.size() << '\n';
    for (const std::string& warning : report.warnings) {
      output << "    - " << warning << '\n';
    }
  }
  output << '\n';
}

void write_human_report(std::ostream& output, const Config& config,
                        const Report& report, std::size_t limit) {
  write_summary(output, config, report);
  output << "Candidate table\n"
            "---------------\n";
  write_table_header(output);
  const std::size_t count =
      limit == 0 ? report.candidates.size()
                 : std::min(limit, report.candidates.size());
  for (std::size_t index = 0; index < count; ++index) {
    write_table_row(output, report.candidates[index]);
  }
  output << "\nshowing            " << count << " of "
         << report.candidates.size() << '\n';
  if (count < report.candidates.size()) {
    output << "... " << report.candidates.size() - count
           << " candidates omitted from this view\n";
  }
  output << '\n';
}

void write_csv(std::ostream& output, const Report& report) {
  output << "# gaffa candidate file\n"
         << "# input," << csv_value(report.input.string()) << '\n'
         << "# raw_peaks," << report.raw_peak_count << '\n'
         << "# final_candidates," << report.selected_count << '\n'
         << "# complete," << (report.complete ? "true" : "false") << '\n';
  for (const std::string& warning : report.warnings) {
    output << "# warning," << csv_value(warning) << '\n';
  }
  output
         << "rank,candidate_id,dm,snr,period_seconds,period_ms,"
            "frequency_hz,acceleration_m_per_s2,jerk_m_per_s3,snap_m_per_s4,"
            "reference_time_seconds,motion_order,phase_bin,phase_bins,"
            "boxcar_width_bins,duty_cycle,members,dm_min,dm_max,"
            "dm_index_min,dm_index_max\n";
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (const CandidateRow& row : report.candidates) {
    output << row.rank << ',' << row.candidate_id << ',' << row.dm << ','
           << row.snr << ',' << row.period_seconds << ','
           << row.period_seconds * 1000.0 << ',' << row.frequency_hz << ','
           << row.acceleration_m_per_s2 << ',' << row.jerk_m_per_s3 << ','
           << row.snap_m_per_s4 << ',' << row.reference_time_seconds << ','
           << motion_order_name(row.motion_order) << ',';
    if (row.phase_bin) {
      output << *row.phase_bin;
    }
    output << ',' << row.phase_bins << ',' << row.boxcar_width_bins << ','
           << row.duty_cycle << ',' << row.member_count << ',' << row.dm_min
           << ',' << row.dm_max << ',' << row.dm_index_min << ','
           << row.dm_index_max << '\n';
  }
}

}  // namespace

OutputPaths make_output_paths(const Config& config,
                              const std::filesystem::path& input) {
  if (!config.candidate_output) {
    throw std::invalid_argument("candidate output path is not configured");
  }
  std::filesystem::path stem = *config.candidate_output;
  if (std::filesystem::is_directory(config.input)) {
    stem /= input.stem();
  }
  return OutputPaths{
      .candidates = stem.string() + ".cand",
      .report = stem.string() + ".out",
  };
}

std::vector<OutputPaths> plan_output_paths(
    const Config& config,
    std::span<const std::filesystem::path> inputs) {
  std::vector<OutputPaths> outputs;
  if (!config.candidate_output) {
    return outputs;
  }
  outputs.reserve(inputs.size());
  for (const auto& input : inputs) {
    outputs.push_back(make_output_paths(config, input));
  }
  return outputs;
}

void validate_output_plan(std::span<const OutputPaths> outputs,
                          bool overwrite) {
  std::set<std::filesystem::path> seen;
  for (const auto& paths : outputs) {
    for (const auto& path : {paths.candidates, paths.report}) {
      const auto normalized =
          std::filesystem::absolute(path).lexically_normal();
      if (!seen.insert(normalized).second) {
        throw std::invalid_argument("multiple inputs map to output file: " +
                                    path.string());
      }
      if (std::filesystem::is_directory(path)) {
        throw std::runtime_error("output path is a directory: " +
                                 path.string());
      }
      if (!overwrite && std::filesystem::exists(path)) {
        throw std::runtime_error("output file already exists: " +
                                 path.string());
      }
    }
  }
}

void render_human_report(std::ostream& output, const Config& config,
                         const Report& report, std::size_t limit) {
  write_human_report(output, config, report, limit);
}

void write_candidate_file(const std::filesystem::path& path,
                          const Report& report, bool overwrite) {
  write_atomically(path, overwrite,
                   [&](std::ostream& output) { write_csv(output, report); });
}

void write_report_file(const std::filesystem::path& path, const Config& config,
                       const Report& report, bool overwrite) {
  write_atomically(path, overwrite, [&](std::ostream& output) {
    write_human_report(output, config, report, 0);
  });
}

}  // namespace gaffa_search
