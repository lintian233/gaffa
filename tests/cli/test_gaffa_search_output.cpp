#include "output.h"
#include "report.h"

#include <gtest/gtest.h>

#include "gaffa/candidate.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

gaffa::DmPeak make_peak(double dm, std::size_t dm_index, float snr,
                        double frequency) {
  return gaffa::DmPeak{
      .dm = dm,
      .dm_index = dm_index,
      .peak = gaffa::PeriodicPeak{
          .motion = gaffa::PeriodicMotion{
              .frequency_hz = frequency,
          },
          .phase_bin = 4,
          .phase_bins = 256,
          .boxcar_width_bins = 8,
          .duty_cycle = 8.0 / 256.0,
          .snr = snr,
      },
  };
}

gaffa_search::FileResult make_result() {
  const gaffa::DmPeak best = make_peak(101.5, 3, 12.5F, 2.0);
  gaffa::CandidateSet candidate_set;
  candidate_set.members = {best};
  candidate_set.candidates = {gaffa::Candidate{
      .best = best,
      .member_begin = 0,
      .member_count = 1,
      .extent = gaffa::CandidateExtent{
          .dm_index_min = 3,
          .dm_index_max = 3,
          .frequency_hz = {.minimum = 2.0, .maximum = 2.0},
      },
  }};

  gaffa_search::FileResult result;
  result.observation_nsamples = 1024;
  result.tsamp_seconds = 1.0 / 1024.0;
  result.observation_seconds = 1.0;
  result.raw_peak_count = 1;
  result.candidates = gaffa::CandidateResult{
      .candidate_set = std::move(candidate_set),
      .selected = {0},
  };
  result.input = "observation.fil";
  result.timing = {.read_seconds = 1.0,
                   .search_seconds = 2.0,
                   .candidate_seconds = 0.5,
                   .total_seconds = 3.5};
  result.search_runs = {gaffa_search::SearchRunInfo{
      .dm_range_id = 0,
      .search_range_id = 0,
      .backend = gaffa_search::Backend::NativeCpu,
      .coordinate = gaffa_search::SearchCoordinate{
          .start_time_seconds = 0.0,
          .valid_duration_seconds = 1.0,
          .tsamp_seconds = 1.0 / 1024.0,
          .source_nsamples = 1024,
          .prepared_nsamples = 1024,
          .padded = false,
      },
      .raw_peak_count = 1,
  }};
  return result;
}

std::filesystem::path temporary_directory() {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path = std::filesystem::temp_directory_path() /
                    ("gaffa-search-output-" + std::to_string(stamp));
  std::filesystem::create_directories(path);
  return path;
}

}  // namespace

TEST(GaffaSearchOutput, BuildsFinalCandidateRows) {
  const gaffa_search::Report report =
      gaffa_search::make_report(make_result());

  ASSERT_EQ(report.candidates.size(), 1U);
  EXPECT_EQ(report.candidates.front().rank, 1U);
  EXPECT_DOUBLE_EQ(report.candidates.front().period_seconds, 0.5);
  EXPECT_DOUBLE_EQ(report.candidates.front().dm_min, 101.5);
  EXPECT_EQ(report.candidates.front().member_count, 1U);
}

TEST(GaffaSearchOutput, StdoutLimitDoesNotChangeReport) {
  const gaffa_search::Report report =
      gaffa_search::make_report(make_result());
  gaffa_search::Config config;
  std::ostringstream output;

  gaffa_search::render_human_report(output, config, report, 1);
  EXPECT_NE(output.str().find("rank  DM        SNR"), std::string::npos);
  EXPECT_NE(output.str().find("2.00000"), std::string::npos);
  EXPECT_EQ(output.str().find("raw_peak"), std::string::npos);
}

TEST(GaffaSearchOutput, RendersEffectiveSearchWindowsAndRanges) {
  const gaffa_search::Report report =
      gaffa_search::make_report(make_result());
  gaffa_search::Config config;
  config.dm_ranges = {
      gaffa_search::DmRangeConfig{.dm_low = 100.0, .dm_step = 0.5, .ndm = 4},
      gaffa_search::DmRangeConfig{.dm_low = 200.0, .dm_step = 1.0, .ndm = 2},
  };
  config.native_cuda_devices = {0, 1};
  config.loki_cuda_devices = {2};
  config.search_ranges = {
      gaffa_search::SearchRangeConfig{
          .id = 0,
          .backend = gaffa_search::Backend::NativeCuda,
          .period_min = 0.018,
          .period_max = 1.0,
          .bins_min = 180,
          .bins_max = 256,
          .window_mode = gaffa_search::WindowMode::Truncate,
      },
      gaffa_search::SearchRangeConfig{
          .id = 1,
          .backend = gaffa_search::Backend::LokiCuda,
          .period_min = 0.020,
          .period_max = 2.0,
          .bins_min = 128,
          .bins_max = 256,
          .window_mode = gaffa_search::WindowMode::ZeroPad,
          .motion = gaffa_search::MotionRangeConfig{
              .accel = gaffa::ValueRange{.minimum = -5.0, .maximum = 5.0},
              .jerk = gaffa::ValueRange{.minimum = -0.2, .maximum = 0.2},
          },
      },
  };

  std::ostringstream output;
  gaffa_search::render_human_report(output, config, report, 1);
  const std::string text = output.str();
  EXPECT_NE(text.find("DM ranges (2)"), std::string::npos);
  EXPECT_NE(text.find("[0] dm=[100.000,101.500] step=0.500 trials=4"),
            std::string::npos);
  EXPECT_NE(text.find("Search ranges (2)"), std::string::npos);
  EXPECT_NE(text.find("backend=native-cuda devices=[0,1]"),
            std::string::npos);
  EXPECT_NE(text.find("period=[0.018000,1.000000] s bins=[180,256] "
                     "motion=frequency-only window=original"),
            std::string::npos);
  EXPECT_NE(text.find("backend=loki-cuda devices=[2]"),
            std::string::npos);
  EXPECT_NE(text.find("window=zero-pad"), std::string::npos);
  EXPECT_NE(text.find("motion=accel[-5.000000,5.000000] "
                     "jerk[-0.200000,0.200000]"),
            std::string::npos);
}

TEST(GaffaSearchOutput, WritesCompleteCandidateAndReportFiles) {
  const auto directory = temporary_directory();
  const auto candidate_path = directory / "result.cand";
  const auto report_path = directory / "result.out";
  const gaffa_search::Report report =
      gaffa_search::make_report(make_result());
  gaffa_search::Config config;

  gaffa_search::write_candidate_file(candidate_path, report, false);
  gaffa_search::write_report_file(report_path, config, report, false);

  std::ifstream candidate_input(candidate_path);
  const std::string candidate_text((std::istreambuf_iterator<char>(candidate_input)),
                                   std::istreambuf_iterator<char>());
  EXPECT_NE(candidate_text.find("rank,candidate_id,dm"),
            std::string::npos);
  EXPECT_NE(candidate_text.find("101.5"), std::string::npos);

  std::ifstream report_input(report_path);
  const std::string report_text((std::istreambuf_iterator<char>(report_input)),
                                std::istreambuf_iterator<char>());
  EXPECT_NE(report_text.find("GAFFA SEARCH\n============"),
            std::string::npos);
  EXPECT_NE(report_text.find("showing            1 of 1"),
            std::string::npos);

  std::filesystem::remove_all(directory);
}
