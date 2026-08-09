#include "progress.h"
#include "progress_renderer.h"

#include <gtest/gtest.h>

#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

TEST(GaffaSearchProgress, TracksPipelineLifecycle) {
  gaffa_search::ProgressTracker tracker;
  tracker.begin_file("observation.fil", 1, 2);
  tracker.begin_read();
  tracker.finish_read(1024, 0.001, 1.024);

  tracker.begin_work(2, 4);
  tracker.begin_dm_range(0);
  EXPECT_EQ(tracker.snapshot().stage, gaffa_search::ProgressStage::Dedispersion);
  tracker.complete_dm_range();
  tracker.begin_search_phase();
  tracker.begin_search_run(0, 3, gaffa_search::Backend::NativeCuda, 2);
  tracker.complete_search_units(2);
  tracker.finish_search_run(17);

  auto snapshot = tracker.snapshot();
  EXPECT_EQ(snapshot.dm_ranges_completed, 1U);
  EXPECT_EQ(snapshot.dm_ranges_total, 2U);
  EXPECT_EQ(snapshot.search_runs_completed, 1U);
  EXPECT_EQ(snapshot.search_runs_total, 4U);
  EXPECT_EQ(snapshot.active_units_completed, 2U);
  EXPECT_EQ(snapshot.active_units_total, 2U);
  EXPECT_EQ(snapshot.raw_peaks, 17U);
  EXPECT_EQ(snapshot.context.backend, gaffa_search::Backend::NativeCuda);

  tracker.begin_candidate(snapshot.raw_peaks);
  tracker.finish_candidate(gaffa_search::ProgressSummary{
      .elapsed_seconds = 0.25,
      .raw_peaks = 17,
      .candidate_groups = 4,
      .harmonic_relations = 2,
      .final_candidates = 3,
  });
  tracker.finish_file();

  snapshot = tracker.snapshot();
  EXPECT_EQ(snapshot.stage, gaffa_search::ProgressStage::Complete);
  EXPECT_EQ(snapshot.summary.final_candidates, 3U);
  EXPECT_EQ(snapshot.context.input, "observation.fil");
}

TEST(GaffaSearchProgress, CountsConcurrentSearchUnits) {
  gaffa_search::ProgressTracker tracker;
  tracker.begin_file("observation.fil", 1, 1);
  tracker.begin_work(1, 1);
  tracker.begin_dm_range(0);
  tracker.complete_dm_range();
  tracker.begin_search_phase();
  tracker.begin_search_run(0, 0, gaffa_search::Backend::NativeCuda, 8000);

  std::vector<std::thread> workers;
  workers.reserve(8);
  for (std::size_t index = 0; index < 8; ++index) {
    workers.emplace_back([&tracker] {
      for (std::size_t count = 0; count < 1000; ++count) {
        tracker.complete_search_units();
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }

  const auto snapshot = tracker.snapshot();
  EXPECT_EQ(snapshot.active_units_completed, 8000U);
  EXPECT_EQ(snapshot.active_units_total, 8000U);
}

TEST(GaffaSearchProgress, FormatsACompactFrame) {
  const gaffa_search::ProgressSnapshot snapshot{
      .stage = gaffa_search::ProgressStage::Search,
      .context = gaffa_search::ProgressContext{
          .input = "observation.fil",
          .file_index = 1,
          .file_count = 1,
          .dm_range_index = 2,
          .search_range_index = 3,
          .backend = gaffa_search::Backend::NativeCuda,
      },
      .search_runs_completed = 3,
      .search_runs_total = 8,
      .active_units_completed = 4,
      .active_units_total = 8,
  };

  const gaffa_search::ProgressFrame frame =
      gaffa_search::format_progress_frame(snapshot, 80);
  EXPECT_NE(frame.header.find("observation.fil"), std::string::npos);
  EXPECT_NE(frame.status.find("search"), std::string::npos);
  EXPECT_NE(frame.status.find("native-cuda"), std::string::npos);
  EXPECT_NE(frame.status.find("tile=4/8"), std::string::npos);
  EXPECT_NE(frame.status.find("[####"), std::string::npos);
  EXPECT_EQ(frame.header.find('\n'), std::string::npos);
  EXPECT_EQ(frame.status.find('\n'), std::string::npos);
  EXPECT_LE(frame.header.size(), 80U);
  EXPECT_LE(frame.status.size() + 2, 80U);
}

TEST(GaffaSearchProgress, TruncatesLongFrameLines) {
  gaffa_search::ProgressSnapshot snapshot{
      .stage = gaffa_search::ProgressStage::Search,
      .context = gaffa_search::ProgressContext{
          .input = "/observations/with/a/very/long/filterbank/path/observation.fil",
          .file_index = 1,
          .file_count = 1,
          .search_range_index = 3,
          .backend = gaffa_search::Backend::LokiCuda,
      },
      .search_runs_completed = 3,
      .search_runs_total = 8,
      .active_units_completed = 4,
      .active_units_total = 8,
  };

  const gaffa_search::ProgressFrame frame =
      gaffa_search::format_progress_frame(snapshot, 40);
  EXPECT_LE(frame.header.size(), 40U);
  EXPECT_LE(frame.status.size() + 2, 40U);
  EXPECT_NE(frame.header.find("..."), std::string::npos);
}

TEST(GaffaSearchProgress, RendererStopsWithoutThrowing) {
  gaffa_search::ProgressTracker tracker;
  tracker.begin_file("observation.fil", 1, 1);
  std::ostringstream output;
  {
    gaffa_search::ProgressRenderer renderer(
        output, gaffa_search::ProgressMode::Plain);
    gaffa_search::ProgressSession session(renderer, tracker);
    tracker.begin_read();
    tracker.finish_read(16, 0.1, 1.6);
    tracker.finish_file();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_EQ(output.str().find('\r'), std::string::npos);
}

TEST(GaffaSearchProgress, InteractiveRendererPrintsFileHeaderOnce) {
  gaffa_search::ProgressTracker tracker;
  tracker.begin_file("observation.fil", 1, 1);
  std::ostringstream output;
  {
    gaffa_search::ProgressRenderer renderer(
        output, gaffa_search::ProgressMode::Interactive);
    gaffa_search::ProgressSession session(renderer, tracker);
    tracker.begin_read();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    tracker.finish_file();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  const std::string text = output.str();
  std::size_t occurrences = 0;
  std::size_t position = 0;
  while ((position = text.find("observation.fil", position)) !=
         std::string::npos) {
    ++occurrences;
    position += std::string("observation.fil").size();
  }
  EXPECT_EQ(occurrences, 1U);
  EXPECT_NE(text.find("\033[2K"), std::string::npos);
  EXPECT_NE(text.find("read"), std::string::npos);
  EXPECT_NE(text.find("complete"), std::string::npos);
}

}  // namespace
