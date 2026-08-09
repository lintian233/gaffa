#pragma once

#include "progress.h"

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <iosfwd>
#include <mutex>
#include <string>
#include <thread>

namespace gaffa_search {

enum class ProgressMode {
  Auto,
  Interactive,
  Plain,
};

struct ProgressFrame {
  std::string header;
  std::string status;
};

ProgressFrame format_progress_frame(const ProgressSnapshot& snapshot,
                                     std::size_t terminal_width = 80);

class ProgressRenderer {
 public:
  ProgressRenderer(std::ostream& output, ProgressMode mode = ProgressMode::Auto);
  ~ProgressRenderer() noexcept;

  ProgressRenderer(const ProgressRenderer&) = delete;
  ProgressRenderer& operator=(const ProgressRenderer&) = delete;

  void start(ProgressTracker& tracker);
  void stop() noexcept;

 private:
  void render_loop() noexcept;
  void render_once(const ProgressSnapshot& snapshot, bool force);
  [[nodiscard]] bool interactive_mode() const noexcept;

  std::ostream& output_;
  ProgressMode mode_;
  ProgressTracker* tracker_ = nullptr;
  std::atomic<bool> running_{false};
  std::mutex wait_mutex_;
  std::condition_variable wait_condition_;
  std::thread thread_;
  std::mutex output_mutex_;
  std::size_t current_file_index_ = 0;
  std::filesystem::path current_input_;
  bool header_rendered_ = false;
  bool status_visible_ = false;
};

class ProgressSession {
 public:
  ProgressSession(ProgressRenderer& renderer, ProgressTracker& tracker);
  ~ProgressSession() noexcept;

  ProgressSession(const ProgressSession&) = delete;
  ProgressSession& operator=(const ProgressSession&) = delete;

 private:
  ProgressRenderer& renderer_;
};

}  // namespace gaffa_search
