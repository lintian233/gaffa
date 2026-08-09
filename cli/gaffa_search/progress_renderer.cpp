#include "progress_renderer.h"

#include "config.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string_view>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace gaffa_search {
namespace {

const char* stage_name(ProgressStage stage) noexcept {
  switch (stage) {
    case ProgressStage::Idle:
      return "";
    case ProgressStage::Read:
      return "read";
    case ProgressStage::Dedispersion:
      return "dedispersion";
    case ProgressStage::Search:
      return "search";
    case ProgressStage::Candidate:
      return "candidate";
    case ProgressStage::Complete:
      return "complete";
    case ProgressStage::Failed:
      return "";
  }
  return "unknown";
}

std::string fixed(double value, int precision) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string progress_bar(std::size_t completed, std::size_t total,
                         std::size_t width = 20) {
  if (total == 0) {
    return "[working]";
  }
  const std::size_t filled =
      std::min(width, completed * width / std::max<std::size_t>(1, total));
  return "[" + std::string(filled, '#') +
         std::string(width - filled, '-') + "]";
}

std::string fit_line(std::string text, std::size_t width) {
  if (width == 0 || text.size() <= width) {
    return text;
  }
  if (width <= 3) {
    text.resize(width);
    return text;
  }
  text.resize(width - 3);
  text += "...";
  return text;
}

std::size_t terminal_width() noexcept {
#ifndef _WIN32
  struct winsize size = {};
  if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col != 0) {
    return size.ws_col;
  }
#endif
  return 80;
}

}  // namespace

ProgressFrame format_progress_frame(const ProgressSnapshot& snapshot,
                                    std::size_t width) {
  std::ostringstream header;
  if (snapshot.context.file_count != 0) {
    header << "[" << snapshot.context.file_index << "/"
           << snapshot.context.file_count << "] ";
  }
  if (!snapshot.context.input.empty()) {
    header << snapshot.context.input.string();
  }

  std::ostringstream status;
  status << stage_name(snapshot.stage);
  switch (snapshot.stage) {
    case ProgressStage::Read:
      if (snapshot.observation_nsamples != 0) {
        status << " done samples=" << snapshot.observation_nsamples
               << " duration=" << fixed(snapshot.duration_seconds, 3) << " s";
      } else {
        status << " working";
      }
      break;
    case ProgressStage::Dedispersion:
      status << " "
             << progress_bar(snapshot.dm_ranges_completed,
                             snapshot.dm_ranges_total)
             << " " << snapshot.dm_ranges_completed << "/"
             << snapshot.dm_ranges_total << " DM";
      break;
    case ProgressStage::Search:
      status << " "
             << progress_bar(snapshot.search_runs_completed,
                             snapshot.search_runs_total)
             << " " << snapshot.search_runs_completed << "/"
             << snapshot.search_runs_total << " "
             << backend_name(snapshot.context.backend) << " range="
             << snapshot.context.search_range_index << " tile="
             << snapshot.active_units_completed << "/"
             << snapshot.active_units_total;
      break;
    case ProgressStage::Candidate:
      status << " working raw_peaks=" << snapshot.raw_peaks;
      break;
    case ProgressStage::Complete:
      status << " done groups=" << snapshot.summary.candidate_groups
             << " harmonics=" << snapshot.summary.harmonic_relations
             << " final=" << snapshot.summary.final_candidates;
      break;
    case ProgressStage::Failed:
      status << " failed";
      break;
    case ProgressStage::Idle:
      status << " waiting";
      break;
  }

  const std::size_t status_width = width > 2 ? width - 2 : width;
  return ProgressFrame{
      .header = fit_line(header.str(), width),
      .status = fit_line(status.str(), status_width),
  };
}

ProgressRenderer::ProgressRenderer(std::ostream& output, ProgressMode mode)
    : output_(output), mode_(mode) {}

ProgressRenderer::~ProgressRenderer() noexcept { stop(); }

bool ProgressRenderer::interactive_mode() const noexcept {
  if (mode_ == ProgressMode::Interactive) {
    return true;
  }
  if (mode_ == ProgressMode::Plain) {
    return false;
  }
#ifndef _WIN32
  return ::isatty(STDERR_FILENO) != 0;
#else
  return false;
#endif
}

void ProgressRenderer::start(ProgressTracker& tracker) {
  stop();
  tracker_ = &tracker;
  current_file_index_ = 0;
  current_input_.clear();
  header_rendered_ = false;
  status_visible_ = false;
  running_.store(true, std::memory_order_release);
  thread_ = std::thread(&ProgressRenderer::render_loop, this);
}

namespace {

void clear_status_line(std::ostream& output) { output << "\r\033[2K"; }

}  // namespace

void ProgressRenderer::stop() noexcept {
  running_.store(false, std::memory_order_release);
  wait_condition_.notify_all();
  if (thread_.joinable()) {
    try {
      thread_.join();
    } catch (...) {
      // A renderer failure must never replace the search exception.
    }
  }
  if (interactive_mode() && status_visible_) {
    try {
      std::lock_guard lock(output_mutex_);
      clear_status_line(output_);
      output_ << '\n';
      output_.flush();
    } catch (...) {
    }
  }
  tracker_ = nullptr;
  status_visible_ = false;
}

void ProgressRenderer::render_once(const ProgressSnapshot& snapshot,
                                   bool force) {
  const bool interactive = interactive_mode();
  const ProgressFrame frame =
      format_progress_frame(snapshot, terminal_width());
  if (!interactive && !force) {
    return;
  }
  std::lock_guard lock(output_mutex_);
  const bool file_changed =
      !header_rendered_ || current_file_index_ != snapshot.context.file_index ||
      current_input_ != snapshot.context.input;
  if (file_changed) {
    if (interactive && status_visible_) {
      clear_status_line(output_);
      output_ << '\n';
    }
    if (!frame.header.empty()) {
      output_ << frame.header << '\n';
    }
    current_file_index_ = snapshot.context.file_index;
    current_input_ = snapshot.context.input;
    header_rendered_ = true;
    status_visible_ = false;
  }
  if (interactive) {
    clear_status_line(output_);
    output_ << "  " << frame.status;
    output_.flush();
    status_visible_ = true;
  } else {
    output_ << "  " << frame.status << '\n';
    output_.flush();
  }
}

void ProgressRenderer::render_loop() noexcept {
  if (tracker_ == nullptr) {
    return;
  }
  ProgressSnapshot previous;
  bool have_previous = false;
  const bool interactive = interactive_mode();
  while (running_.load(std::memory_order_acquire)) {
    const ProgressSnapshot current = tracker_->snapshot();
    const bool stage_changed =
        !have_previous || current.stage != previous.stage ||
        current.context.file_index != previous.context.file_index ||
        current.context.input != previous.context.input ||
        current.context.search_range_index !=
            previous.context.search_range_index ||
        current.context.backend != previous.context.backend;
    const bool run_completed =
        !have_previous || current.search_runs_completed !=
                              previous.search_runs_completed;
    const bool dm_completed =
        !have_previous || current.dm_ranges_completed !=
                              previous.dm_ranges_completed;
    if (current.stage != ProgressStage::Idle &&
        (interactive || stage_changed || run_completed || dm_completed)) {
      render_once(current, true);
    }
    previous = current;
    have_previous = true;

    std::unique_lock lock(wait_mutex_);
    wait_condition_.wait_for(lock, std::chrono::milliseconds(200), [&] {
      return !running_.load(std::memory_order_acquire);
    });
  }
  if (tracker_ != nullptr) {
    render_once(tracker_->snapshot(), true);
  }
}

ProgressSession::ProgressSession(ProgressRenderer& renderer,
                                 ProgressTracker& tracker)
    : renderer_(renderer) {
  renderer_.start(tracker);
}

ProgressSession::~ProgressSession() noexcept { renderer_.stop(); }

}  // namespace gaffa_search
