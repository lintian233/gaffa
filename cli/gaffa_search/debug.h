#pragma once

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <thread>

namespace gaffa_search::detail {

using DebugClock = std::chrono::steady_clock;

inline bool debug_enabled() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("GAFFA_DEBUG");
    return value != nullptr && *value != '\0' && *value != '0';
  }();
  return enabled;
}

inline double debug_seconds(DebugClock::time_point begin) noexcept {
  return std::chrono::duration<double>(DebugClock::now() - begin).count();
}

inline void debug_print(std::string_view message) {
  static const auto process_begin = DebugClock::now();
  static std::mutex output_mutex;
  const std::lock_guard lock(output_mutex);
  std::cerr << "[gaffa-debug +" << debug_seconds(process_begin) << "s t="
            << std::this_thread::get_id() << "] " << message << '\n';
}

}  // namespace gaffa_search::detail

#define DEBUGPRINT(expression)                                                \
  do {                                                                         \
    if (::gaffa_search::detail::debug_enabled()) {                             \
      std::ostringstream gaffa_debug_stream;                                  \
      gaffa_debug_stream << expression;                                       \
      ::gaffa_search::detail::debug_print(gaffa_debug_stream.str());          \
    }                                                                          \
  } while (false)

