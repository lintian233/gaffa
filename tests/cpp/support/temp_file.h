#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

namespace gaffa::test {

class TempFile {
public:
  explicit TempFile(const std::string& prefix)
      : path_(std::filesystem::temp_directory_path() / unique_name(prefix)) {}

  TempFile(TempFile&& other) noexcept : path_(std::move(other.path_)) {
    other.path_.clear();
  }

  TempFile& operator=(TempFile&& other) noexcept {
    if (this != &other) {
      cleanup();
      path_ = std::move(other.path_);
      other.path_.clear();
    }
    return *this;
  }

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  ~TempFile() { cleanup(); }

  const std::filesystem::path& path() const { return path_; }

private:
  static std::string unique_name(const std::string& prefix) {
    static std::atomic<std::uint64_t> counter{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return prefix + "_" + std::to_string(now) + "_" +
           std::to_string(counter.fetch_add(1)) + ".fil";
  }

  void cleanup() {
    if (!path_.empty()) {
      std::error_code error;
      std::filesystem::remove(path_, error);
    }
  }

  std::filesystem::path path_;
};

}  // namespace gaffa::test
