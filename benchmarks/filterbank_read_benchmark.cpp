#include "gaffa/filterbank.h"
#include "gaffa/filterbank_legacy.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace {

struct Args {
  std::string mode;
  std::filesystem::path path;
  int iterations = 1;
};

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <legacy|eager|eager-preserve> <filterbank.fil> [iterations]\n";
}

Args parse_args(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    usage(argv[0]);
    throw std::invalid_argument("invalid argument count");
  }

  Args args;
  args.mode = argv[1];
  args.path = argv[2];
  if (argc == 4) {
    args.iterations = std::stoi(argv[3]);
  }
  if (args.iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  if (args.mode != "legacy" && args.mode != "eager" &&
      args.mode != "eager-preserve") {
    usage(argv[0]);
    throw std::invalid_argument("unknown benchmark mode");
  }
  return args;
}

std::size_t sample_count(const gaffa::FilterbankHeader& header) {
  return static_cast<std::size_t>(header.nsamples) *
         static_cast<std::size_t>(header.nifs) *
         static_cast<std::size_t>(header.nchans);
}

std::size_t sample_bytes(const gaffa::FilterbankData& data) {
  return std::visit(
      [](const auto& samples) { return samples.size() * sizeof(samples[0]); },
      data.samples);
}

template <typename Func>
double time_once(Func&& func) {
  const auto start = std::chrono::steady_clock::now();
  func();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

void run_legacy(const std::filesystem::path& path, int iterations) {
  for (int index = 0; index < iterations; ++index) {
    std::unique_ptr<Filterbank> filterbank;
    const double seconds =
        time_once([&] { filterbank = std::make_unique<Filterbank>(path.string()); });

    const auto bytes = static_cast<double>(filterbank->nsamples) *
                       static_cast<double>(filterbank->nifs) *
                       static_cast<double>(filterbank->nchans) *
                       (static_cast<double>(filterbank->nbits) / 8.0);
    std::cout << "mode=legacy"
              << " iteration=" << index
              << " seconds=" << seconds
              << " mib_per_sec=" << (bytes / 1024.0 / 1024.0 / seconds)
              << " nsamples=" << filterbank->nsamples
              << " nifs=" << filterbank->nifs
              << " nchans=" << filterbank->nchans
              << " nbits=" << filterbank->nbits
              << " foff=" << filterbank->foff
              << '\n';
  }
}

void run_eager(const std::filesystem::path& path, int iterations, bool preserve) {
  gaffa::FilterbankReadOptions options;
  options.channel_order = preserve ? gaffa::ChannelOrderPolicy::PreserveFileOrder
                                   : gaffa::ChannelOrderPolicy::FrequencyAscending;
  for (int index = 0; index < iterations; ++index) {
    gaffa::FilterbankData data;
    const double seconds = time_once([&] { data = gaffa::read_filterbank(path, options); });
    const auto bytes = static_cast<double>(sample_bytes(data));
    std::cout << "mode=" << (preserve ? "eager-preserve" : "eager")
              << " iteration=" << index
              << " seconds=" << seconds
              << " mib_per_sec=" << (bytes / 1024.0 / 1024.0 / seconds)
              << " nsamples=" << data.header.nsamples
              << " nifs=" << data.header.nifs
              << " nchans=" << data.header.nchans
              << " nbits=" << data.header.nbits
              << " sample_count=" << sample_count(data.header)
              << " foff=" << data.header.foff
              << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (args.mode == "legacy") {
      run_legacy(args.path, args.iterations);
    } else {
      run_eager(args.path, args.iterations, args.mode == "eager-preserve");
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
