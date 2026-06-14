#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace gaffa {

inline constexpr std::size_t default_filterbank_io_buffer_bytes =
    64ULL * 1024ULL * 1024ULL;

enum class ChannelOrderPolicy {
  FrequencyAscending,
  PreserveFileOrder,
};

enum class ReverseBackend {
  Auto,
  CpuScalar,
  CpuOpenmp,
};

struct FilterbankReadOptions {
  std::size_t io_buffer_bytes = default_filterbank_io_buffer_bytes;
  ChannelOrderPolicy channel_order = ChannelOrderPolicy::FrequencyAscending;
  ReverseBackend reverse_backend = ReverseBackend::Auto;
  std::size_t openmp_min_rows = 4096;
};

struct FilterbankHeader {
  std::int64_t header_size = 0;
  std::int64_t nsamples = 0;

  int telescope_id = 0;
  int machine_id = 0;
  int data_type = 1;
  int barycentric = 0;
  int pulsarcentric = 0;
  int ibeam = 0;
  int nbeams = 0;
  int npuls = 0;
  int nbins = 0;
  int nbits = 0;
  int nifs = 1;
  int nchans = 0;

  double az_start = 0.0;
  double za_start = 0.0;
  double src_raj = 0.0;
  double src_dej = 0.0;
  double tstart = 0.0;
  double tsamp = 0.0;
  double fch1 = 0.0;
  double foff = 0.0;
  double refdm = 0.0;
  double period = 0.0;

  std::string rawdatafile;
  std::string source_name;
  std::vector<double> frequency_table;
  bool uses_frequency_table = false;
};

using FilterbankSamples =
    std::variant<std::vector<std::uint8_t>, std::vector<std::uint16_t>,
                 std::vector<float>>;

struct FilterbankData {
  FilterbankHeader header;
  FilterbankSamples samples;
};

FilterbankData read_filterbank(
    const std::filesystem::path& path,
    const FilterbankReadOptions& options = FilterbankReadOptions{});

}  // namespace gaffa
