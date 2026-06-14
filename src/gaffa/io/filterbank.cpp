#include "gaffa/filterbank.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gaffa {
namespace {

template <typename T>
void read_exact(std::istream& input, T& value, const char* field_name) {
  static_assert(std::is_trivially_copyable_v<T>);
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) {
    throw std::runtime_error(std::string("Failed to read filterbank field: ") +
                             field_name);
  }
}

std::string read_string(std::istream& input) {
  int length = 0;
  read_exact(input, length, "string length");
  if (length <= 0 || length > 4096) {
    throw std::runtime_error("Invalid filterbank string length");
  }

  std::string value(static_cast<std::size_t>(length), '\0');
  input.read(value.data(), length);
  if (!input) {
    throw std::runtime_error("Failed to read filterbank string");
  }
  return value;
}

std::int64_t checked_file_size(const std::filesystem::path& path) {
  const auto size = std::filesystem::file_size(path);
  if (size > static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error("Filterbank file is too large");
  }
  return static_cast<std::int64_t>(size);
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char* what) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(std::string("Filterbank size overflow: ") + what);
  }
  return lhs * rhs;
}

std::size_t bytes_per_sample(int nbits) {
  switch (nbits) {
  case 8:
    return sizeof(std::uint8_t);
  case 16:
    return sizeof(std::uint16_t);
  case 32:
    return sizeof(float);
  default:
    throw std::runtime_error("Unsupported filterbank nbits");
  }
}

bool frequency_table_is_descending(const FilterbankHeader& header) {
  return header.frequency_table.size() >= 2 &&
         header.frequency_table.front() > header.frequency_table.back();
}

bool should_reverse_channels(const FilterbankHeader& header,
                             ChannelOrderPolicy policy) {
  if (policy == ChannelOrderPolicy::PreserveFileOrder) {
    return false;
  }
  if (header.uses_frequency_table) {
    return frequency_table_is_descending(header);
  }
  return header.foff < 0.0;
}

void populate_frequency_table(FilterbankHeader& header) {
  if (header.uses_frequency_table) {
    if (header.frequency_table.size() != static_cast<std::size_t>(header.nchans)) {
      throw std::runtime_error("Frequency table length does not match nchans");
    }
    return;
  }

  header.frequency_table.resize(static_cast<std::size_t>(header.nchans));
  for (int channel = 0; channel < header.nchans; ++channel) {
    header.frequency_table[static_cast<std::size_t>(channel)] =
        header.fch1 + static_cast<double>(channel) * header.foff;
  }
}

void normalize_frequency_metadata(FilterbankHeader& header,
                                  bool reversed_samples) {
  if (!reversed_samples) {
    return;
  }

  std::reverse(header.frequency_table.begin(), header.frequency_table.end());
  if (!header.frequency_table.empty()) {
    header.fch1 = header.frequency_table.front();
  }
  if (!header.uses_frequency_table) {
    header.foff = std::abs(header.foff);
  }
}

FilterbankHeader read_header(std::istream& input,
                             const std::filesystem::path& path) {
  FilterbankHeader header;
  bool expecting_rawdatafile = false;
  bool expecting_source_name = false;
  bool expecting_frequency_table = false;

  const std::string start = read_string(input);
  if (start != "HEADER_START") {
    throw std::runtime_error("Non-standard filterbank file: missing HEADER_START");
  }

  while (true) {
    const std::string key = read_string(input);
    if (key == "HEADER_END") {
      break;
    }

    if (expecting_rawdatafile) {
      header.rawdatafile = key;
      expecting_rawdatafile = false;
    } else if (expecting_source_name) {
      header.source_name = key;
      expecting_source_name = false;
    } else if (key == "rawdatafile") {
      expecting_rawdatafile = true;
    } else if (key == "source_name") {
      expecting_source_name = true;
    } else if (key == "FREQUENCY_START") {
      expecting_frequency_table = true;
      header.uses_frequency_table = true;
      header.frequency_table.clear();
    } else if (key == "FREQUENCY_END") {
      expecting_frequency_table = false;
      header.nchans = static_cast<int>(header.frequency_table.size());
    } else if (key == "fchannel") {
      if (!expecting_frequency_table) {
        throw std::runtime_error("fchannel outside FREQUENCY block");
      }
      double frequency = 0.0;
      read_exact(input, frequency, "fchannel");
      header.frequency_table.push_back(frequency);
      header.fch1 = 0.0;
      header.foff = 0.0;
      header.uses_frequency_table = true;
    } else if (key == "az_start") {
      read_exact(input, header.az_start, "az_start");
    } else if (key == "za_start") {
      read_exact(input, header.za_start, "za_start");
    } else if (key == "src_raj") {
      read_exact(input, header.src_raj, "src_raj");
    } else if (key == "src_dej") {
      read_exact(input, header.src_dej, "src_dej");
    } else if (key == "tstart") {
      read_exact(input, header.tstart, "tstart");
    } else if (key == "tsamp") {
      read_exact(input, header.tsamp, "tsamp");
    } else if (key == "period") {
      read_exact(input, header.period, "period");
    } else if (key == "fch1") {
      read_exact(input, header.fch1, "fch1");
    } else if (key == "foff") {
      read_exact(input, header.foff, "foff");
    } else if (key == "nchans") {
      read_exact(input, header.nchans, "nchans");
    } else if (key == "telescope_id") {
      read_exact(input, header.telescope_id, "telescope_id");
    } else if (key == "machine_id") {
      read_exact(input, header.machine_id, "machine_id");
    } else if (key == "data_type") {
      read_exact(input, header.data_type, "data_type");
    } else if (key == "ibeam") {
      read_exact(input, header.ibeam, "ibeam");
    } else if (key == "nbeams") {
      read_exact(input, header.nbeams, "nbeams");
    } else if (key == "nbits") {
      read_exact(input, header.nbits, "nbits");
    } else if (key == "barycentric") {
      read_exact(input, header.barycentric, "barycentric");
    } else if (key == "pulsarcentric") {
      read_exact(input, header.pulsarcentric, "pulsarcentric");
    } else if (key == "nbins") {
      read_exact(input, header.nbins, "nbins");
    } else if (key == "nsamples") {
      int ignored_nsamples = 0;
      read_exact(input, ignored_nsamples, "nsamples");
    } else if (key == "nifs") {
      read_exact(input, header.nifs, "nifs");
    } else if (key == "npuls") {
      read_exact(input, header.npuls, "npuls");
    } else if (key == "refdm") {
      read_exact(input, header.refdm, "refdm");
    } else {
      throw std::runtime_error("Unknown filterbank header field: " + key);
    }
  }

  const auto end_position = input.tellg();
  if (end_position < 0) {
    throw std::runtime_error("Failed to determine filterbank header size");
  }
  header.header_size = static_cast<std::int64_t>(end_position);

  if (header.nchans <= 0) {
    throw std::runtime_error("Invalid filterbank nchans");
  }
  if (header.nifs <= 0) {
    throw std::runtime_error("Invalid filterbank nifs");
  }

  const std::int64_t data_bytes = checked_file_size(path) - header.header_size;
  if (data_bytes < 0) {
    throw std::runtime_error("Filterbank header is larger than file");
  }

  const std::size_t row_bytes =
      checked_multiply(checked_multiply(static_cast<std::size_t>(header.nifs),
                                        static_cast<std::size_t>(header.nchans),
                                        "row elements"),
                       bytes_per_sample(header.nbits), "row bytes");
  if (row_bytes == 0 || data_bytes % static_cast<std::int64_t>(row_bytes) != 0) {
    throw std::runtime_error("Filterbank data size is not a whole number of rows");
  }
  header.nsamples = data_bytes / static_cast<std::int64_t>(row_bytes);
  populate_frequency_table(header);
  return header;
}

bool use_openmp(ReverseBackend backend, std::size_t rows,
                std::size_t threshold) {
  if (backend == ReverseBackend::CpuScalar) {
    return false;
  }
#ifdef _OPENMP
  if (backend == ReverseBackend::CpuOpenmp) {
    return true;
  }
  return backend == ReverseBackend::Auto && rows >= threshold;
#else
  (void)backend;
  (void)rows;
  (void)threshold;
  return false;
#endif
}

template <typename T>
void reverse_rows(const T* src, T* dst, std::size_t rows, std::size_t nifs,
                  std::size_t nchans, ReverseBackend backend,
                  std::size_t openmp_min_rows) {
  const bool parallel = use_openmp(backend, rows, openmp_min_rows);
#ifdef _OPENMP
#pragma omp parallel for if(parallel) schedule(static)
#endif
  for (std::int64_t row = 0; row < static_cast<std::int64_t>(rows); ++row) {
    for (std::size_t if_index = 0; if_index < nifs; ++if_index) {
      const T* row_src =
          src + (static_cast<std::size_t>(row) * nifs + if_index) * nchans;
      T* row_dst =
          dst + (static_cast<std::size_t>(row) * nifs + if_index) * nchans;
      for (std::size_t channel = 0; channel < nchans; ++channel) {
        row_dst[channel] = row_src[nchans - 1 - channel];
      }
    }
  }
  (void)parallel;
}

template <typename T>
std::vector<T> read_samples(std::istream& input, const FilterbankHeader& header,
                            const FilterbankReadOptions& options,
                            bool reverse_channels) {
  const std::size_t nsamples = static_cast<std::size_t>(header.nsamples);
  const std::size_t nifs = static_cast<std::size_t>(header.nifs);
  const std::size_t nchans = static_cast<std::size_t>(header.nchans);
  const std::size_t row_elems = checked_multiply(nifs, nchans, "row elements");
  const std::size_t total_elems =
      checked_multiply(nsamples, row_elems, "total elements");

  std::vector<T> samples(total_elems);
  if (total_elems == 0) {
    return samples;
  }

  if (!reverse_channels) {
    input.read(reinterpret_cast<char*>(samples.data()),
               static_cast<std::streamsize>(total_elems * sizeof(T)));
    if (!input) {
      throw std::runtime_error("Failed to read filterbank samples");
    }
    return samples;
  }

  const std::size_t row_bytes = checked_multiply(row_elems, sizeof(T), "row bytes");
  const std::size_t rows_per_chunk =
      std::max<std::size_t>(1, options.io_buffer_bytes / row_bytes);
  const std::size_t elems_per_chunk =
      checked_multiply(rows_per_chunk, row_elems, "chunk elements");
  std::vector<T> io_buffer(elems_per_chunk);

  std::size_t rows_read = 0;
  while (rows_read < nsamples) {
    const std::size_t rows_this_chunk =
        std::min(rows_per_chunk, nsamples - rows_read);
    const std::size_t elems_this_chunk = rows_this_chunk * row_elems;

    input.read(reinterpret_cast<char*>(io_buffer.data()),
               static_cast<std::streamsize>(elems_this_chunk * sizeof(T)));
    if (!input) {
      throw std::runtime_error("Failed to read filterbank samples");
    }

    reverse_rows(io_buffer.data(), samples.data() + rows_read * row_elems,
                 rows_this_chunk, nifs, nchans, options.reverse_backend,
                 options.openmp_min_rows);
    rows_read += rows_this_chunk;
  }

  return samples;
}

FilterbankSamples read_typed_samples(std::istream& input,
                                     const FilterbankHeader& header,
                                     const FilterbankReadOptions& options,
                                     bool reverse_channels) {
  switch (header.nbits) {
  case 8:
    return read_samples<std::uint8_t>(input, header, options, reverse_channels);
  case 16:
    return read_samples<std::uint16_t>(input, header, options, reverse_channels);
  case 32:
    return read_samples<float>(input, header, options, reverse_channels);
  default:
    throw std::runtime_error("Unsupported filterbank nbits");
  }
}

}  // namespace

FilterbankData read_filterbank(const std::filesystem::path& path,
                               const FilterbankReadOptions& options) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open filterbank file: " + path.string());
  }

  FilterbankHeader header = read_header(input, path);
  const bool reverse_channels =
      should_reverse_channels(header, options.channel_order);
  FilterbankSamples samples =
      read_typed_samples(input, header, options, reverse_channels);
  normalize_frequency_metadata(header, reverse_channels);
  return FilterbankData{std::move(header), std::move(samples)};
}

}  // namespace gaffa
