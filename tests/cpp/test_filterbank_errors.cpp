#include "gaffa/filterbank.h"
#include "support/temp_file.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using gaffa::test::TempFile;

template <typename T>
void write_value(std::ofstream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_string(std::ofstream& output, const std::string& value) {
  const int length = static_cast<int>(value.size());
  write_value(output, length);
  output.write(value.data(), length);
}

template <typename T>
void write_field(std::ofstream& output, const std::string& name,
                 const T& value) {
  write_string(output, name);
  write_value(output, value);
}

struct FixtureHeader {
  int nbits = 8;
  int nifs = 1;
  int nchans = 4;
  double fch1 = 1000.0;
  double foff = 1.0;
};

void write_header(std::ofstream& output, const FixtureHeader& header) {
  write_string(output, "HEADER_START");
  write_string(output, "source_name");
  write_string(output, "test-source");
  write_field(output, "nifs", header.nifs);
  write_field(output, "nchans", header.nchans);
  write_field(output, "nbits", header.nbits);
  write_field(output, "fch1", header.fch1);
  write_field(output, "foff", header.foff);
  write_string(output, "HEADER_END");
}

template <typename T>
TempFile write_fixture(const std::string& prefix, const FixtureHeader& header,
                       const std::vector<T>& samples) {
  TempFile file(prefix);
  std::ofstream output(file.path(), std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to create test fixture");
  }

  write_header(output, header);
  output.write(reinterpret_cast<const char*>(samples.data()),
               static_cast<std::streamsize>(samples.size() * sizeof(T)));
  return file;
}

template <typename Func>
TempFile write_raw_fixture(const std::string& prefix, Func&& write_body) {
  TempFile file(prefix);
  std::ofstream output(file.path(), std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to create raw test fixture");
  }
  write_body(output);
  return file;
}

void expect_read_throws(const std::filesystem::path& path,
                        const std::string& message_fragment) {
  try {
    (void)gaffa::read_filterbank(path);
    FAIL() << "Expected read_filterbank to throw";
  } catch (const std::exception& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find(message_fragment), std::string::npos) << message;
  }
}

}  // namespace

TEST(FilterbankErrors, RejectsMissingFile) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      "gaffa_filterbank_missing_input.fil";
  std::error_code error;
  std::filesystem::remove(path, error);

  expect_read_throws(path, "Failed to open filterbank file");
}

TEST(FilterbankErrors, RejectsMissingHeaderStart) {
  const auto file = write_raw_fixture("gaffa_filterbank_bad_start",
                                      [](std::ofstream& output) {
                                        write_string(output, "NOT_HEADER");
                                      });

  expect_read_throws(file.path(), "missing HEADER_START");
}

TEST(FilterbankErrors, RejectsInvalidStringLength) {
  const auto file = write_raw_fixture("gaffa_filterbank_bad_string_length",
                                      [](std::ofstream& output) {
                                        const int length = 0;
                                        write_value(output, length);
                                      });

  expect_read_throws(file.path(), "Invalid filterbank string length");
}

TEST(FilterbankErrors, RejectsTruncatedString) {
  const auto file = write_raw_fixture("gaffa_filterbank_truncated_string",
                                      [](std::ofstream& output) {
                                        const int length = 12;
                                        write_value(output, length);
                                        output.write("HEADER", 6);
                                      });

  expect_read_throws(file.path(), "Failed to read filterbank string");
}

TEST(FilterbankErrors, RejectsUnknownHeaderField) {
  const auto file = write_raw_fixture("gaffa_filterbank_unknown_field",
                                      [](std::ofstream& output) {
                                        write_string(output, "HEADER_START");
                                        write_string(output, "unknown_key");
                                      });

  expect_read_throws(file.path(), "Unknown filterbank header field");
}

TEST(FilterbankErrors, RejectsFchannelOutsideFrequencyBlock) {
  const auto file = write_raw_fixture("gaffa_filterbank_fchannel_outside",
                                      [](std::ofstream& output) {
                                        write_string(output, "HEADER_START");
                                        write_string(output, "fchannel");
                                      });

  expect_read_throws(file.path(), "fchannel outside FREQUENCY block");
}

TEST(FilterbankErrors, RejectsInvalidNchans) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_invalid_nchans",
      FixtureHeader{.nbits = 8, .nifs = 1, .nchans = 0}, {});

  expect_read_throws(file.path(), "Invalid filterbank nchans");
}

TEST(FilterbankErrors, RejectsInvalidNifs) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_invalid_nifs",
      FixtureHeader{.nbits = 8, .nifs = 0, .nchans = 4}, {});

  expect_read_throws(file.path(), "Invalid filterbank nifs");
}

TEST(FilterbankErrors, RejectsUnsupportedNbits) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_unsupported_nbits",
      FixtureHeader{.nbits = 4, .nifs = 1, .nchans = 4}, {});

  expect_read_throws(file.path(), "Unsupported filterbank nbits");
}

TEST(FilterbankErrors, RejectsNonWholeRowDataSize) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_partial_row",
      FixtureHeader{.nbits = 8, .nifs = 1, .nchans = 4}, {1, 2, 3});

  expect_read_throws(file.path(), "whole number of rows");
}

TEST(FilterbankErrors, RejectsFrequencyTableLengthMismatch) {
  const auto file = write_raw_fixture(
      "gaffa_filterbank_frequency_mismatch", [](std::ofstream& output) {
        write_string(output, "HEADER_START");
        write_string(output, "source_name");
        write_string(output, "test-source");
        write_field(output, "nifs", 1);
        write_field(output, "nchans", 3);
        write_field(output, "nbits", 8);
        write_string(output, "FREQUENCY_START");
        write_field(output, "fchannel", 10.0);
        write_string(output, "HEADER_END");
      });

  expect_read_throws(file.path(), "Frequency table length does not match nchans");
}
