#include "gaffa/filterbank.h"
#include "gaffa/filterbank_legacy.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
  std::vector<double> frequency_table;
};

template <typename T>
std::filesystem::path write_fixture(const std::string& name,
                                    const FixtureHeader& header,
                                    const std::vector<T>& samples) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / name;
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to create test fixture");
  }

  write_string(output, "HEADER_START");
  write_string(output, "source_name");
  write_string(output, "test-source");
  write_field(output, "telescope_id", 0);
  write_field(output, "machine_id", 0);
  write_field(output, "data_type", 1);
  write_field(output, "nifs", header.nifs);
  write_field(output, "nchans", header.nchans);
  write_field(output, "nbits", header.nbits);
  write_field(output, "tsamp", 1.0);
  write_field(output, "tstart", 0.0);

  if (header.frequency_table.empty()) {
    write_field(output, "fch1", header.fch1);
    write_field(output, "foff", header.foff);
  } else {
    write_string(output, "FREQUENCY_START");
    for (double frequency : header.frequency_table) {
      write_field(output, "fchannel", frequency);
    }
    write_string(output, "FREQUENCY_END");
  }

  write_string(output, "HEADER_END");
  output.write(reinterpret_cast<const char*>(samples.data()),
               static_cast<std::streamsize>(samples.size() * sizeof(T)));
  return path;
}

template <typename T>
const std::vector<T>& samples_as(const gaffa::FilterbankData& data) {
  return std::get<std::vector<T>>(data.samples);
}

std::filesystem::path test_data_path(const std::string& filename) {
  return std::filesystem::path(GAFFA_TEST_DATA_DIR) / filename;
}

std::size_t legacy_sample_count(const Filterbank& filterbank) {
  return static_cast<std::size_t>(filterbank.nsamples) *
         static_cast<std::size_t>(filterbank.nifs) *
         static_cast<std::size_t>(filterbank.nchans);
}

template <typename T>
void expect_samples_equal_legacy(const Filterbank& legacy,
                                 const gaffa::FilterbankData& data) {
  const auto& samples = std::get<std::vector<T>>(data.samples);
  ASSERT_EQ(samples.size(), legacy_sample_count(legacy));
  ASSERT_NE(legacy.data, nullptr);
  EXPECT_EQ(std::memcmp(samples.data(), legacy.data, samples.size() * sizeof(T)), 0);
}

void expect_data_equal_legacy(const Filterbank& legacy,
                              const gaffa::FilterbankData& data) {
  switch (legacy.nbits) {
  case 8:
    expect_samples_equal_legacy<std::uint8_t>(legacy, data);
    break;
  case 16:
    expect_samples_equal_legacy<std::uint16_t>(legacy, data);
    break;
  case 32:
    expect_samples_equal_legacy<float>(legacy, data);
    break;
  default:
    FAIL() << "Unsupported legacy nbits: " << legacy.nbits;
  }
}

}  // namespace

TEST(Filterbank, ReadsPositiveFoffDirectly) {
  const auto path = write_fixture<std::uint8_t>(
      "gaffa_filterbank_positive_foff.fil",
      FixtureHeader{.nbits = 8, .nifs = 1, .nchans = 4, .fch1 = 1000.0, .foff = 1.0},
      {1, 2, 3, 4, 5, 6, 7, 8});

  const gaffa::FilterbankData data = gaffa::read_filterbank(path);

  EXPECT_EQ(data.header.nsamples, 2);
  EXPECT_EQ(data.header.nchans, 4);
  EXPECT_DOUBLE_EQ(data.header.fch1, 1000.0);
  EXPECT_DOUBLE_EQ(data.header.foff, 1.0);
  EXPECT_EQ((samples_as<std::uint8_t>(data)),
            (std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(Filterbank, NormalizesNegativeFoffToAscendingFrequency) {
  const auto path = write_fixture<std::uint8_t>(
      "gaffa_filterbank_negative_foff.fil",
      FixtureHeader{.nbits = 8, .nifs = 1, .nchans = 4, .fch1 = 1000.0, .foff = -1.0},
      {1, 2, 3, 4, 5, 6, 7, 8});

  gaffa::FilterbankReadOptions options;
  options.io_buffer_bytes = 4;
  const gaffa::FilterbankData data = gaffa::read_filterbank(path, options);

  EXPECT_DOUBLE_EQ(data.header.fch1, 997.0);
  EXPECT_DOUBLE_EQ(data.header.foff, 1.0);
  EXPECT_EQ((data.header.frequency_table),
            (std::vector<double>{997.0, 998.0, 999.0, 1000.0}));
  EXPECT_EQ((samples_as<std::uint8_t>(data)),
            (std::vector<std::uint8_t>{4, 3, 2, 1, 8, 7, 6, 5}));
}

TEST(Filterbank, CanPreserveFileOrder) {
  const auto path = write_fixture<std::uint8_t>(
      "gaffa_filterbank_preserve_order.fil",
      FixtureHeader{.nbits = 8, .nifs = 1, .nchans = 4, .fch1 = 1000.0, .foff = -1.0},
      {1, 2, 3, 4});

  gaffa::FilterbankReadOptions options;
  options.channel_order = gaffa::ChannelOrderPolicy::PreserveFileOrder;
  const gaffa::FilterbankData data = gaffa::read_filterbank(path, options);

  EXPECT_DOUBLE_EQ(data.header.fch1, 1000.0);
  EXPECT_DOUBLE_EQ(data.header.foff, -1.0);
  EXPECT_EQ((data.header.frequency_table),
            (std::vector<double>{1000.0, 999.0, 998.0, 997.0}));
  EXPECT_EQ((samples_as<std::uint8_t>(data)), (std::vector<std::uint8_t>{1, 2, 3, 4}));
}

TEST(Filterbank, ReversesEachIfIndependently) {
  const auto path = write_fixture<std::uint16_t>(
      "gaffa_filterbank_multi_if.fil",
      FixtureHeader{.nbits = 16, .nifs = 2, .nchans = 3, .fch1 = 100.0, .foff = -2.0},
      {1, 2, 3, 10, 11, 12});

  const gaffa::FilterbankData data = gaffa::read_filterbank(path);

  EXPECT_EQ((samples_as<std::uint16_t>(data)),
            (std::vector<std::uint16_t>{3, 2, 1, 12, 11, 10}));
}

TEST(Filterbank, SupportsFloatSamples) {
  const auto path = write_fixture<float>(
      "gaffa_filterbank_float.fil",
      FixtureHeader{.nbits = 32, .nifs = 1, .nchans = 2, .fch1 = 10.0, .foff = -0.5},
      {1.5F, 2.5F});

  const gaffa::FilterbankData data = gaffa::read_filterbank(path);

  EXPECT_EQ((samples_as<float>(data)), (std::vector<float>{2.5F, 1.5F}));
}

TEST(Filterbank, UsesDescendingFrequencyTableForNormalization) {
  const auto path = write_fixture<std::uint8_t>(
      "gaffa_filterbank_frequency_table.fil",
      FixtureHeader{.nbits = 8,
                    .nifs = 1,
                    .nchans = 3,
                    .frequency_table = {30.0, 20.0, 10.0}},
      {7, 8, 9});

  const gaffa::FilterbankData data = gaffa::read_filterbank(path);

  EXPECT_TRUE(data.header.uses_frequency_table);
  EXPECT_DOUBLE_EQ(data.header.fch1, 10.0);
  EXPECT_EQ((data.header.frequency_table), (std::vector<double>{10.0, 20.0, 30.0}));
  EXPECT_EQ((samples_as<std::uint8_t>(data)), (std::vector<std::uint8_t>{9, 8, 7}));
}

TEST(Filterbank, MatchesLegacyReaderOnBaseTestFixture) {
  const std::filesystem::path path = test_data_path("basetest.fil");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "basetest.fil is not available";
  }

  Filterbank legacy(path.string());
  ASSERT_GT(legacy.nchans, 0);
  ASSERT_GT(legacy.nifs, 0);
  ASSERT_GT(legacy.nsamples, 0);
  ASSERT_TRUE(legacy.nbits == 8 || legacy.nbits == 16 || legacy.nbits == 32);
  ASSERT_NE(legacy.data, nullptr);

  const gaffa::FilterbankData data = gaffa::read_filterbank(path);

  EXPECT_EQ(data.header.header_size, legacy.header_size);
  EXPECT_EQ(data.header.nsamples, legacy.nsamples);
  EXPECT_EQ(data.header.telescope_id, legacy.telescope_id);
  EXPECT_EQ(data.header.machine_id, legacy.machine_id);
  EXPECT_EQ(data.header.data_type, legacy.data_type);
  EXPECT_EQ(data.header.barycentric, legacy.barycentric);
  EXPECT_EQ(data.header.pulsarcentric, legacy.pulsarcentric);
  EXPECT_EQ(data.header.ibeam, legacy.ibeam);
  EXPECT_EQ(data.header.nbeams, legacy.nbeams);
  EXPECT_EQ(data.header.nbits, legacy.nbits);
  EXPECT_EQ(data.header.nifs, legacy.nifs);
  EXPECT_EQ(data.header.nchans, legacy.nchans);
  EXPECT_DOUBLE_EQ(data.header.tstart, legacy.tstart);
  EXPECT_DOUBLE_EQ(data.header.tsamp, legacy.tsamp);
  EXPECT_DOUBLE_EQ(data.header.fch1, legacy.fch1);
  EXPECT_DOUBLE_EQ(data.header.foff, legacy.foff);
  EXPECT_DOUBLE_EQ(data.header.refdm, legacy.refdm);
  EXPECT_EQ(data.header.source_name, std::string(legacy.source_name));

  ASSERT_EQ(data.header.frequency_table.size(),
            static_cast<std::size_t>(legacy.nchans));
  for (int channel = 0; channel < legacy.nchans; ++channel) {
    EXPECT_DOUBLE_EQ(data.header.frequency_table[static_cast<std::size_t>(channel)],
                     legacy.frequency_table[channel]);
  }

  expect_data_equal_legacy(legacy, data);
}

TEST(Filterbank, MatchesLegacyReaderOnLarge240MFixture) {
  const std::filesystem::path path = test_data_path("basedata_240M.fil");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "basedata_240M.fil is not available";
  }

  Filterbank legacy(path.string());
  ASSERT_GT(legacy.nchans, 0);
  ASSERT_GT(legacy.nifs, 0);
  ASSERT_GT(legacy.nsamples, 0);
  ASSERT_TRUE(legacy.nbits == 8 || legacy.nbits == 16 || legacy.nbits == 32);
  ASSERT_NE(legacy.data, nullptr);

  const gaffa::FilterbankData data = gaffa::read_filterbank(path);

  EXPECT_EQ(data.header.header_size, legacy.header_size);
  EXPECT_EQ(data.header.nsamples, legacy.nsamples);
  EXPECT_EQ(data.header.telescope_id, legacy.telescope_id);
  EXPECT_EQ(data.header.machine_id, legacy.machine_id);
  EXPECT_EQ(data.header.data_type, legacy.data_type);
  EXPECT_EQ(data.header.barycentric, legacy.barycentric);
  EXPECT_EQ(data.header.pulsarcentric, legacy.pulsarcentric);
  EXPECT_EQ(data.header.ibeam, legacy.ibeam);
  EXPECT_EQ(data.header.nbeams, legacy.nbeams);
  EXPECT_EQ(data.header.nbits, legacy.nbits);
  EXPECT_EQ(data.header.nifs, legacy.nifs);
  EXPECT_EQ(data.header.nchans, legacy.nchans);
  EXPECT_DOUBLE_EQ(data.header.tstart, legacy.tstart);
  EXPECT_DOUBLE_EQ(data.header.tsamp, legacy.tsamp);
  EXPECT_DOUBLE_EQ(data.header.fch1, legacy.fch1);
  EXPECT_DOUBLE_EQ(data.header.foff, legacy.foff);
  EXPECT_DOUBLE_EQ(data.header.refdm, legacy.refdm);
  EXPECT_EQ(data.header.source_name, std::string(legacy.source_name));

  ASSERT_EQ(data.header.frequency_table.size(),
            static_cast<std::size_t>(legacy.nchans));
  for (int channel = 0; channel < legacy.nchans; ++channel) {
    EXPECT_DOUBLE_EQ(data.header.frequency_table[static_cast<std::size_t>(channel)],
                     legacy.frequency_table[channel]);
  }

  expect_data_equal_legacy(legacy, data);
}
