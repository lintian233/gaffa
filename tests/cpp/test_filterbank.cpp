#include "gaffa/filterbank.h"
#include "support/temp_file.h"

#include <gtest/gtest.h>

#include <cstdint>
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
  double tsamp = 1.0;
  double tstart = 0.0;
  std::string source_name = "test-source";
  std::string rawdatafile;
  std::vector<double> frequency_table;

  bool include_optional_fields = false;
  int barycentric = 0;
  int pulsarcentric = 0;
  int ibeam = 0;
  int nbeams = 0;
  int nbins = 0;
  int npuls = 0;
  double az_start = 0.0;
  double za_start = 0.0;
  double src_raj = 0.0;
  double src_dej = 0.0;
  double period = 0.0;
  double refdm = 0.0;
};

void write_header(std::ofstream& output, const FixtureHeader& header) {
  write_string(output, "HEADER_START");
  if (!header.rawdatafile.empty()) {
    write_string(output, "rawdatafile");
    write_string(output, header.rawdatafile);
  }
  write_string(output, "source_name");
  write_string(output, header.source_name);
  write_field(output, "telescope_id", 0);
  write_field(output, "machine_id", 0);
  write_field(output, "data_type", 1);
  write_field(output, "nifs", header.nifs);
  write_field(output, "nchans", header.nchans);
  write_field(output, "nbits", header.nbits);
  write_field(output, "tsamp", header.tsamp);
  write_field(output, "tstart", header.tstart);

  if (header.include_optional_fields) {
    write_field(output, "barycentric", header.barycentric);
    write_field(output, "pulsarcentric", header.pulsarcentric);
    write_field(output, "ibeam", header.ibeam);
    write_field(output, "nbeams", header.nbeams);
    write_field(output, "nbins", header.nbins);
    write_field(output, "npuls", header.npuls);
    write_field(output, "az_start", header.az_start);
    write_field(output, "za_start", header.za_start);
    write_field(output, "src_raj", header.src_raj);
    write_field(output, "src_dej", header.src_dej);
    write_field(output, "period", header.period);
    write_field(output, "refdm", header.refdm);
    write_field(output, "nsamples", 123456);
  }

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

template <typename T>
const std::vector<T>& samples_as(const gaffa::FilterbankData& data) {
  return std::get<std::vector<T>>(data.samples);
}

}  // namespace

TEST(Filterbank, ReadsPositiveFoffDirectly) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_positive_foff",
      FixtureHeader{.nbits = 8,
                    .nifs = 1,
                    .nchans = 4,
                    .fch1 = 1000.0,
                    .foff = 1.0},
      {1, 2, 3, 4, 5, 6, 7, 8});

  const gaffa::FilterbankData data = gaffa::read_filterbank(file.path());

  EXPECT_EQ(data.header.nsamples, 2);
  EXPECT_EQ(data.header.nchans, 4);
  EXPECT_DOUBLE_EQ(data.header.fch1, 1000.0);
  EXPECT_DOUBLE_EQ(data.header.foff, 1.0);
  EXPECT_EQ((samples_as<std::uint8_t>(data)),
            (std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(Filterbank, NormalizesNegativeFoffToAscendingFrequency) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_negative_foff",
      FixtureHeader{.nbits = 8,
                    .nifs = 1,
                    .nchans = 4,
                    .fch1 = 1000.0,
                    .foff = -1.0},
      {1, 2, 3, 4, 5, 6, 7, 8});

  gaffa::FilterbankReadOptions options;
  options.io_buffer_bytes = 4;
  const gaffa::FilterbankData data =
      gaffa::read_filterbank(file.path(), options);

  EXPECT_DOUBLE_EQ(data.header.fch1, 997.0);
  EXPECT_DOUBLE_EQ(data.header.foff, 1.0);
  EXPECT_EQ((data.header.frequency_table),
            (std::vector<double>{997.0, 998.0, 999.0, 1000.0}));
  EXPECT_EQ((samples_as<std::uint8_t>(data)),
            (std::vector<std::uint8_t>{4, 3, 2, 1, 8, 7, 6, 5}));
}

TEST(Filterbank, CanPreserveFileOrder) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_preserve_order",
      FixtureHeader{.nbits = 8,
                    .nifs = 1,
                    .nchans = 4,
                    .fch1 = 1000.0,
                    .foff = -1.0},
      {1, 2, 3, 4});

  gaffa::FilterbankReadOptions options;
  options.channel_order = gaffa::ChannelOrderPolicy::PreserveFileOrder;
  const gaffa::FilterbankData data =
      gaffa::read_filterbank(file.path(), options);

  EXPECT_DOUBLE_EQ(data.header.fch1, 1000.0);
  EXPECT_DOUBLE_EQ(data.header.foff, -1.0);
  EXPECT_EQ((data.header.frequency_table),
            (std::vector<double>{1000.0, 999.0, 998.0, 997.0}));
  EXPECT_EQ((samples_as<std::uint8_t>(data)),
            (std::vector<std::uint8_t>{1, 2, 3, 4}));
}

TEST(Filterbank, ReversesEachIfIndependently) {
  const auto file = write_fixture<std::uint16_t>(
      "gaffa_filterbank_multi_if",
      FixtureHeader{.nbits = 16,
                    .nifs = 2,
                    .nchans = 3,
                    .fch1 = 100.0,
                    .foff = -2.0},
      {1, 2, 3, 10, 11, 12});

  const gaffa::FilterbankData data = gaffa::read_filterbank(file.path());

  EXPECT_EQ((samples_as<std::uint16_t>(data)),
            (std::vector<std::uint16_t>{3, 2, 1, 12, 11, 10}));
}

TEST(Filterbank, SupportsFloatSamples) {
  const auto file = write_fixture<float>(
      "gaffa_filterbank_float",
      FixtureHeader{.nbits = 32,
                    .nifs = 1,
                    .nchans = 2,
                    .fch1 = 10.0,
                    .foff = -0.5},
      {1.5F, 2.5F});

  const gaffa::FilterbankData data = gaffa::read_filterbank(file.path());

  EXPECT_EQ((samples_as<float>(data)), (std::vector<float>{2.5F, 1.5F}));
}

TEST(Filterbank, UsesDescendingFrequencyTableForNormalization) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_frequency_table",
      FixtureHeader{.nbits = 8,
                    .nifs = 1,
                    .nchans = 3,
                    .frequency_table = {30.0, 20.0, 10.0}},
      {7, 8, 9});

  const gaffa::FilterbankData data = gaffa::read_filterbank(file.path());

  EXPECT_TRUE(data.header.uses_frequency_table);
  EXPECT_DOUBLE_EQ(data.header.fch1, 10.0);
  EXPECT_EQ((data.header.frequency_table),
            (std::vector<double>{10.0, 20.0, 30.0}));
  EXPECT_EQ((samples_as<std::uint8_t>(data)),
            (std::vector<std::uint8_t>{9, 8, 7}));
}

TEST(Filterbank, PreservesAscendingFrequencyTable) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_ascending_frequency_table",
      FixtureHeader{.nbits = 8,
                    .nifs = 1,
                    .nchans = 3,
                    .frequency_table = {10.0, 20.0, 30.0}},
      {7, 8, 9});

  const gaffa::FilterbankData data = gaffa::read_filterbank(file.path());

  EXPECT_TRUE(data.header.uses_frequency_table);
  EXPECT_DOUBLE_EQ(data.header.fch1, 0.0);
  EXPECT_DOUBLE_EQ(data.header.foff, 0.0);
  EXPECT_EQ((data.header.frequency_table),
            (std::vector<double>{10.0, 20.0, 30.0}));
  EXPECT_EQ((samples_as<std::uint8_t>(data)),
            (std::vector<std::uint8_t>{7, 8, 9}));
}

TEST(Filterbank, ParsesRawdatafileAndOptionalHeaderFields) {
  FixtureHeader header;
  header.rawdatafile = "raw.dat";
  header.include_optional_fields = true;
  header.barycentric = 1;
  header.pulsarcentric = 2;
  header.ibeam = 3;
  header.nbeams = 4;
  header.nbins = 5;
  header.npuls = 6;
  header.az_start = 11.5;
  header.za_start = 12.5;
  header.src_raj = 13.5;
  header.src_dej = 14.5;
  header.period = 15.5;
  header.refdm = 16.5;

  const auto file =
      write_fixture<std::uint8_t>("gaffa_filterbank_optional_fields", header,
                                  {1, 2, 3, 4});

  const gaffa::FilterbankData data = gaffa::read_filterbank(file.path());

  EXPECT_EQ(data.header.rawdatafile, "raw.dat");
  EXPECT_EQ(data.header.source_name, "test-source");
  EXPECT_EQ(data.header.barycentric, 1);
  EXPECT_EQ(data.header.pulsarcentric, 2);
  EXPECT_EQ(data.header.ibeam, 3);
  EXPECT_EQ(data.header.nbeams, 4);
  EXPECT_EQ(data.header.nbins, 5);
  EXPECT_EQ(data.header.npuls, 6);
  EXPECT_DOUBLE_EQ(data.header.az_start, 11.5);
  EXPECT_DOUBLE_EQ(data.header.za_start, 12.5);
  EXPECT_DOUBLE_EQ(data.header.src_raj, 13.5);
  EXPECT_DOUBLE_EQ(data.header.src_dej, 14.5);
  EXPECT_DOUBLE_EQ(data.header.period, 15.5);
  EXPECT_DOUBLE_EQ(data.header.refdm, 16.5);
  EXPECT_EQ(data.header.nsamples, 1);
}

TEST(Filterbank, CpuScalarReverseMatchesAuto) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_scalar_backend",
      FixtureHeader{.nbits = 8,
                    .nifs = 1,
                    .nchans = 4,
                    .fch1 = 1000.0,
                    .foff = -1.0},
      {1, 2, 3, 4, 5, 6, 7, 8});

  const gaffa::FilterbankData auto_data = gaffa::read_filterbank(file.path());
  gaffa::FilterbankReadOptions scalar_options;
  scalar_options.reverse_backend = gaffa::ReverseBackend::CpuScalar;
  const gaffa::FilterbankData scalar_data =
      gaffa::read_filterbank(file.path(), scalar_options);

  EXPECT_EQ((samples_as<std::uint8_t>(scalar_data)),
            (samples_as<std::uint8_t>(auto_data)));
  EXPECT_EQ(scalar_data.header.frequency_table, auto_data.header.frequency_table);
}

TEST(Filterbank, CpuOpenmpReverseMatchesAuto) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_openmp_backend",
      FixtureHeader{.nbits = 8,
                    .nifs = 1,
                    .nchans = 4,
                    .fch1 = 1000.0,
                    .foff = -1.0},
      {1, 2, 3, 4, 5, 6, 7, 8});

  const gaffa::FilterbankData auto_data = gaffa::read_filterbank(file.path());
  gaffa::FilterbankReadOptions openmp_options;
  openmp_options.reverse_backend = gaffa::ReverseBackend::CpuOpenmp;
  const gaffa::FilterbankData openmp_data =
      gaffa::read_filterbank(file.path(), openmp_options);

  EXPECT_EQ((samples_as<std::uint8_t>(openmp_data)),
            (samples_as<std::uint8_t>(auto_data)));
  EXPECT_EQ(openmp_data.header.frequency_table, auto_data.header.frequency_table);
}

TEST(Filterbank, TinyIoBufferStillReadsAllRows) {
  const auto file = write_fixture<std::uint8_t>(
      "gaffa_filterbank_tiny_buffer",
      FixtureHeader{.nbits = 8,
                    .nifs = 1,
                    .nchans = 4,
                    .fch1 = 1000.0,
                    .foff = -1.0},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  gaffa::FilterbankReadOptions options;
  options.io_buffer_bytes = 1;
  const gaffa::FilterbankData data =
      gaffa::read_filterbank(file.path(), options);

  EXPECT_EQ((samples_as<std::uint8_t>(data)),
            (std::vector<std::uint8_t>{4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9}));
}
