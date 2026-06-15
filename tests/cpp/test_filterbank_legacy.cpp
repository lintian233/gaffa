#include "gaffa/filterbank.h"
#include "gaffa/filterbank_legacy.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

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
  EXPECT_EQ(std::memcmp(samples.data(), legacy.data, samples.size() * sizeof(T)),
            0);
}

void expect_header_equal_legacy(const Filterbank& legacy,
                                const gaffa::FilterbankData& data) {
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
    EXPECT_DOUBLE_EQ(
        data.header.frequency_table[static_cast<std::size_t>(channel)],
        legacy.frequency_table[channel]);
  }
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

void expect_filterbank_equal_legacy(const Filterbank& legacy,
                                    const gaffa::FilterbankData& data) {
  ASSERT_GT(legacy.nchans, 0);
  ASSERT_GT(legacy.nifs, 0);
  ASSERT_GT(legacy.nsamples, 0);
  ASSERT_TRUE(legacy.nbits == 8 || legacy.nbits == 16 || legacy.nbits == 32);
  ASSERT_NE(legacy.data, nullptr);

  expect_header_equal_legacy(legacy, data);
  expect_data_equal_legacy(legacy, data);
}

}  // namespace

TEST(FilterbankLegacy, MatchesBaseTestFixture) {
  const std::filesystem::path path = test_data_path("basetest.fil");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "basetest.fil is not available";
  }

  Filterbank legacy(path.string());
  const gaffa::FilterbankData data = gaffa::read_filterbank(path);

  expect_filterbank_equal_legacy(legacy, data);
}

TEST(FilterbankLegacy, MatchesLarge240MFixture) {
  const std::filesystem::path path = test_data_path("basedata_240M.fil");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "basedata_240M.fil is not available";
  }

  Filterbank legacy(path.string());
  const gaffa::FilterbankData data = gaffa::read_filterbank(path);

  expect_filterbank_equal_legacy(legacy, data);
}
