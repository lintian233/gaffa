#include "gaffa/dedispersion.h"
#include "gaffa/filterbank.h"
#include "gaffa/filterbank_view.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
gaffa::HostSampleView<T> make_view(const std::vector<T>& samples,
                                   std::size_t nsamples,
                                   std::size_t nchans) {
  return gaffa::make_host_sample_view<T>(
      std::span<const T>(samples),
      gaffa::SampleShape{nsamples, 1, nchans});
}

gaffa::SingleDmDedispersionPlan single_plan(double dm) {
  return gaffa::SingleDmDedispersionPlan{
      .dm = dm,
      .ref_frequency_mhz = 2000.0,
      .tsamp = 0.003111606,
      .chan_begin = 0,
      .chan_end = 3,
  };
}

gaffa::MultiDmDedispersionPlan multi_plan(std::size_t ndm) {
  return gaffa::MultiDmDedispersionPlan{
      .dm_low = 0.0,
      .dm_step = 1.0,
      .ndm = ndm,
      .ref_frequency_mhz = 2000.0,
      .tsamp = 0.003111606,
      .chan_begin = 0,
      .chan_end = 3,
  };
}

std::filesystem::path test_data_path(const std::string& filename) {
  return std::filesystem::path(GAFFA_TEST_DATA_DIR) / filename;
}

}  // namespace

TEST(DedispersedResultView, ExposesRowsWithoutCopying) {
  gaffa::DedispersedResult<float> result{
      .data = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
      .shape = gaffa::DedispersedShape{2, 3},
  };

  const auto view = result.view();
  EXPECT_EQ(view.size(), 6);
  EXPECT_FALSE(view.empty());
  EXPECT_EQ(view.dm_series(0)[0], 1.0F);
  EXPECT_EQ(view.dm_series(0)[2], 3.0F);
  EXPECT_EQ(view.dm_series(1)[0], 4.0F);
  EXPECT_EQ(view.dm_series(1)[2], 6.0F);

  auto mutable_view = result.mutable_view();
  mutable_view.dm_series(1)[1] = 42.0F;
  EXPECT_EQ(result.data[4], 42.0F);
}

TEST(DedispersedResultView, RejectsInvalidRowsAndShape) {
  const std::vector<float> data{1.0F, 2.0F, 3.0F};
  const gaffa::DedispersedResultView<float> mismatched{
      std::span<const float>(data),
      gaffa::DedispersedShape{2, 2},
  };
  EXPECT_THROW((void)mismatched.dm_series(0), std::invalid_argument);

  const gaffa::DedispersedResultView<float> valid{
      std::span<const float>(data),
      gaffa::DedispersedShape{1, 3},
  };
  EXPECT_THROW((void)valid.dm_series(1), std::out_of_range);

  const gaffa::DedispersedResultView<float> overflowing{
      std::span<const float>{},
      gaffa::DedispersedShape{std::numeric_limits<std::size_t>::max(), 2},
  };
  EXPECT_THROW((void)overflowing.size(), std::overflow_error);
}

TEST(DedispersionCpu, SingleDmZeroEqualsChannelSumForUint8) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4, 5, 6};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0};

  const auto result = gaffa::dedisperse_single_dm_cpu(
      make_view(samples, 2, 3), frequency, single_plan(0.0));

  EXPECT_EQ(result.shape.ndm, 1);
  EXPECT_EQ(result.shape.nsamples, 2);
  EXPECT_EQ(result.data, (std::vector<std::uint32_t>{6, 15}));
}

TEST(DedispersionCpu, SingleDmTrimsTailByMaxDelay) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4, 5, 6};
  const std::vector<double> frequency{1000.0, 2000.0, 2000.0};

  const auto result = gaffa::dedisperse_single_dm_cpu(
      make_view(samples, 2, 3), frequency, single_plan(1.0));

  EXPECT_EQ(result.shape.ndm, 1);
  EXPECT_EQ(result.shape.nsamples, 1);
  EXPECT_EQ(result.data, (std::vector<std::uint32_t>{9}));
}

TEST(DedispersionCpu, SpectrumDmZeroCopiesSelectedChannels) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4, 5, 6};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(0.0);
  plan.chan_begin = 1;
  plan.chan_end = 3;

  const auto result =
      gaffa::dedisperse_spectrum_cpu(make_view(samples, 2, 3), frequency, plan);

  EXPECT_EQ(result.shape.nsamples, 2);
  EXPECT_EQ(result.shape.nifs, 1);
  EXPECT_EQ(result.shape.nchans, 2);
  EXPECT_EQ(result.nsamples(), 2);
  EXPECT_EQ(result.nchans(), 2);
  EXPECT_EQ(result.size(), 4);
  EXPECT_EQ(result.dm, 0.0);
  EXPECT_EQ(result.tsamp, plan.tsamp);
  EXPECT_EQ(result.chan_begin, 1);
  EXPECT_EQ(result.chan_end, 3);
  EXPECT_EQ(result.data, (std::vector<std::uint8_t>{2, 3, 5, 6}));
}

TEST(DedispersionCpu, SpectrumTrimsTailAndAlignsChannels) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4, 5, 6};
  const std::vector<double> frequency{1000.0, 2000.0, 2000.0};

  const auto result = gaffa::dedisperse_spectrum_cpu(
      make_view(samples, 2, 3), frequency, single_plan(1.0));

  EXPECT_EQ(result.shape.nsamples, 1);
  EXPECT_EQ(result.shape.nifs, 1);
  EXPECT_EQ(result.shape.nchans, 3);
  EXPECT_EQ(result.data, (std::vector<std::uint8_t>{4, 2, 3}));
}

TEST(DedispersionCpu, SpectrumChannelSumMatchesSingleDm) {
  const std::vector<std::uint8_t> samples{
      1,  2,  3,
      4,  5,  6,
      7,  8,  9,
      10, 11, 12,
  };
  const std::vector<double> frequency{1000.0, 2000.0, 2000.0};

  const auto spectrum = gaffa::dedisperse_spectrum_cpu(
      make_view(samples, 4, 3), frequency, single_plan(1.0));
  const auto single = gaffa::dedisperse_single_dm_cpu(
      make_view(samples, 4, 3), frequency, single_plan(1.0));

  ASSERT_EQ(single.shape.nsamples, spectrum.shape.nsamples);
  ASSERT_EQ(spectrum.shape.nchans, 3);
  for (std::size_t time = 0; time < spectrum.shape.nsamples; ++time) {
    std::uint32_t sum = 0;
    for (std::size_t channel = 0; channel < spectrum.shape.nchans; ++channel) {
      sum += spectrum.data[time * spectrum.shape.nchans + channel];
    }
    EXPECT_EQ(single.data[time], sum);
  }
}

TEST(DedispersionCpu, SpectrumPreservesInputDtype) {
  const std::vector<std::uint16_t> samples_u16{1000, 2000, 3000, 4000};
  const std::vector<float> samples_f32{1.5F, 2.5F, 3.5F, 4.5F};
  const std::vector<double> frequency{1000.0, 2000.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(0.0);
  plan.chan_end = 2;

  const auto result_u16 =
      gaffa::dedisperse_spectrum_cpu(make_view(samples_u16, 2, 2), frequency,
                                     plan);
  const auto result_f32 =
      gaffa::dedisperse_spectrum_cpu(make_view(samples_f32, 2, 2), frequency,
                                     plan);

  EXPECT_EQ(result_u16.data, samples_u16);
  EXPECT_EQ(result_f32.data, samples_f32);
}

TEST(DedispersionCpu, SpectrumRejectsInvalidPlan) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4};
  const std::vector<double> frequency{1000.0, 2000.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(-1.0);
  plan.chan_end = 2;

  EXPECT_THROW((void)gaffa::dedisperse_spectrum_cpu(
                   make_view(samples, 2, 2), frequency, plan),
               std::invalid_argument);
}

TEST(DedispersionCpu, SupportsUint16OutputAsUint32) {
  const std::vector<std::uint16_t> samples{1000, 2000, 3000, 4000};
  const std::vector<double> frequency{1000.0, 2000.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(0.0);
  plan.chan_end = 2;

  const auto result = gaffa::dedisperse_single_dm_cpu(
      make_view(samples, 2, 2), frequency, plan);

  EXPECT_EQ(result.data, (std::vector<std::uint32_t>{3000, 7000}));
}

TEST(DedispersionCpu, SupportsFloatOutputAsFloat) {
  const std::vector<float> samples{1.5F, 2.5F, 3.5F, 4.5F};
  const std::vector<double> frequency{1000.0, 2000.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(0.0);
  plan.chan_end = 2;

  const auto result =
      gaffa::dedisperse_single_dm_cpu(make_view(samples, 2, 2), frequency, plan);

  EXPECT_EQ(result.data, (std::vector<float>{4.0F, 8.0F}));
}

TEST(DedispersionCpu, MultiDmMatchesRepeatedSingleDm) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4, 5, 6};
  const std::vector<double> frequency{1000.0, 2000.0, 2000.0};

  const auto multi = gaffa::dedisperse_multi_dm_cpu(
      make_view(samples, 2, 3), frequency, multi_plan(2));
  const auto dm0 = gaffa::dedisperse_single_dm_cpu(
      make_view(samples, 2, 3), frequency, single_plan(0.0));
  const auto dm1 = gaffa::dedisperse_single_dm_cpu(
      make_view(samples, 2, 3), frequency, single_plan(1.0));

  EXPECT_EQ(multi.shape.ndm, 2);
  EXPECT_EQ(multi.shape.nsamples, 1);
  EXPECT_EQ((std::vector<std::uint32_t>{multi.data[0]}),
            (std::vector<std::uint32_t>{dm0.data[0]}));
  EXPECT_EQ((std::vector<std::uint32_t>{multi.data[1]}), dm1.data);
}

TEST(DedispersionCpu, MultiDmValidLengthUsesLargestDmAndLowestFrequency) {
  const std::vector<std::uint8_t> samples{
      1,  2,  3,
      4,  5,  6,
      7,  8,  9,
      10, 11, 12,
  };
  const std::vector<double> frequency{1000.0, 1500.0, 2000.0};
  const auto result = gaffa::dedisperse_multi_dm_cpu(
      make_view(samples, 4, 3), frequency, multi_plan(3));

  EXPECT_EQ(result.shape.ndm, 3);
  EXPECT_EQ(result.shape.nsamples, 2);
  EXPECT_EQ(result.data.size(), 6);
}

TEST(DedispersionCpu, ChannelRangeIsHalfOpen) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4, 5, 6};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(0.0);
  plan.chan_begin = 1;
  plan.chan_end = 3;

  const auto result =
      gaffa::dedisperse_single_dm_cpu(make_view(samples, 2, 3), frequency, plan);

  EXPECT_EQ(result.data, (std::vector<std::uint32_t>{5, 11}));
}

TEST(DedispersionCpu, SubbandDegenerateMatchesMultiDm) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4, 5, 6};
  const std::vector<double> frequency{1000.0, 2000.0, 2000.0};
  const gaffa::MultiDmDedispersionPlan plan = multi_plan(2);
  const gaffa::SubbandDedispersionOptions options{
      .subband_channels = 1,
      .ndm_per_nominal = 1,
  };

  const auto direct =
      gaffa::dedisperse_multi_dm_cpu(make_view(samples, 2, 3), frequency, plan);
  const auto subband = gaffa::dedisperse_subband_cpu(
      make_view(samples, 2, 3), frequency, plan, options);

  EXPECT_EQ(subband.shape.ndm, direct.shape.ndm);
  EXPECT_EQ(subband.shape.nsamples, direct.shape.nsamples);
  EXPECT_EQ(subband.data, direct.data);
}

TEST(DedispersionCpu, SubbandNormalModeProducesExpectedDmZeroSums) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<double> frequency{1000.0, 1200.0, 1500.0, 2000.0};
  gaffa::MultiDmDedispersionPlan plan = multi_plan(1);
  plan.chan_end = 4;
  const gaffa::SubbandDedispersionOptions options{
      .subband_channels = 2,
      .ndm_per_nominal = 32,
  };

  const auto result = gaffa::dedisperse_subband_cpu(
      make_view(samples, 2, 4), frequency, plan, options);

  EXPECT_EQ(result.shape.ndm, 1);
  EXPECT_EQ(result.shape.nsamples, 2);
  EXPECT_EQ(result.data, (std::vector<std::uint32_t>{10, 26}));
}

TEST(DedispersionCpu, RejectsNifsOtherThanOne) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4};
  const std::vector<double> frequency{1000.0, 2000.0};
  const gaffa::HostSampleView<std::uint8_t> view =
      gaffa::make_host_sample_view<std::uint8_t>(
          std::span<const std::uint8_t>(samples),
          gaffa::SampleShape{1, 2, 2});

  EXPECT_THROW((void)gaffa::dedisperse_single_dm_cpu(view, frequency,
                                                     single_plan(0.0)),
               std::invalid_argument);
}

TEST(DedispersionCpu, RejectsFrequencySizeMismatch) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4};
  const std::vector<double> frequency{1000.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(0.0);
  plan.chan_end = 2;

  EXPECT_THROW((void)gaffa::dedisperse_single_dm_cpu(
                   make_view(samples, 2, 2), frequency, plan),
               std::invalid_argument);
}

TEST(DedispersionCpu, RejectsNullSampleData) {
  const std::vector<double> frequency{1000.0, 2000.0};
  const gaffa::HostSampleView<std::uint8_t> view{
      std::span<const std::uint8_t>{}, gaffa::SampleShape{2, 1, 2}};
  gaffa::SingleDmDedispersionPlan plan = single_plan(0.0);
  plan.chan_end = 2;

  EXPECT_THROW((void)gaffa::dedisperse_single_dm_cpu(view, frequency, plan),
               std::invalid_argument);
}

TEST(DedispersionCpu, RejectsInvalidFrequencyValues) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4};
  const std::vector<double> frequency{
      1000.0, std::numeric_limits<double>::quiet_NaN()};
  gaffa::SingleDmDedispersionPlan plan = single_plan(0.0);
  plan.chan_end = 2;

  EXPECT_THROW((void)gaffa::dedisperse_single_dm_cpu(
                   make_view(samples, 2, 2), frequency, plan),
               std::invalid_argument);
}

TEST(DedispersionCpu, RejectsInvalidMultiDmPlan) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4};
  const std::vector<double> frequency{1000.0, 2000.0};
  gaffa::MultiDmDedispersionPlan plan = multi_plan(0);
  plan.chan_end = 2;

  EXPECT_THROW((void)gaffa::dedisperse_multi_dm_cpu(
                   make_view(samples, 2, 2), frequency, plan),
               std::invalid_argument);
}

TEST(DedispersionCpu, RejectsNegativeSingleDm) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4};
  const std::vector<double> frequency{1000.0, 2000.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(-1.0);
  plan.chan_end = 2;

  EXPECT_THROW((void)gaffa::dedisperse_single_dm_cpu(
                   make_view(samples, 2, 2), frequency, plan),
               std::invalid_argument);
}

TEST(DedispersionCpu, RejectsNegativeMultiDmLow) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4};
  const std::vector<double> frequency{1000.0, 2000.0};
  gaffa::MultiDmDedispersionPlan plan = multi_plan(2);
  plan.dm_low = -1.0;
  plan.chan_end = 2;

  EXPECT_THROW((void)gaffa::dedisperse_multi_dm_cpu(
                   make_view(samples, 2, 2), frequency, plan),
               std::invalid_argument);
}

TEST(DedispersionCpu, RejectsSelectedFrequencyAboveReference) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4};
  const std::vector<double> frequency{1000.0, 2000.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(1.0);
  plan.ref_frequency_mhz = 1500.0;
  plan.chan_end = 2;

  EXPECT_THROW((void)gaffa::dedisperse_single_dm_cpu(
                   make_view(samples, 2, 2), frequency, plan),
               std::invalid_argument);
}

TEST(DedispersionCpu, RejectsEmptyValidOutputRange) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4};
  const std::vector<double> frequency{1000.0, 2000.0};
  gaffa::SingleDmDedispersionPlan plan = single_plan(2.0);
  plan.chan_end = 2;

  EXPECT_THROW((void)gaffa::dedisperse_single_dm_cpu(
                   make_view(samples, 2, 2), frequency, plan),
               std::invalid_argument);
}

TEST(DedispersionCpu, RejectsInvalidSubbandOptions) {
  const std::vector<std::uint8_t> samples{1, 2, 3, 4};
  const std::vector<double> frequency{1000.0, 2000.0};
  gaffa::MultiDmDedispersionPlan plan = multi_plan(1);
  plan.chan_end = 2;
  const gaffa::SubbandDedispersionOptions options{
      .subband_channels = 0,
      .ndm_per_nominal = 1,
  };

  EXPECT_THROW((void)gaffa::dedisperse_subband_cpu(
                   make_view(samples, 2, 2), frequency, plan, options),
               std::invalid_argument);
}

TEST(DedispersionCpu, BaseTestFixtureCpuModesAgree) {
  const std::filesystem::path path = test_data_path("basetest.fil");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "basetest.fil is not available";
  }

  const gaffa::FilterbankData filterbank = gaffa::read_filterbank(path);
  ASSERT_EQ(filterbank.header.nifs, 1);
  ASSERT_EQ(filterbank.header.nbits, 8);

  const auto samples = gaffa::sample_view<std::uint8_t>(filterbank);
  const gaffa::MultiDmDedispersionPlan multi_plan{
      .dm_low = 0.0,
      .dm_step = 1.0,
      .ndm = 10,
      .ref_frequency_mhz = filterbank.header.frequency_table.back(),
      .tsamp = filterbank.header.tsamp,
      .chan_begin = 0,
      .chan_end = static_cast<std::size_t>(filterbank.header.nchans),
  };
  const gaffa::SubbandDedispersionOptions degenerate_subband{
      .subband_channels = 1,
      .ndm_per_nominal = 1,
  };

  const auto multi =
      gaffa::dedisperse_multi_dm_cpu(samples, filterbank.header.frequency_table,
                                     multi_plan);
  const auto subband = gaffa::dedisperse_subband_cpu(
      samples, filterbank.header.frequency_table, multi_plan,
      degenerate_subband);

  ASSERT_EQ(multi.shape.ndm, multi_plan.ndm);
  ASSERT_LE(multi.shape.nsamples, samples.shape.nsamples);
  EXPECT_EQ(subband.shape.ndm, multi.shape.ndm);
  EXPECT_EQ(subband.shape.nsamples, multi.shape.nsamples);
  EXPECT_EQ(subband.data, multi.data);

  for (std::size_t dm_index = 0; dm_index < multi_plan.ndm; ++dm_index) {
    const gaffa::SingleDmDedispersionPlan single_plan{
        .dm = multi_plan.dm_low +
              static_cast<double>(dm_index) * multi_plan.dm_step,
        .ref_frequency_mhz = multi_plan.ref_frequency_mhz,
        .tsamp = multi_plan.tsamp,
        .chan_begin = multi_plan.chan_begin,
        .chan_end = multi_plan.chan_end,
    };
    const auto single = gaffa::dedisperse_single_dm_cpu(
        samples, filterbank.header.frequency_table, single_plan);
    ASSERT_EQ(single.shape.ndm, 1);
    ASSERT_GE(single.shape.nsamples, multi.shape.nsamples);

    const auto begin =
        multi.data.begin() +
        static_cast<std::ptrdiff_t>(dm_index * multi.shape.nsamples);
    const auto end =
        begin + static_cast<std::ptrdiff_t>(multi.shape.nsamples);
    const std::vector<std::uint32_t> single_prefix(
        single.data.begin(),
        single.data.begin() +
            static_cast<std::ptrdiff_t>(multi.shape.nsamples));
    EXPECT_EQ((std::vector<std::uint32_t>{begin, end}), single_prefix);
  }
}
