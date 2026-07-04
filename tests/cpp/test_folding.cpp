#include "gaffa/folding.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

gaffa::HostSampleView<float> make_view(const std::vector<float>& data,
                                       std::size_t nsamples,
                                       std::size_t nchans) {
  return gaffa::make_host_sample_view<float>(
      std::span<const float>(data),
      gaffa::SampleShape{
          .nsamples = nsamples,
          .nifs = 1,
          .nchans = nchans,
      });
}

template <typename T>
gaffa::HostSampleView<T> make_typed_view(const std::vector<T>& data,
                                         std::size_t nsamples,
                                         std::size_t nchans) {
  return gaffa::make_host_sample_view<T>(
      std::span<const T>(data),
      gaffa::SampleShape{
          .nsamples = nsamples,
          .nifs = 1,
          .nchans = nchans,
      });
}

std::size_t cube_index(const gaffa::FoldedCube& cube, std::size_t subint,
                       std::size_t channel, std::size_t phase) {
  return (subint * cube.nchans + channel) * cube.nbin + phase;
}

gaffa::FoldedProfile reference_fold_time_series(
    const std::vector<float>& input,
    const gaffa::FoldOptions& options) {
  std::vector<double> sums(options.nbin, 0.0);
  std::vector<double> exposure(options.nbin, 0.0);
  const double bin_step =
      options.tsamp / options.period * static_cast<double>(options.nbin);

  for (std::size_t index = 0; index < input.size(); ++index) {
    const double low_scaled =
        static_cast<double>(options.start_sample + index) * bin_step;
    const double high_scaled = low_scaled + bin_step;
    const auto first_bin = static_cast<std::int64_t>(std::floor(low_scaled));
    const auto last_bin =
        static_cast<std::int64_t>(std::ceil(high_scaled) - 1.0);

    for (std::int64_t bin = first_bin; bin <= last_bin; ++bin) {
      const double bin_left = static_cast<double>(bin);
      const double bin_right = bin_left + 1.0;
      const double weight =
          std::min(high_scaled, bin_right) - std::max(low_scaled, bin_left);
      if (!(weight > 0.0)) {
        continue;
      }
      const auto signed_nbin = static_cast<std::int64_t>(options.nbin);
      std::int64_t wrapped = bin % signed_nbin;
      if (wrapped < 0) {
        wrapped += signed_nbin;
      }
      const auto phase = static_cast<std::size_t>(wrapped);
      sums[phase] += static_cast<double>(input[index]) * weight;
      exposure[phase] += weight;
    }
  }

  std::vector<float> profile(options.nbin, 0.0F);
  for (std::size_t phase = 0; phase < options.nbin; ++phase) {
    if (exposure[phase] > 0.0) {
      profile[phase] = static_cast<float>(sums[phase] / exposure[phase]);
    }
  }

  return gaffa::FoldedProfile{
      .profile = std::move(profile),
      .exposure = std::move(exposure),
      .nbin = options.nbin,
  };
}

}  // namespace

TEST(Folding, TimeSeriesAccumulatesSingleBinIntervals) {
  const std::vector<float> input{1.0F, 3.0F};

  const auto folded = gaffa::fold_time_series_cpu(
      input, gaffa::FoldOptions{
                 .period = 1.0,
                 .tsamp = 0.25,
                 .nbin = 2,
             });

  ASSERT_EQ(folded.profile.size(), 2);
  ASSERT_EQ(folded.exposure.size(), 2);
  EXPECT_FLOAT_EQ(folded.profile[0], 2.0F);
  EXPECT_FLOAT_EQ(folded.profile[1], 0.0F);
  EXPECT_DOUBLE_EQ(folded.exposure[0], 1.0);
  EXPECT_DOUBLE_EQ(folded.exposure[1], 0.0);
}

TEST(Folding, TimeSeriesSplitsSamplesAcrossTwoBins) {
  const std::vector<float> input{2.0F};

  const auto folded = gaffa::fold_time_series_cpu(
      input, gaffa::FoldOptions{
                 .period = 1.0,
                 .tsamp = 0.375,
                 .nbin = 4,
             });

  ASSERT_EQ(folded.profile.size(), 4);
  EXPECT_FLOAT_EQ(folded.profile[0], 2.0F);
  EXPECT_FLOAT_EQ(folded.profile[1], 2.0F);
  EXPECT_FLOAT_EQ(folded.profile[2], 0.0F);
  EXPECT_FLOAT_EQ(folded.profile[3], 0.0F);
  EXPECT_DOUBLE_EQ(folded.exposure[0], 1.0);
  EXPECT_DOUBLE_EQ(folded.exposure[1], 0.5);
}

TEST(Folding, TimeSeriesSupportsSamplesSpanningManyBins) {
  const std::vector<float> input{2.0F};

  const auto folded = gaffa::fold_time_series_cpu(
      input, gaffa::FoldOptions{
                 .period = 1.0,
                 .tsamp = 0.75,
                 .nbin = 4,
             });

  ASSERT_EQ(folded.profile.size(), 4);
  EXPECT_FLOAT_EQ(folded.profile[0], 2.0F);
  EXPECT_FLOAT_EQ(folded.profile[1], 2.0F);
  EXPECT_FLOAT_EQ(folded.profile[2], 2.0F);
  EXPECT_FLOAT_EQ(folded.profile[3], 0.0F);
  EXPECT_DOUBLE_EQ(folded.exposure[0], 1.0);
  EXPECT_DOUBLE_EQ(folded.exposure[1], 1.0);
  EXPECT_DOUBLE_EQ(folded.exposure[2], 1.0);
}

TEST(Folding, TimeSeriesConstantInputStaysConstantWhereExposed) {
  const std::vector<float> input(8, 5.0F);

  const auto folded = gaffa::fold_time_series_cpu(
      input, gaffa::FoldOptions{
                 .period = 1.0,
                 .tsamp = 0.125,
                 .nbin = 4,
             });

  ASSERT_EQ(folded.profile.size(), 4);
  for (float value : folded.profile) {
    EXPECT_FLOAT_EQ(value, 5.0F);
  }
  for (double exposure : folded.exposure) {
    EXPECT_DOUBLE_EQ(exposure, 1.0);
  }
}

TEST(Folding, TimeSeriesStartSampleOffsetsPhase) {
  const std::vector<float> input{5.0F};

  const auto folded = gaffa::fold_time_series_cpu(
      input, gaffa::FoldOptions{
                 .period = 1.0,
                 .tsamp = 0.25,
                 .nbin = 4,
                 .start_sample = 1,
             });

  ASSERT_EQ(folded.profile.size(), 4);
  EXPECT_FLOAT_EQ(folded.profile[0], 0.0F);
  EXPECT_FLOAT_EQ(folded.profile[1], 5.0F);
  EXPECT_FLOAT_EQ(folded.profile[2], 0.0F);
  EXPECT_FLOAT_EQ(folded.profile[3], 0.0F);
  EXPECT_DOUBLE_EQ(folded.exposure[1], 1.0);
}

TEST(Folding, TimeSeriesParallelReductionMatchesReference) {
  std::vector<float> input(8192);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) *
                       0.25F +
                   static_cast<float>(index % 5) * 0.1F;
  }
  const gaffa::FoldOptions options{
      .period = 1.0,
      .tsamp = 0.375,
      .nbin = 16,
      .start_sample = 3,
  };

  const auto folded = gaffa::fold_time_series_cpu(input, options);
  const auto expected = reference_fold_time_series(input, options);

  ASSERT_EQ(folded.nbin, expected.nbin);
  ASSERT_EQ(folded.profile.size(), expected.profile.size());
  ASSERT_EQ(folded.exposure.size(), expected.exposure.size());
  for (std::size_t phase = 0; phase < options.nbin; ++phase) {
    EXPECT_NEAR(folded.profile[phase], expected.profile[phase], 1.0e-5F);
    EXPECT_NEAR(folded.exposure[phase], expected.exposure[phase], 1.0e-9);
  }
}

TEST(Folding, TimeSeriesRejectsInvalidInputs) {
  EXPECT_THROW((void)gaffa::fold_time_series_cpu(
                   std::vector<float>{}, gaffa::FoldOptions{
                                             .period = 1.0,
                                             .tsamp = 0.1,
                                             .nbin = 4,
                                         }),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::fold_time_series_cpu(
                   std::vector<float>{1.0F}, gaffa::FoldOptions{
                                                 .period = 0.0,
                                                 .tsamp = 0.1,
                                                 .nbin = 4,
                                             }),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::fold_time_series_cpu(
                   std::vector<float>{NAN}, gaffa::FoldOptions{
                                               .period = 1.0,
                                               .tsamp = 0.1,
                                               .nbin = 4,
                                           }),
               std::invalid_argument);
}

TEST(Folding, SpectrumFoldsIntoSubintChannelPhaseCube) {
  const std::vector<float> samples{
      1.0F, 2.0F,
      3.0F, 4.0F,
      5.0F, 6.0F,
      7.0F, 8.0F,
  };

  const auto cube = gaffa::fold_spectrum_cpu(
      make_view(samples, 4, 2),
      gaffa::FoldSpectrumOptions{
          .period = 1.0,
          .tsamp = 0.25,
          .nbin = 2,
          .subint_samples = 2,
      });

  ASSERT_EQ(cube.nsubint, 2);
  ASSERT_EQ(cube.nchans, 2);
  ASSERT_EQ(cube.nbin, 2);
  ASSERT_EQ(cube.data.size(), 8);
  ASSERT_EQ(cube.exposure.size(), 4);

  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 0, 0)], 2.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 1, 0)], 3.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 0, 1)], 6.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 1, 1)], 7.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 0, 1)], 0.0F);
  EXPECT_DOUBLE_EQ(cube.exposure[0], 1.0);
  EXPECT_DOUBLE_EQ(cube.exposure[1], 0.0);
  EXPECT_DOUBLE_EQ(cube.exposure[2], 0.0);
  EXPECT_DOUBLE_EQ(cube.exposure[3], 1.0);
}

TEST(Folding, SpectrumSupportsUint8InputWithoutConversion) {
  const std::vector<std::uint8_t> samples{
      1, 2,
      3, 4,
      5, 6,
      7, 8,
  };

  const auto cube = gaffa::fold_spectrum_cpu(
      make_typed_view(samples, 4, 2),
      gaffa::FoldSpectrumOptions{
          .period = 1.0,
          .tsamp = 0.25,
          .nbin = 2,
          .subint_samples = 2,
      });

  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 0, 0)], 2.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 1, 0)], 3.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 0, 1)], 6.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 1, 1)], 7.0F);
}

TEST(Folding, SpectrumSupportsUint16InputWithoutConversion) {
  const std::vector<std::uint16_t> samples{
      100, 200,
      300, 400,
  };

  const auto cube = gaffa::fold_spectrum_cpu(
      make_typed_view(samples, 2, 2),
      gaffa::FoldSpectrumOptions{
          .period = 1.0,
          .tsamp = 0.25,
          .nbin = 2,
          .subint_samples = 2,
      });

  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 0, 0)], 200.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 1, 0)], 300.0F);
}

TEST(Folding, SpectrumCanDownsampleFrequencyDuringFold) {
  const std::vector<std::uint8_t> samples{
      1, 3, 10, 14,
      2, 4, 20, 24,
  };

  const auto cube = gaffa::fold_spectrum_cpu(
      make_typed_view(samples, 2, 4),
      gaffa::FoldSpectrumOptions{
          .period = 1.0,
          .tsamp = 0.25,
          .nbin = 2,
          .subint_samples = 2,
          .output_channels = 2,
      });

  ASSERT_EQ(cube.nchans, 2);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 0, 0)], 2.5F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 1, 0)], 17.0F);
}

TEST(Folding, SpectrumParallelSubintsHandleSubbandAndPhaseSplit) {
  const std::vector<std::uint8_t> samples{
      2, 4, 10, 14,
      6, 8, 20, 24,
      10, 12, 30, 34,
      14, 16, 40, 44,
      18, 20, 50, 54,
      22, 24, 60, 64,
      26, 28, 70, 74,
      30, 32, 80, 84,
  };

  const auto cube = gaffa::fold_spectrum_cpu(
      make_typed_view(samples, 8, 4),
      gaffa::FoldSpectrumOptions{
          .period = 1.0,
          .tsamp = 0.375,
          .nbin = 4,
          .subint_samples = 2,
          .output_channels = 2,
      });

  ASSERT_EQ(cube.nsubint, 4);
  ASSERT_EQ(cube.nchans, 2);
  ASSERT_EQ(cube.nbin, 4);

  EXPECT_DOUBLE_EQ(cube.exposure[0], 1.0);
  EXPECT_DOUBLE_EQ(cube.exposure[1], 1.0);
  EXPECT_DOUBLE_EQ(cube.exposure[2], 1.0);
  EXPECT_DOUBLE_EQ(cube.exposure[3], 0.0);
  EXPECT_DOUBLE_EQ(cube.exposure[4], 1.0);
  EXPECT_DOUBLE_EQ(cube.exposure[5], 1.0);
  EXPECT_DOUBLE_EQ(cube.exposure[6], 0.0);
  EXPECT_DOUBLE_EQ(cube.exposure[7], 1.0);

  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 0, 0)], 3.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 0, 1)], 5.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 0, 2)], 7.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 1, 0)], 12.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 1, 1)], 17.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 0, 1, 2)], 22.0F);

  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 0, 0)], 13.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 0, 1)], 15.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 0, 2)], 0.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 0, 3)], 11.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 1, 0)], 37.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 1, 1)], 42.0F);
  EXPECT_FLOAT_EQ(cube.data[cube_index(cube, 1, 1, 3)], 32.0F);
}

TEST(Folding, SpectrumRejectsInvalidInputs) {
  const std::vector<float> samples{1.0F, 2.0F};
  const std::vector<float> four_channel_samples{1.0F, 2.0F, 3.0F, 4.0F};

  EXPECT_THROW((void)gaffa::fold_spectrum_cpu(
                   gaffa::HostSampleView<float>{},
                   gaffa::FoldSpectrumOptions{
                       .period = 1.0,
                       .tsamp = 0.1,
                       .nbin = 4,
                       .subint_samples = 1,
                   }),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::fold_spectrum_cpu(
                   gaffa::make_host_sample_view<float>(
                       std::span<const float>(samples),
                       gaffa::SampleShape{
                           .nsamples = 1,
                           .nifs = 2,
                           .nchans = 1,
                       }),
                   gaffa::FoldSpectrumOptions{
                       .period = 1.0,
                       .tsamp = 0.1,
                       .nbin = 4,
                       .subint_samples = 1,
                   }),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::fold_spectrum_cpu(
                   make_view(samples, 2, 1),
                   gaffa::FoldSpectrumOptions{
                       .period = 1.0,
                       .tsamp = 0.1,
                       .nbin = 4,
                       .subint_samples = 0,
                   }),
               std::invalid_argument);
  EXPECT_THROW((void)gaffa::fold_spectrum_cpu(
                   make_view(four_channel_samples, 1, 4),
                   gaffa::FoldSpectrumOptions{
                       .period = 1.0,
                       .tsamp = 0.1,
                       .nbin = 4,
                       .subint_samples = 1,
                       .output_channels = 3,
                   }),
               std::invalid_argument);
}

TEST(Folding, DedispersedSpectrumFoldProjectsCommonProducts) {
  const gaffa::DedispersedSpectrum<float> spectrum{
      .data = {
          1.0F, 2.0F,
          3.0F, 4.0F,
          5.0F, 6.0F,
          7.0F, 8.0F,
      },
      .shape = gaffa::SampleShape{
          .nsamples = 4,
          .nifs = 1,
          .nchans = 2,
      },
      .tsamp = 0.25,
  };

  const auto folded = gaffa::fold_dedispersed_spectrum_cpu(
      spectrum,
      gaffa::FoldDedispersedSpectrumOptions{
          .period = 1.0,
          .nbin = 2,
          .tsubint = 0.5,
      });

  ASSERT_EQ(folded.cube.nsubint, 2);
  ASSERT_EQ(folded.cube.nchans, 2);
  ASSERT_EQ(folded.cube.nbin, 2);
  EXPECT_EQ(folded.nsubint, 2);
  EXPECT_EQ(folded.nchans, 2);
  EXPECT_EQ(folded.nbin, 2);
  EXPECT_DOUBLE_EQ(folded.period, 1.0);
  EXPECT_DOUBLE_EQ(folded.tsamp, 0.25);
  EXPECT_DOUBLE_EQ(folded.tsubint, 0.5);

  ASSERT_EQ(folded.profile.size(), 2);
  ASSERT_EQ(folded.freq_phase.size(), 4);
  ASSERT_EQ(folded.time_phase.size(), 4);
  EXPECT_FLOAT_EQ(folded.profile[0], 2.5F);
  EXPECT_FLOAT_EQ(folded.profile[1], 6.5F);
  EXPECT_FLOAT_EQ(folded.freq_phase[0], 2.0F);
  EXPECT_FLOAT_EQ(folded.freq_phase[1], 6.0F);
  EXPECT_FLOAT_EQ(folded.freq_phase[2], 3.0F);
  EXPECT_FLOAT_EQ(folded.freq_phase[3], 7.0F);
  EXPECT_FLOAT_EQ(folded.time_phase[0], 2.5F);
  EXPECT_FLOAT_EQ(folded.time_phase[1], 0.0F);
  EXPECT_FLOAT_EQ(folded.time_phase[2], 0.0F);
  EXPECT_FLOAT_EQ(folded.time_phase[3], 6.5F);
}

TEST(Folding, DedispersedSpectrumFoldCanDownsampleFrequencyFirst) {
  const gaffa::DedispersedSpectrum<float> spectrum{
      .data = {
          1.0F, 3.0F, 10.0F, 14.0F,
          2.0F, 4.0F, 20.0F, 24.0F,
      },
      .shape = gaffa::SampleShape{
          .nsamples = 2,
          .nifs = 1,
          .nchans = 4,
      },
      .tsamp = 0.25,
  };

  const auto folded = gaffa::fold_dedispersed_spectrum_cpu(
      spectrum,
      gaffa::FoldDedispersedSpectrumOptions{
          .period = 1.0,
          .nbin = 2,
          .tsubint = 0.5,
          .output_channels = 2,
      });

  ASSERT_EQ(folded.cube.nchans, 2);
  ASSERT_EQ(folded.freq_phase.size(), 4);
  EXPECT_FLOAT_EQ(folded.freq_phase[0], 2.5F);
  EXPECT_FLOAT_EQ(folded.freq_phase[2], 17.0F);
}

TEST(Folding, DedispersedSpectrumFoldSupportsUint8) {
  const gaffa::DedispersedSpectrum<std::uint8_t> spectrum{
      .data = {
          1, 3, 10, 14,
          2, 4, 20, 24,
      },
      .shape = gaffa::SampleShape{
          .nsamples = 2,
          .nifs = 1,
          .nchans = 4,
      },
      .tsamp = 0.25,
  };

  const auto folded = gaffa::fold_dedispersed_spectrum_cpu(
      spectrum,
      gaffa::FoldDedispersedSpectrumOptions{
          .period = 1.0,
          .nbin = 2,
          .tsubint = 0.5,
          .output_channels = 2,
      });

  ASSERT_EQ(folded.cube.nchans, 2);
  EXPECT_FLOAT_EQ(folded.freq_phase[0], 2.5F);
  EXPECT_FLOAT_EQ(folded.freq_phase[2], 17.0F);
}

TEST(Folding, DedispersedSpectrumFoldRejectsInvalidInputs) {
  const gaffa::DedispersedSpectrum<float> spectrum{
      .data = {1.0F, 2.0F, 3.0F},
      .shape = gaffa::SampleShape{
          .nsamples = 2,
          .nifs = 1,
          .nchans = 2,
      },
      .tsamp = 0.25,
  };

  EXPECT_THROW(
      (void)gaffa::fold_dedispersed_spectrum_cpu(
          spectrum,
          gaffa::FoldDedispersedSpectrumOptions{
              .period = 1.0,
              .nbin = 2,
              .tsubint = 0.5,
          }),
      std::invalid_argument);

  const gaffa::DedispersedSpectrum<float> valid{
      .data = {1.0F, 2.0F},
      .shape = gaffa::SampleShape{
          .nsamples = 1,
          .nifs = 1,
          .nchans = 2,
      },
      .tsamp = 0.25,
  };
  EXPECT_THROW(
      (void)gaffa::fold_dedispersed_spectrum_cpu(
          valid,
          gaffa::FoldDedispersedSpectrumOptions{
              .period = 1.0,
              .nbin = 2,
              .tsubint = 0.0,
          }),
      std::invalid_argument);
}
