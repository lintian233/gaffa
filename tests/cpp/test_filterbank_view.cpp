#include "gaffa/filterbank_view.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <variant>
#include <vector>

namespace {

gaffa::FilterbankData make_uint8_filterbank() {
  gaffa::FilterbankHeader header;
  header.nsamples = 2;
  header.nifs = 1;
  header.nchans = 4;
  header.nbits = 8;
  return gaffa::FilterbankData{
      header, std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6, 7, 8}};
}

}  // namespace

TEST(FilterbankView, CreatesTypedHostSampleView) {
  const gaffa::FilterbankData filterbank = make_uint8_filterbank();

  const auto view = gaffa::sample_view<std::uint8_t>(filterbank);

  EXPECT_EQ(view.data.data(),
            std::get<std::vector<std::uint8_t>>(filterbank.samples).data());
  EXPECT_EQ(view.shape.nsamples, 2);
  EXPECT_EQ(view.shape.nifs, 1);
  EXPECT_EQ(view.shape.nchans, 4);
  EXPECT_EQ(view.size(), 8);
  EXPECT_FALSE(view.empty());
}

TEST(FilterbankView, CreatesMutableHostSampleView) {
  gaffa::FilterbankData filterbank = make_uint8_filterbank();

  auto view = gaffa::mutable_sample_view<std::uint8_t>(filterbank);
  view.data[0] = 42;

  EXPECT_EQ(std::get<std::vector<std::uint8_t>>(filterbank.samples).front(), 42);
}

TEST(FilterbankView, CreatesAnyHostSampleView) {
  const gaffa::FilterbankData filterbank = make_uint8_filterbank();

  const gaffa::AnyHostSampleView view = gaffa::sample_view(filterbank);

  ASSERT_TRUE(std::holds_alternative<gaffa::HostSampleView<std::uint8_t>>(view));
  const auto typed_view = std::get<gaffa::HostSampleView<std::uint8_t>>(view);
  EXPECT_EQ(typed_view.size(), 8);
}

TEST(FilterbankView, RejectsDtypeMismatch) {
  const gaffa::FilterbankData filterbank = make_uint8_filterbank();

  EXPECT_THROW((void)gaffa::sample_view<float>(filterbank), std::runtime_error);
}

TEST(FilterbankView, RejectsNegativeShape) {
  gaffa::FilterbankData filterbank = make_uint8_filterbank();
  filterbank.header.nsamples = -1;

  EXPECT_THROW((void)gaffa::sample_view<std::uint8_t>(filterbank),
               std::runtime_error);
}

TEST(FilterbankView, RejectsSampleSizeMismatch) {
  gaffa::FilterbankData filterbank = make_uint8_filterbank();
  filterbank.header.nsamples = 3;

  EXPECT_THROW((void)gaffa::sample_view<std::uint8_t>(filterbank),
               std::runtime_error);
}

TEST(FilterbankView, RejectsShapeOverflow) {
  const gaffa::SampleShape shape{
      std::numeric_limits<std::size_t>::max(), 2, 1};

  EXPECT_THROW((void)gaffa::sample_element_count(shape), std::overflow_error);
}
