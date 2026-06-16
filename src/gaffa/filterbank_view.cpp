#include "gaffa/filterbank_view.h"

#include <type_traits>
#include <variant>

namespace gaffa {

AnyHostSampleView sample_view(const FilterbankData& filterbank) {
  return std::visit(
      [&](const auto& samples) -> AnyHostSampleView {
        using Vector = std::decay_t<decltype(samples)>;
        using T = typename Vector::value_type;
        return sample_view<T>(filterbank);
      },
      filterbank.samples);
}

}  // namespace gaffa
