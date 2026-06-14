#include "gaffa/vector_add.hpp"

#include "gaffa/vector_add_cuda.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

namespace gaffa {

std::vector<float> vector_add(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("vector_add requires inputs with the same length");
  }
  if (lhs.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("vector_add input length exceeds CUDA kernel index range");
  }
  if (lhs.empty()) {
    return {};
  }

  return vector_add_cuda(lhs, rhs);
}

}  // namespace gaffa
