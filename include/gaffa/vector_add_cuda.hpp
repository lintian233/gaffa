#pragma once

#include <vector>

namespace gaffa {

std::vector<float> vector_add_cuda(const std::vector<float>& lhs, const std::vector<float>& rhs);

}  // namespace gaffa
